import os
import random
import pickle
import time
import logging
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

import torch
import torch.nn as nn
from data_utils import load_vocab, load_data, select_features

from global_graph.utils import mask_test_edges, sparse_to_tuple, sparse_mx_to_torch_sparse_tensor, normalize_adj
from global_graph.metrics import get_roc_score
from global_graph.model_hier import HierGlobalGCN
from global_graph.utils import save_checkpoint, load_checkpoint
from utils import create_logger, get_available_pretrain_methods, get_available_data_types
from features import get_features_generator, get_available_features_generators

from feature_match import CMD

from tensorboardX import SummaryWriter
import warnings
warnings.simplefilter(action='ignore') #, category=FutureWarning

# np.set_printoptions(threshold=10e6)
# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=1000000)

def train(args, model, features, adj_norm, adj_tensor, drug_nums, adj_label, adj_mask, optimizer, pred_loss, om_loss, fm_loss, epoch): #summary_writer, 
    model.train()
    optimizer.zero_grad()

    # embedding
    outputs_g, outputs_l, local_embed, global_embed = model(features, adj_norm, adj_tensor, drug_nums)

    labels = adj_label.to_dense().view(-1)
    if args.molecular_graph:
        total_loss = torch.mean(pred_loss(outputs_l, labels) * adj_mask)
    elif args.no_fmom:
        tv_loss = torch.mean(pred_loss(outputs_g, labels) * adj_mask)
        bv_loss = torch.mean(pred_loss(outputs_l, labels) * adj_mask)
        total_loss = tv_loss + bv_loss
    elif args.no_fm:
        tv_loss = torch.mean(pred_loss(outputs_g, labels) * adj_mask)
        bv_loss = torch.mean(pred_loss(outputs_l, labels) * adj_mask)
        omloss = torch.mean(om_loss(torch.log(outputs_g), outputs_l) * adj_mask)
        total_loss = tv_loss + bv_loss + omloss
    else:
        tv_loss = torch.mean(pred_loss(outputs_g, labels) * adj_mask)
        bv_loss = torch.mean(pred_loss(outputs_l, labels) * adj_mask)
        omloss = torch.mean(om_loss(torch.log(outputs_g), outputs_l) * adj_mask)   
        fmloss = fm_loss(local_embed, global_embed)
        total_loss = args.alpha_loss * tv_loss + args.beta_loss * bv_loss + args.gamma_loss * omloss + args.theta_loss * fmloss
        
    total_loss.backward()
    if args.vocab_path == 'datachem/drug_list_deep.csv':
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
    optimizer.step()

    return total_loss
        
def valid(model, features, adj_norm, adj_orig, adj_tensor, drug_nums, val_edges, val_edges_false, epoch):  
    model.eval()
    with torch.no_grad():
        val_roc, val_ap, val_f1, val_acc, _, _ = get_roc_score(model, features, adj_norm, adj_orig, adj_tensor, drug_nums, val_edges, val_edges_false)

    return val_roc, val_ap, val_f1, val_acc

def run(args, params):
    ###############################
    #    logging & Parameters 
    ###############################    
    logging_file_path = os.path.join(args.model_result_path, args.logging_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    p_stream = logging.StreamHandler()
    f_stream = logging.FileHandler(logging_file_path, mode='a', encoding='utf-8')
    p_stream.setFormatter(formatter)
    f_stream.setFormatter(formatter)
    logger.addHandler(p_stream)
    logger.addHandler(f_stream)    
    logger.setLevel('INFO')  

    logger.info(f'Job Script: {sys.argv[0]}')
    logger.info(f'Job Arguments:\n {args}')

    ###############################
    #       Seed value
    ###############################    
    seed_val = args.seed
    logger.info(f'random seed number: {seed_val}')
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True

    #####################################
    #            Get Data
    #####################################

    if args.vocab_path is not None:
        smiles2idx = load_vocab(args.vocab_path) 
    else: 
        smiles2idx = None 

    if smiles2idx is not None:
        idx2smiles = [''] * len(smiles2idx)
        for smiles, smiles_idx in smiles2idx.items():
            idx2smiles[smiles_idx] = smiles 
    else:
        idx2smiles = None


    adj, adj_train, adj_train_false, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_data(args, 
                                                                                                                                          filepath=args.data_path, 
                                                                                                                                          smiles2idx=smiles2idx
                                                                                                                                          )
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    logger.info('Number of nodes: {}, number of edges: {}'.format(num_nodes, num_edges))

    if args.use_input_features:
        features_orig, features_generated = select_features(args, idx2smiles, num_nodes)
    else:
        features_orig = select_features(args, idx2smiles, num_nodes)
        
    #####################################
    #   Get Train and Validate Data
    #####################################


    num_features = args.hidden_size 
    args.num_features = num_features 
    features_nonzero = 0
    args.features_nonzero = features_nonzero

    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()
    adj = adj_train

    num_edges_w = adj.sum()
    args.num_edges_w = num_edges_w
    num_nodes_w = adj.shape[0]
    pos_weight = float(num_nodes_w ** 2 - num_edges_w) / num_edges_w

    adj_tensor = torch.FloatTensor(adj.toarray())
    drug_nums = adj.toarray().shape[0]
    args.drug_nums = drug_nums

    adj_norm = normalize_adj(adj)

    adj_label = adj_train
    adj_mask = pos_weight * adj_train.toarray() + adj_train_false.toarray()
    adj_mask_un = adj.toarray() - adj_train.toarray() - adj_train_false.toarray()
    adj_mask = torch.flatten(torch.Tensor(adj_mask))
    adj_mask_un = torch.flatten(torch.Tensor(adj_mask_un))

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_label)
    features = features_orig

    #####################################
    #           Set Up Model
    #####################################
    logger.info(f"Get model")
    model = HierGlobalGCN(args, num_features, features_nonzero,
                          dropout=args.dropout,
                          bias=args.bias,
                          sparse=False
                          )
 
    logger.info('Get loss')
    pred_loss = nn.BCEWithLogitsLoss(reduction='none')
    kld_loss = nn.KLDivLoss(reduction='none')
    cmd_loss = CMD(args.cmd_k)

    logger.info('Get optimizer')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # Set Device
    if (args.use_gpu and torch.cuda.is_available()):
        args.cuda = True
        torch.cuda.empty_cache()
        torch.cuda.set_device(args.gpu_id)

        adj_norm = adj_norm.cuda()
        adj_label = adj_label.cuda()
        adj_tensor = adj_tensor.cuda()
        adj_mask = adj_mask.cuda()
        adj_mask_un = adj_mask_un.cuda()
        model.cuda()
        logger.info(f"data and model to gpu")

    #####################################
    #        Train and Validation
    #####################################   
    best_score = 0.0
    best_epoch = 0

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        total_loss = train(args, model, 
                           features, adj_norm, adj_tensor, drug_nums, adj_label, adj_mask, 
                           optimizer, pred_loss, kld_loss, cmd_loss, epoch
                           ) #
        
        val_roc, val_ap, val_f1, val_acc = valid(model, features, 
                                                 adj_norm, adj_orig, 
                                                 adj_tensor, drug_nums,
                                                 val_edges, val_edges_false, epoch
                                                 ) 
        logger.info('Epoch: {} train_loss= {:.4f} val_roc= {:.4f} val_ap= {:.4f}, val_f1= {:.4f}, val_acc={:.4f}'.format(
            epoch + 1, total_loss, val_roc, val_ap, val_f1, val_acc))
        
        if val_roc > best_score:
            best_score = val_roc
            best_epoch = epoch
            if args.model_result_path:
                save_checkpoint(os.path.join(args.model_result_path, 'model.pt'), model, args)

    logger.info(f'Model train and validation process complete')

    #####################################
    #       Test
    #####################################   
    
    if args.model_result_path:
        logger.info(f'Model best validation roc_auc = {best_score:.6f} on epoch {best_epoch}')
        test_model = load_checkpoint(os.path.join(args.model_result_path, 'model.pt'), cuda=args.use_gpu, logger=logger)

    test_roc, tet_ap, test_f1, test_acc, test_preds_all, test_labels_all = get_roc_score(test_model, features, 
                                                                                         adj_norm, adj_orig, 
                                                                                         adj_tensor, drug_nums, 
                                                                                         test_edges, test_edges_false, 
                                                                                         test = True
                                                                                        )

    test_res_dict = {'preds_all':test_preds_all, 'labels_all':test_labels_all}
    with open(os.path.join(args.model_result_path, 'test_res.pickle'), 'wb') as f:
        pickle.dump(test_res_dict, f)

    logger.info('Test ROC: {:.5f}'.format(test_roc))
    logger.info('Test AP: {:.5f}'.format(tet_ap))
    logger.info('Test F1: {:.5f}'.format(test_f1))
    logger.info('Test ACC: {:.5f}'.format(test_acc))

    logger.info(f'Model test complete')



 


