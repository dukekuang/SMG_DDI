import argparse
import os
import sys
import json

import experiment
from utils import create_logger, get_available_pretrain_methods, get_available_data_types
from features import get_features_generator, get_available_features_generators


def get_args():
    
    args_parser = argparse.ArgumentParser(description="Arguments for train and test")
    
    # data
    args_parser.add_argument('--data_path', type=str, default='datachem/ZhangDDI_train.csv')
    args_parser.add_argument('--separate_val_path', type=str, default='datachem/ZhangDDI_valid.csv')
    args_parser.add_argument('--separate_test_path', type=str, default='datachem/ZhangDDI_test.csv')
    args_parser.add_argument('--vocab_path', type=str, default='datachem/drug_list_zhang.csv', choices=['datachem/drug_list_zhang.csv', 'datachem/drug_list_miner.csv', 'datachem/drug_list_deep.csv'])

    # train optimizer
    args_parser.add_argument('--learning_rate', type=float, default=0.001)
    args_parser.add_argument('--epochs', type=int, default=250)
    args_parser.add_argument('--weight_decay', type=float, default=0)
    args_parser.add_argument('--use_gpu', action='store_true', default=False)
    args_parser.add_argument('--gpu_id', type=int, default=0)

    # model
    args_parser.add_argument('--hidden_size', type=int, default=300, help='Dimensionality of hidden layers in MPN')
    args_parser.add_argument('--alpha', type=float, default=0.2)
    args_parser.add_argument('--num_heads1', type=int, default=1)
    args_parser.add_argument('--num_heads2', type=int, default=3)

    args_parser.add_argument('--depth', type=int, default=3, help='Number of message passing steps')
    args_parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'], help='Activation function') 
    args_parser.add_argument('--undirected', action='store_true', default=False, help='Undirected edges (always sum the two relevant bond vectors)')
    args_parser.add_argument('--output_size', type=int, default=1, help='output dim for higher-capacity FFN')
    args_parser.add_argument('--ffn_hidden_size', type=int, default=264, help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    args_parser.add_argument('--ffn_num_layers', type=int, default=3, help='Number of layers in FFN after MPN encoding')
    args_parser.add_argument('--atom_messages', action='store_true', default=False, help='Use messages on atoms instead of messages on bonds')
    args_parser.add_argument('--features_only', action='store_true', default=False, help='Use only the additional features in an FFN, no graph network')
    args_parser.add_argument('--weight_tying', action='store_false', default=True)
    args_parser.add_argument('--no_cache', action='store_true', default=False, help='Turn off caching mol2graph computation')
    args_parser.add_argument('--attn_output', action='store_true', default=True)
    args_parser.add_argument('--attn_num_d', type=int, default=30, help='Number of units in attention weight parameters 1.')
    args_parser.add_argument('--attn_num_r', type=int, default=10, help='Number of units in attention weight parameters 2.')

    args_parser.add_argument('--FF_hidden1', type=int, default=300, help='Number of units in FF hidden layer 1.')
    args_parser.add_argument('--FF_hidden2', type=int, default=281, help='Number of units in FF hidden layer 2.')
    args_parser.add_argument('--FF_output', type=int, default=264, help='Number of units in FF hidden layer 3.')
    args_parser.add_argument('--gcn_hidden1', type=int, default=300, help='Number of units in GCN hidden layer 1.')
    args_parser.add_argument('--gcn_hidden2', type=int, default=281, help='Number of units in GCN hidden layer 2.')
    args_parser.add_argument('--gcn_hidden3', type=int, default=264, help='Number of units in GCN hidden layer 3.')
    args_parser.add_argument('--bias', action='store_true', default=True)
    args_parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    args_parser.add_argument('--clip', type=float, default=1.0, help='clip coefficient')

    args_parser.add_argument('--smiles_based', action='store_true', default=False)
    args_parser.add_argument('--pooling', type=str, choices=['max', 'sum', 'lstm'], default='sum')
    args_parser.add_argument('--emb_size', type=int, default=None)

    args_parser.add_argument('--seq_len', type=int, default=60)
    args_parser.add_argument('--radius', type=int, default=1)
    args_parser.add_argument('--data_type', type=str, choices=get_available_data_types(), default='small')
    args_parser.add_argument('--min_freq', type=int, default=3)
    args_parser.add_argument('--num_edges_w', type=float, default=7000)
    args_parser.add_argument('--alpha_loss', type=float, default=1, help='intra-view prediction loss')
    args_parser.add_argument('--beta_loss', type=float, default=0.5, help='inter-view prediction loss')
    args_parser.add_argument('--gamma_loss', type=float, default=1, help='output space matching loss')
    args_parser.add_argument('--theta_loss', type=float, default=1, help='feature matching loss')
    args_parser.add_argument('--cmd_k', default=5, type=int, help='cmd moments')
    args_parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=True)

    # Pretrain model
    args_parser.add_argument('--graph_encoder', type=str, default='dmpnn', choices=['dmpnn', 'ggnn', 'gin', 'gcn', 'graphsage'])
    args_parser.add_argument('--graph_encoder_no_pretrain', action='store_true', default=False, help='no pretrain weight for gin, gcn and graphsage')
    args_parser.add_argument('--pretrained_gnn_path', type=str, default='./pretrain/pretrain_gin_contextpred_seed_0.pth', help='The path of the pretrained gnn.')
    args_parser.add_argument('--jt', action='store_true', default=False, help='only use junction tree (default: false)')
    args_parser.add_argument('--pretrain', type=str, choices=get_available_pretrain_methods(), default='mol2vec')
    args_parser.add_argument('--pretrain_path', type=str, default=None)

    # Save result
    args_parser.add_argument('--model_result_path', help='model result folder path', type=str, default='mycode_res/zhang_ddi')
    args_parser.add_argument('--quiet', action='store_true', default=False, help='Skip non-essential print statements')   
    args_parser.add_argument('--use_input_features_generator', type=str, default=None, choices=get_available_features_generators())   
    args_parser.add_argument('--input_features_size', type=int, default=0, help='Number of input features dimension.')
    args_parser.add_argument('--best_score_path', type=str, default=None, help='help to record the best score in files.')
    args_parser.add_argument('--seed', type=int, default=10)

    args_parser.add_argument('--molecular_graph', action='store_true', default=False, help='use molecular graph only')
    args_parser.add_argument('--get_initial_drugpair_embedding', help='whether to get initial drug pair embedding', default=False)
    args_parser.add_argument('--get_final_drugpair_embedding', help='whether to get final drug pair embedding', default=False)
    args_parser.add_argument('--get_initial_drugpair_embedding_save_file', help='path to save initial drug pair embedding', type=str,  default='./embeddings/initial')
    args_parser.add_argument('--get_final_drugpair_embedding_save_file', help='path to save final drug pair embedding', type=str,  default='./embeddings/final')

    args_parser.add_argument('--logging_file', help='logger file name', type=str, default='train_valid_test.log')
    args_parser.add_argument('--params_file', help='model hyper-parameter file.', type=str, default='params.json')

    args_parser.add_argument('--no_fmom', action='store_true', default=False, help='no distribution matching')
    args_parser.add_argument('--no_fm', action='store_true', default=False, help='no outputspace matching')

    args = args_parser.parse_args()
    if args.vocab_path == 'datachem/drug_list_deep.csv':
        args.learning_rate = 0.0001

    args.use_input_features = args.use_input_features_generator in get_available_features_generators()
    if args.use_input_features:
        args.input_features_size = 512
    
    return args


def main():
    args = get_args()
    params = {}

    if not os.path.exists(args.model_result_path):
        os.makedirs(args.model_result_path)
        print(f'Create {args.model_result_path}')

    experiment.run(args, params)

if __name__ == '__main__':
    main()