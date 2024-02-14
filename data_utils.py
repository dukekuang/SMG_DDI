import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Union, Tuple, List, Dict
from argparse import Namespace
from features import get_features_generator

def load_vocab(filepath: str):
    df = pd.read_csv(filepath, index_col=False)
    smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    return smiles2id

def load_csv_data(filepath: str, smiles2id: dict, is_train_file: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get the edge for the inter-view graph. 
    Return "edges" and "edges false"
    """
    df = pd.read_csv(filepath, index_col=False)

    edges = []  # interaction
    edges_false = []    # no interaction
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2id.keys() and smiles_2 in smiles2id.keys():
            idx_1 = smiles2id[smiles_1]
            idx_2 = smiles2id[smiles_2]
            label = int(row_dict['label'])
        else:
            continue
        if label > 0:
            edges.append((idx_1, idx_2))
            edges.append((idx_2, idx_1))
        else:
            edges_false.append((idx_1, idx_2))
            edges_false.append((idx_2, idx_1))

    if is_train_file:
        edges = np.array(edges, dtype=np.int_)
        edges_false = np.array(edges_false, dtype=np.int_)
        return edges, edges_false
    else:
        edges = np.array(edges, dtype=np.int_)
        edges_false = np.array(edges_false, dtype=np.int_)
        return edges, edges_false

# focus on the output tuple, the adj is sparse
def load_data(args: Namespace, filepath: str, smiles2idx: dict = None) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext = os.path.splitext(filepath)[-1] 
    if args.separate_val_path is not None and args.separate_test_path is not None and ext == '.csv':
        """
        .csv file can only provide (node1, node2) edges
        1. load vocab file
        2. load edges
        3. construct adj 
        """
        assert smiles2idx is not None
        num_nodes = len(smiles2idx)
        train_edges, train_edges_false = load_csv_data(filepath, smiles2idx, is_train_file=True)
        val_edges, val_edges_false = load_csv_data(args.separate_val_path, smiles2idx, is_train_file=False)
        test_edges, test_edges_false = load_csv_data(args.separate_test_path, smiles2idx, is_train_file=False)

        all_edges = np.concatenate([train_edges, val_edges, test_edges], axis=0)
        # print(all_edges)
        data = np.ones(all_edges.shape[0])

        adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])),
                            shape=(num_nodes, num_nodes))
        # print(adj.toarray())
        data_train = np.ones(train_edges.shape[0])
        data_train_false = np.ones(train_edges_false.shape[0])
        
        adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),
                                  shape=(num_nodes, num_nodes))
        adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])), \
                                              shape=(num_nodes, num_nodes))
        # print(adj_train.toarray())

        return adj, adj_train, adj_train_false, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def select_features(args: Namespace, idx2smiles: List[str] = None, num_nodes: int = None) -> Union[sp.dia_matrix, List[str], np.ndarray]:
    if args.use_input_features:
        assert idx2smiles is not None
        num_nodes = len(idx2smiles)
        fg_func = get_features_generator(args.use_input_features_generator)
        try:
            num_features = fg_func(idx2smiles[0]).shape[0]
        except AttributeError:
            num_features = np.array(fg_func(idx2smiles[0])).shape[0]
        features = np.zeros((num_nodes, num_features), dtype=np.float)
        for smiles_idx, smiles in enumerate(idx2smiles):
            features[smiles_idx, :] = fg_func(smiles)
        return idx2smiles, features
    else:
        return idx2smiles