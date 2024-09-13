
import torch
from sklearn.utils import shuffle
import numpy as np
# from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
import networkx as nx
import scipy.sparse as sp

# D^{-1}A,
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def mx_to_sparse_tensor(mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

    
def load_data(G):
    """Load network (graph)"""
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # The format of PyTorch tensors
    adj = mx_to_sparse_tensor(adj)
    # Initialize the one-hot feature
    features = torch.eye(len(G.nodes())).to_sparse()
    return adj, features


def supervised_getdata(datasets_name,seed):
    if datasets_name == "mgtab":
        mgtab = Data()
        mgtab.x = torch.load("Dataset/MGTAB/features.pt")
        mgtab.y = torch.load("Dataset/MGTAB/labels_bot.pt")
        mgtab.edge_index = torch.load("Dataset/MGTAB/edge_index_0_1.pt")
        mgtab.edge_type = torch.load("Dataset/MGTAB/edge_type_01.pt")
        user_number = mgtab.x.shape[0]

        # Shuffle Dataset
        shuffled_idx = shuffle(np.array(range(user_number)), random_state=seed) 
        shuffled_idx_G = np.array(range(user_number))
        mgtab.train_idx = torch.tensor(shuffled_idx[:int(0.7* user_number)])
        mgtab.val_idx = torch.tensor(shuffled_idx[int(0.7*user_number): int(0.9*user_number)])
        mgtab.test_idx = torch.tensor(shuffled_idx[int(0.9*user_number):])
        mgtab.idx = torch.tensor(shuffled_idx)
        
        # Community Preception of Social Relationships (adjacent matrix and feature)
        G = nx.Graph()
        G.add_nodes_from(shuffled_idx_G)
        G.add_edges_from(mgtab.edge_index.T.tolist())
        adj_all, G_features = load_data(G)
        bin_adj_all = (adj_all.to_dense() > 0).float()
        return mgtab,adj_all, G_features, bin_adj_all
    
    if datasets_name == "cresci-15":
        cresci_15 = Data()
        
        cresci_15.x = torch.load("Dataset/Cresci-15/feature_cat_num_tweet.pt")
        cresci_15.y = torch.load("Dataset/Cresci-15/label.pt")
        cresci_15.edge_index = torch.load("Dataset/Cresci-15/edge_index.pt")
        cresci_15.edge_type = torch.load("Dataset/Cresci-15/edge_type.pt")
        user_number = cresci_15.x.shape[0]
       
        # follow previous work, no messing with the dataset
        shuffled_idx = np.array(range(user_number))
        cresci_15.train_idx = torch.tensor(shuffled_idx[:int(0.7* user_number)])
        cresci_15.val_idx = torch.tensor(shuffled_idx[int(0.7*user_number): int(0.9*user_number)])
        cresci_15.test_idx = torch.tensor(shuffled_idx[int(0.9*user_number):])
        cresci_15.idx = torch.tensor(shuffled_idx)
        
        # Community Preception of Social Relationships (adjacent matrix and feature)
        G = nx.Graph()
        G.add_nodes_from(shuffled_idx)
        G.add_edges_from(mgtab.edge_index.T.tolist())
        adj_all, G_features = load_data(G)
        bin_adj_all = (adj_all.to_dense() > 0).float()
        return mgtab,adj_all, G_features, bin_adj_all
    if datasets_name == "twibot-20":
        twibot20 = Data()
        twibot20.x = torch.load("Dataset/Twibot-20/processed_data/feature_cat_num_tweet_des_6_11.pt")
        twibot20.y = torch.load("Dataset/Twibot-20/node_label.pt")
        twibot20.edge_index = torch.load("Dataset/Twibot-20/edge_index.pt")
        twibot20.edge_type = torch.load("Dataset/Twibot-20/edge_type.pt")
        user_number = twibot20.x.shape[0]

        # follow previous work, no messing with the dataset
        shuffled_idx = np.array(range(user_number))       
        twibot20.train_idx = torch.tensor(shuffled_idx[:int(0.7* user_number)])
        twibot20.val_idx = torch.tensor(shuffled_idx[int(0.7*user_number): int(0.9*user_number)])
        twibot20.test_idx = torch.tensor(shuffled_idx[int(0.9*user_number):])
        twibot20.idx = torch.tensor(shuffled_idx)
        
        G = nx.Graph()
        G.add_nodes_from(shuffled_idx)
        G.add_edges_from(twibot20.edge_index.T.tolist())
        adj_all, G_features = load_data(G)
        bin_adj_all = (adj_all.to_dense() > 0).float()
        return twibot20,adj_all, G_features, bin_adj_all
        
        
