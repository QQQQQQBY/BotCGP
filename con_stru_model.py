import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import argparse
import torch
import numpy as np
import random
from get_data import supervised_getdata
from torch_geometric.nn import SAGEConv, RGCNConv
from torch.nn.parameter import Parameter
from train import train
# from abliation.community_heat import community_embedding
import math
import sklearn

# Auto-Encoder (Community Perception Deep Clustering Module)
class AE(nn.Module):

    def __init__(self, feature_dim, encoderdim1, encoderdim2, decoderdim1, hidden_num):
        super(AE, self).__init__()
        self.enc_in = Linear(feature_dim, encoderdim1)
        self.hidden_enc = nn.ModuleList([Linear(encoderdim1, encoderdim1) for i in range(hidden_num)])
        self.z_layer = Linear(encoderdim1, encoderdim2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dec_in = Linear(encoderdim2, decoderdim1)
        self.hidden_dec = nn.ModuleList([Linear(decoderdim1, decoderdim1) for i in range(hidden_num)])
        self.x_bar_layer = Linear(decoderdim1, feature_dim)

    def forward(self, x):
        enc_result = []
        enc_result.append(F.relu(self.enc_in(x)))
        for layer in self.hidden_enc:
            enc_result.append(F.relu(layer(enc_result[-1])))
        z = self.dropout1(self.z_layer(enc_result[-1])) # 中间层

        dec = self.dropout1(self.dec_in(z))
        for layer in self.hidden_dec:
            dec = F.relu(layer(dec))
        x_bar = self.x_bar_layer(dec)

        return x_bar, enc_result , z


# Eq.(7)
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # XWA
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        # Eq.(7)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    

def cluster(data, k, num_iter, init=None, cluster_temp=5):
    # process data
    data = torch.diag(1. / torch.norm(data, p=2, dim=1)) @ data
    # if init is None:
    #     data_np = data.detach().numpy()
    #     norm = (data_np ** 2).sum(axis=1)
    #     init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
    #     init = torch.tensor(init, requires_grad=True).cuda()
    #     if num_iter == 0:
    #         return init

    # Eq.(8) -- C
    mu = init
    for t in range(num_iter):
        # Eq.(8)
        dist = data @ mu.t() # mu -- Cluster center
        r = torch.softmax(cluster_temp * dist, 1) # Eq.(8) -- F
        cluster_r = r.sum(dim=0) 
        cluster_r += 1e-8 # Prevent 0
        # Update Cluster Data
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1) 
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean 
        mu = new_mu

    # Eq.(8)
    dist = data @ mu.t() 
    r = torch.softmax(cluster_temp * dist, 1)
    return mu, r, dist

# Community Preception of Social Relationships
class Modularity(nn.Module):
    def __init__(self, mo_nfeat, mo_nhid, mo_nout, mo_dropout, mo_K, mo_cluster_temp):
        super(Modularity, self).__init__()
        self.GCN = GCN(mo_nfeat, mo_nhid, mo_nout, mo_dropout)
        self.distmult = nn.Parameter(torch.rand(mo_nout))
        self.sigmoid = nn.Sigmoid()
        self.K = mo_K
        self.cluster_temp = mo_cluster_temp
        self.init = torch.rand(self.K, mo_nout).cuda()

    def forward(self, x, adj, num_iter=1, mu=None):
        embeds = self.GCN(x, adj)
        mu_init, _, _ = cluster(embeds, self.K, num_iter, init=mu, cluster_temp=self.cluster_temp)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist

# GNN-based Social Bot Detection Module, Eq.(17)
class BotDetection(nn.Module):
    def __init__(self, community_dim, class_num):
        super(BotDetection, self).__init__() 
        self.to_label1 = Linear(community_dim, community_dim//2)
        self.to_label2 = Linear(community_dim//2, class_num)
    def forward(self, x):
        x = F.relu(self.to_label1(x))
        x = F.softmax(self.to_label2(x), dim=1)
        return x

# BotCP Model  
class CommunityBot(nn.Module):
    def __init__(self, datasets_name, feature_dim, encoderdim1, encoderdim2, decoderdim1, hidden_num, cat_dim, num_dim, tweet_dim, des_dim, hidden_dim, community_num, class_num, mo_nfeat, mo_nhid, mo_nout, mo_dropout, mo_K, mo_cluster_temp, con_K):
        super(CommunityBot, self).__init__()
        # Community Preception of User-posted Content -- AutoEncoder
        self.au = AE(feature_dim, encoderdim1, encoderdim2, decoderdim1, hidden_num)
        # Community Preception of User-posted Content -- K=\{k_1, k_2, ..., k_n\}
        self.cluster_layer = Parameter(torch.Tensor(con_K, encoderdim2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1

        # Community Preception of Social Relationships  
        self.modular = Modularity(mo_nfeat, mo_nhid, mo_nout, mo_dropout, mo_K, mo_cluster_temp)

        # GNN-based Social Bot Detection Module, Eq.(13)
        self.cat = Linear(cat_dim, hidden_dim)
        self.num = Linear(num_dim, hidden_dim)
        self.tweet = Linear(tweet_dim, hidden_dim)
        if datasets_name == "twibot-20":
            self.des = Linear(des_dim, hidden_dim)
            self.feature = Linear(hidden_dim * 4, hidden_dim)
        else:            
            self.feature = Linear(hidden_dim * 3, hidden_dim)
        # GNN-based Social Bot Detection Module, Eq.(14)
        self.gnn_in = RGCNConv(encoderdim1,encoderdim1,num_relations=2)

        # GNN-based Social Bot Detection Module, Eq.(15)
        self.hidden_gnn = nn.ModuleList([RGCNConv(encoderdim1 + encoderdim1 + mo_nout, encoderdim1,num_relations=2) for i in range(hidden_num)])        
        self.gnn_out = RGCNConv(encoderdim1,encoderdim2,num_relations=2)
        # GNN-based Social Bot Detection Module, Eq.(16)
        self.gnn_community = RGCNConv(encoderdim2,community_num,num_relations=2)

        # GNN-based Social Bot Detection Module, Eq.(17)
        self.class_model = BotDetection(community_num, class_num)

        # Dropout
        self.dropout1 = nn.Dropout(p=0.3)

    def forward(self, datasets_name, cat_dim, num_dim, tweet_num, des_num, user_feature, edge_index, edge_type, train, G_features, adj_all, num_cluster_iter, mu, n_id):
        # Community Preception of User-posted Content -- AutoEncoder      
        x_bar, enc_result , z = self.au(user_feature[:, cat_dim + num_dim:])

        # Community Preception of Social Relationships
            # mu -- Cluster Center
            # r -- F,  measure the similarity of the node embeddings to the clustering centers
        mu, r, embeds, _ = self.modular( G_features, adj_all, num_cluster_iter, mu=mu)

        # GNN-based Social Bot Detection Module, Eq.(13)
        if datasets_name == "twibot-20":
            cat = user_feature[:,0:cat_dim]
            num = user_feature[:,cat_dim: cat_dim + num_dim]
            tweet = user_feature[:, cat_dim + num_dim: cat_dim + num_dim + tweet_dim]
            des = user_feature[:, cat_dim + num_dim + tweet_dim:]
            cat = self.cat(cat)
            num = self.num(num)
            tweet = self.tweet(tweet)
            des = self.des(des)
            feature = torch.cat([cat, num, tweet,des], dim = 1)
            feature = self.dropout1(self.feature(feature))
        else:
            cat = user_feature[:,0:cat_dim]
            num = user_feature[:,cat_dim: cat_dim + num_dim]
            tweet = user_feature[:,cat_dim + num_dim:]
            cat = self.cat(cat)
            num = self.num(num)
            tweet = self.tweet(tweet)
            feature = torch.cat([cat, num, tweet], dim = 1)
            feature = self.dropout1(self.feature(feature))

        # GNN-based Social Bot Detection Module, Eq.(14)
        feature = self.gnn_in(feature,edge_index, edge_type)

        # GNN-based Social Bot Detection Module, Eq.(15)
        for i, layer in enumerate(self.hidden_gnn):
            # enc_result -- autoencoder (the feature of encoder layer)
            # embeds -- Social Relationships' Representations
            feature = torch.cat((feature, enc_result[i], embeds[n_id]),dim = 1)
            feature = self.dropout1(layer(feature, edge_index, edge_type))
        feature = self.dropout1(self.gnn_out(feature,edge_index, edge_type))

        # GNN-based Social Bot Detection Module, Eq.(16)
        feature = self.dropout1(self.gnn_community(feature, edge_index, edge_type))
        feature = torch.sigmoid(feature)

        # GNN-based Social Bot Detection Module, Eq.(17)
        predict = self.class_model(feature)

        if (train):
            # Community Preception of User-posted Content -- Eq.(4)
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()           
            return predict, x_bar, q, mu, r    

        return predict, x_bar, mu, r





def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    # Key Hyper-Parameters
    parser = argparse.ArgumentParser(
        description='mgtab train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--datasets_name', default="mgtab", type=str)
    parser.add_argument('--encoderdim1', default=128, type=int)
    parser.add_argument('--encoderdim2', default=64, type=int)
    parser.add_argument('--decoderdim1', default=64, type=int)
    parser.add_argument('--hidden_num', default=1, type=int)
    parser.add_argument('--cat_dim', default=10, type=int)
    parser.add_argument('--num_dim', default=10, type=int)
    parser.add_argument('--tweet_dim', default=768, type=int)
    parser.add_argument('--des_dim', default=768, type=int)
    parser.add_argument('--community_num', default=32, type=int)
    parser.add_argument('--class_num', default=2, type=int)
    parser.add_argument('--fl', default=0.6, type=int)
    parser.add_argument('--mse', default=0.1, type=int)
    parser.add_argument('--kl', default=0.2, type=int)
    parser.add_argument('--mo', default=0.1, type=int)
    parser.add_argument('--auto_epoch', default=150, type=int)
    parser.add_argument('--th', type=float, default=0.45)
    parser.add_argument('--name', type=str, default='community', help='name of logger')
    parser.add_argument('--mo_nfeat', default=128, type=int)
    parser.add_argument('--mo_nhid', default=128, type=int)
    parser.add_argument('--mo_nout', default=50, type=int)
    parser.add_argument('--mo_dropout', default=0.3, type=int)
    parser.add_argument('--mo_K', default=64, type=int)
    parser.add_argument('--con_K', default=32, type=int)
    parser.add_argument('--mo_cluster_temp', default=100, type=int)
    parser.add_argument('--num_cluster_iter', default=1, type=int)
    parser.add_argument('--log_dir', type=str, default='Log/con16_stru32_rgcn.log', help='dir of logger')
    parser.add_argument('--mo_embed_dim', default=50, type=int)
    args = parser.parse_args()
    setup_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    args.log_dir = f'Log/con_{args.con_K}_mo_{args.mo_K}_M_{args.community_num}_rgcn.log'
    if args.datasets_name == "twibot-20":
        args.cat_dim = 11
        args.num_dim = 6
    if args.datasets_name == "cresci-15":
        args.cat_dim = 1
        args.num_dim = 5
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Process Data
        # adj_all -- The regularized adjacency matrix
        # G_features -- one-hot feature
        # bin_adj_all -- Dense form of adj_all
    dataset, adj_all, G_features, bin_adj_all = supervised_getdata(args.datasets_name, args.seed)
            
    args.mo_nfeat = G_features.shape[1]
    bin_adj_all = bin_adj_all.to(args.device)
    dataset = dataset.to(args.device)
    G_features = G_features.to(args.device)
    args.feature_dim = dataset.x[:, args.cat_dim + args.num_dim:].shape[1]
    adj_all = adj_all.to(args.device)

    # Instantiation Model
    model = CommunityBot(args.datasets_name, args.feature_dim, args.encoderdim1, args.encoderdim2, args.decoderdim1, args.hidden_num, args.cat_dim, args.num_dim, args.tweet_dim, args.des_dim, args.hidden_dim, args.community_num, args.class_num, args.mo_nfeat, args.mo_nhid, args.mo_nout, args.mo_dropout, args.mo_K, args.mo_cluster_temp, args.con_K)
    model = model.to(args.device)
    ae_model = model.au

    # Train Model
    train(args.datasets_name, args.cat_dim, args.num_dim, args.tweet_dim, args.des_dim, dataset, model, ae_model, args, adj_all, G_features, bin_adj_all)
    