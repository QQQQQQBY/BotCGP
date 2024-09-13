from torch.optim import Adam
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.loader import NeighborSampler
import os
# from kmeans import kmeans
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import logging
import time
import numpy as np
import random
import datetime

def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        c[nanix] = x[torch.randperm(N)[:ndead]]
    return c

def get_logger(name,log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    # logger 
    fileHandler = logging.FileHandler(filename=log_dir,mode='w+',encoding='utf-8')
    fileHandler.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter("%(asctime)s|%(levelname)-8s|%(filename)10s:%(lineno)4s|%(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, torch.LongTensor(ids.data.numpy()), 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(args.device)
        alpha = self.alpha[torch.LongTensor(ids.data.numpy()).view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# Community Preception of Social Relationships -- Eq.(9) Discrete modularity value
def make_modularity_matrix(adj, args):
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]).to(args.device) - torch.eye(adj.shape[0]).to(args.device)) 
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def loss_modularity(r, bin_adj, mod, args):
    bin_adj_nodiag = bin_adj * (
            torch.ones(bin_adj.shape[0], bin_adj.shape[0]).to(args.device) - torch.eye(bin_adj.shape[0]).to(args.device))
    return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()

def pre_train_autoencoder(ae_model, feature, con_K, auto_epoch):
    setup_seed(10)
    optimizer = Adam(ae_model.parameters(), lr=1e-3)
    print("Train autoencoder...")
    for epoch in tqdm(range(auto_epoch)):
        x_bar, _, _ = ae_model(feature[:, 20:])
        loss = F.mse_loss(x_bar, feature[:, 20:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(ae_model.state_dict(), f'SaveModel/model_autoencoder_{con_K}.pkl')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        torch.empty_cache()

def train(datasets_name, cat_dim, num_dim, tweet_num, des_num, dataset, model, ae_model, args, adj_all, G_features, bin_adj_all):
    min_loss = 1000
    # Record log
    logger = get_logger(args.name,args.log_dir)
    logger.info('test logger')
    begin_time = time.localtime()
    begin_time = time.strftime('%Y-%m-%d %H:%M:%S',begin_time)
    logger.info(f'Beginning Time of Procedure:{begin_time}')
    
    # Partition dataset
    train_loader = NeighborSampler(dataset.edge_index, sizes=[-1], batch_size=1024, num_workers=6, node_idx=dataset.train_idx, shuffle=True)
    val_loader = NeighborSampler(dataset.edge_index,  sizes=[-1], batch_size=1024, num_workers=6, node_idx=dataset.val_idx, shuffle=True)
    test_loader = NeighborSampler(dataset.edge_index,  sizes=[-1], batch_size=1024, num_workers=6, node_idx=dataset.test_idx, shuffle=True)
    batch_num = len(train_loader)

    # Initial training autoencoder
    if(not os.path.exists(f'SaveModel/model_autoencoder_{args.con_K}.pkl')):
            pre_train_autoencoder(ae_model, dataset.x, args.con_K,args.auto_epoch)
    ae_model.load_state_dict(torch.load(f'SaveModel/model_autoencoder_{args.con_K}.pkl'))
    # Initialize Cluster Center -- K
    with torch.no_grad():
        _, _, z = ae_model(dataset.x[:, args.cat_dim + args.num_dim:])
    model.cluster_layer.data = kmeans(z.data, args.con_K).to(args.device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    model.train()

    # Initialize Cluser Center -- C
    mu = torch.rand(args.mo_K, args.mo_embed_dim).to(args.device)

    # Community Preception of Social Relationships -- Q, Eq.(9)
    test_object = make_modularity_matrix(bin_adj_all, args)

    mu_all = torch.zeros(args.mo_K, args.mo_embed_dim).to(args.device)
    count = 0
    for i in tqdm(range(args.epoch)):
        print('begin the {} training'.format(i+1))
        logger.info(f'begin the {i+1} training')
        total_loss = total_f1 = total_pre = total_acc = total_recall = 0

        # n_id: The nodes involved in this batch training
        # adjs：The edge_index and the index (e_id) of the edge involved in this batch training
        for batch_size,n_id, adjs in tqdm(train_loader):
            feature = dataset.x[n_id]
            label = dataset.y[n_id[:batch_size]]
            optimizer.zero_grad()  
            edge_index = adjs.edge_index.to(args.device)
            edge_type = dataset.edge_type[adjs.e_id.to(args.device)]

            # x_bar: Reconstructed embeddings
            
            predict, x_bar, q, mu, r = model(datasets_name, cat_dim, num_dim, tweet_num, des_num, feature, edge_index, edge_type, True, G_features, adj_all, args.num_cluster_iter, mu, n_id)           
            mu_all = mu_all + mu
            count = count + 1

            # Community Preception of User-posted Content -- Eq.(11) - Eq.(12)
            loss_mo = loss_modularity(r, bin_adj_all, test_object, args)
            loss_mo = -loss_mo

            # Community Preception of User-posted Content -- Eq.(5)
            p = target_distribution(q)       

            # MSE (Eq.(3)) and KL Divergence (Eq.(6))
            mse_loss = F.mse_loss(x_bar, feature[:, 20:])
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

            # FocalLoss (Eq.(18))
            focal_loss = FocalLoss(2)
            label = label.cpu()
            predict = predict.cpu()
            label_mask = (label != -1)
            label_loss = focal_loss(predict[:batch_size][label_mask], label[label_mask])

            # Eq.(19)
            loss = args.fl * label_loss + args.mse * mse_loss + args.kl * kl_loss + args.mo * loss_mo
            
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
            label = label.cpu()
            predict = predict[:batch_size].max(1)[1][label_mask].to('cpu').detach().numpy()            
            pre_result = (predict > args.th)

            # evaluation index
            if datasets_name == "mgtab":
                f1 = f1_score(label[label_mask], pre_result, average='macro')
                total_f1 = total_f1 + f1
                pre = precision_score(label[label_mask], pre_result, average='macro')
                total_pre = total_pre + pre
                acc = accuracy_score(label[label_mask], pre_result)
                total_acc = total_acc + acc
                recall = recall_score(label[label_mask], pre_result, average='macro')
                total_recall = total_recall + recall
            else:
                f1 = f1_score(label[label_mask], pre_result)
                total_f1 = total_f1 + f1
                pre = precision_score(label[label_mask], pre_result)
                total_pre = total_pre + pre
                acc = accuracy_score(label[label_mask], pre_result)
                total_acc = total_acc + acc
                recall = recall_score(label[label_mask], pre_result)
                total_recall = total_recall + recall
        loss = total_loss / batch_num
        f1 = total_f1 / batch_num
        pre = total_pre / batch_num
        acc = total_acc / batch_num
        recall = total_recall / batch_num
        print("Train set results:",
          "train_loss={:.4f}".format(loss),
          "train_accuracy= {:.4f}".format(acc),
          "train_precision= {:.4f}".format(pre),
          "train_recall= {:.4f}".format(recall),
          "train_f1= {:.4f}".format(f1))
        logger.info(f"Train set results: train_loss = {loss}, train_accuracy= {acc}, precision= {pre}, recall= {recall}, f1_score= {f1}")
        print('End the {} training'.format(i+1))
        logger.info(f'End the {i+1} training')

        # Verification
        print("Begin validing...")
        logger.info("Begin validing...")
        mu = mu_all / count
        loss_val,f1_val,pre_val,acc_val,recall_val = val_test(datasets_name, cat_dim, num_dim, tweet_num, des_num, val_loader,model, ae_model, args, dataset, adj_all, G_features, mu, test_object, bin_adj_all)
        print("Val set results:",
          "val_loss={:.4f}".format(loss_val),
          "val_accuracy= {:.4f}".format(acc_val),
          "val_precision= {:.4f}".format(pre_val),
          "val_recall= {:.4f}".format(recall_val),
          "val_f1= {:.4f}".format(f1_val))
        
        logger.info(f"Val set results: Val_loss = {loss_val}, Val_accuracy= {acc_val}, Val_precision= {pre_val}, Val_recall= {recall_val}, val_f1= {f1_val}")
        # Save Model
        if loss_val < min_loss:
            min_loss = loss_val
            print("save model...")
            logger.info("save model...")
            torch.save(model.state_dict(),f"SaveModel/con{args.con_K}_stru{args.mo_K}_M_{args.community_num}_rgcn.pth")

        print("End validing...")
        logger.info("End validing...")

        # Test
        print("Begin testing...")
        logger.info("Begin testing...")
        loss_test,f1_test,pre_test,acc_test,recall_test = val_test(datasets_name, cat_dim, num_dim, tweet_num, des_num, test_loader,model, ae_model, args, dataset, adj_all, G_features, mu, test_object, bin_adj_all)
        
        print("Test set results:",
          "test_loss={:.4f}".format(loss_test),
          "test_accuracy= {:.4f}".format(acc_test),
          "test_precision= {:.4f}".format(pre_test),
          "test_recall= {:.4f}".format(recall_test),
          "test_f1= {:.4f}".format(f1_test))
        logger.info(f"Test set results: Test_loss = {loss_test}, Test_accuracy= {acc_test}, Test_precision= {pre_test}, Test_recall= {recall_test}, Test_f1= {f1_test}")

        print("End testing...")
        logger.info("End testing...")

    # Best Results
    model.load_state_dict(torch.load(f"SaveModel/con{args.con_K}_stru{args.mo_K}_M_{args.community_num}_rgcn.pth"))  
    loss_test,f1_test,pre_test,acc_test,recall_test = val_test(datasets_name, cat_dim, num_dim, tweet_num, des_num, test_loader,model, ae_model, args, dataset, adj_all, G_features, mu, test_object, bin_adj_all)
    print("Best model Test set results:",
          "test_loss={:.4f}".format(loss_test),
          "test_accuracy= {:.4f}".format(acc_test),
          "test_precision= {:.4f}".format(pre_test),
          "test_recall= {:.4f}".format(recall_test),
          "test_f1= {:.4f}".format(f1_test))

    logger.info(f"Best model Test set results: Test_loss = {loss_test}, Test_accuracy= {acc_test}, Test_precision= {pre_test}, Test_recall= {recall_test}, Test_f1= {f1_test}")
    
    # Ending Time
    end_time = time.localtime()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S',end_time)
    print('Ending Time:',end_time)
    logger.info(f'Ending Time:{end_time}')
    startTime= datetime.datetime.strptime(begin_time,"%Y-%m-%d %H:%M:%S")
    endTime= datetime.datetime.strptime(end_time,"%Y-%m-%d %H:%M:%S")
    m,s = divmod((endTime- startTime).seconds,60)
    h, m = divmod(m, 60)
    print(f'Consuming Time:{h}h{m}m{s}s')
    logger.info(f'Consuming Time:{h}h{m}m{s}s')




def val_test(datsets_name, cat_dim, num_dim, tweet_dim, des_dim, dataset, model, ae_model, args, data, adj_all, G_features, mu, test_object, bin_adj_all):
    batch_num = len(dataset)
    ae_model.load_state_dict(torch.load(f'SaveModel/model_autoencoder_{args.con_K}.pkl'))
    # Cluster Center K
    with torch.no_grad():
        _, _, z = ae_model(data.x[:, cat_dim + num_dim:])
    model.cluster_layer.data = kmeans(z.data, args.con_K).to(args.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        torch.empty_cache()
    model.eval()
    total_loss = total_f1 = total_pre = total_acc = total_recall = 0

    with torch.no_grad():
        for batch_size,n_id, adjs in dataset:
            feature = data.x[n_id]
            label = data.y[n_id[:batch_size]]            
            edge_type = data.edge_type[adjs.e_id.to(args.device)]
            edge_index = adjs.edge_index.to(args.device)
            predict, x_bar, mu, r = model(datsets_name, cat_dim, num_dim, tweet_dim, des_dim, feature, edge_index, edge_type, False, G_features, adj_all, args.num_cluster_iter, mu, n_id)           
            
            # Community Preception of User-posted Content -- Eq.(11) - Eq.(12)
            loss_mo = loss_modularity(r, bin_adj_all, test_object, args)
            loss_mo = -loss_mo
            # MSE (Eq.(3))
            mse_loss = F.mse_loss(x_bar, feature[:,20:])
                       
            label = label.cpu()
            predict = predict.cpu()
            focal_loss = FocalLoss(2)
            label_mask = (label != -1)

            # FocalLoss (Eq.(18))
            label_loss = focal_loss(predict[:batch_size][label_mask], label[label_mask])
            loss = args.fl * label_loss + args.mse * mse_loss + args.mo * loss_mo
            total_loss = total_loss + loss            
            predict = predict[:batch_size].max(1)[1][label_mask].to('cpu').detach().numpy()                   
            pre_result = (predict > args.th)

            # evaluation index
            if datasets_name == "mgtab":
                f1 = f1_score(label[label_mask], pre_result, average='macro')
                total_f1 = total_f1 + f1
                pre = precision_score(label[label_mask], pre_result, average='macro')
                total_pre = total_pre + pre
                acc = accuracy_score(label[label_mask], pre_result)
                total_acc = total_acc + acc
                recall = recall_score(label[label_mask], pre_result, average='macro')
                total_recall = total_recall + recall
            else:
                f1 = f1_score(label[label_mask], pre_result)
                total_f1 = total_f1 + f1
                pre = precision_score(label[label_mask], pre_result)
                total_pre = total_pre + pre
                acc = accuracy_score(label[label_mask], pre_result)
                total_acc = total_acc + acc
                recall = recall_score(label[label_mask], pre_result)
                total_recall = total_recall + recall

    loss = total_loss / batch_num
    f1 = total_f1 / batch_num
    pre = total_pre / batch_num
    acc = total_acc / batch_num
    recall = total_recall / batch_num
    return loss, f1, pre, acc, recall


    