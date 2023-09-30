from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import  degree

from sklearn import preprocessing

import argparse

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer as GNNE
import os
import random
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import collections

import numpy as np 

from gnn import GNNConv
from utils import Classifier, set_random_seed
criterion = nn.CrossEntropyLoss()
from sklearn.metrics import accuracy_score as acc
from copy import deepcopy

from torch_geometric.nn import GCNConv as GCNLayer


from torch_geometric.utils import to_dense_adj

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) # 
    # print(f"Random seed set as {seed}")

def filter_node(idx, edge_type, edge_weight, filtered_nodes):
    for nd in filtered_nodes:
        mask=torch.ones(idx.shape[1]).bool()
        nd_mask=idx[0]!=nd
        mask=mask & nd_mask
        nd_mask=idx[1]!=nd
        mask=mask & nd_mask
        idx=torch.stack((idx[0][mask],idx[1][mask]),dim=0)
        edge_type= edge_type[mask]
        edge_weight=edge_weight[mask]
    return idx, edge_type, edge_weight


class Dist():
    def __init__(self, ):
        pass

    def measure_dist(self, X, edge, train_mask, valid_mask, y_train, y_valid,num_nodes=48758):
        X_ori=X
        adj=to_dense_adj(edge,max_num_nodes=num_nodes)[0].cpu().double()
        row, col = edge
        deg = degree(col, X.size(0), dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        

        deg_inv_sqrt = deg_inv_sqrt.view(-1,1)
        norm = ((deg_inv_sqrt @ deg_inv_sqrt.transpose(0,1) )).cpu()
        norm[range(len(norm)),range(len(norm))]=1
        
        scaler=preprocessing.StandardScaler()
        X=scaler.fit_transform(X)
        X=torch.from_numpy(X)



        lr1=GradientBoostingClassifier(learning_rate=0.05, min_samples_split=2,min_samples_leaf=2,max_depth=5,max_features='sqrt', 
        subsample=0.8,random_state=2023)
        lr1.fit(X[train_mask],y_train)
        
        print('acc before GCN on train set is: %s'%lr1.score(X[train_mask],y_train))
        print('acc before GCN on test set is: %s'%lr1.score(X[valid_mask],y_valid))
        train_mask=train_mask

        X_new= (norm * adj) @ X +X
        lr2=GradientBoostingClassifier(learning_rate=0.05, min_samples_split=2,min_samples_leaf=2,max_depth=5,max_features='sqrt', 
        subsample=0.8,random_state=2023)

        lr2.fit(X_new[train_mask],y_train)
        print('acc after GCN on train set is: %s'%lr2.score(X_new[train_mask],y_train))
        print('acc after GCN on test set is: %s'%lr2.score(X_new[valid_mask],y_valid))
        p1=torch.from_numpy(lr1.predict_proba(X_ori)[edge[0]])
        p1=torch.sum(-p1 * torch.log(p1),dim=1)
        deg_inv_sqrt=deg_inv_sqrt.cpu()
        emb=X_ori[edge[0]]+ X_ori[edge[1]]* deg_inv_sqrt[edge[0]]* deg_inv_sqrt[edge[1]]
        
        p2=torch.from_numpy(lr2.predict_proba(emb))
        p2= torch.sum(-p2 * torch.log(p2),dim=1)
        
        values= p2-p1
        values[torch.isnan(values)]=0

        mask=torch.ones(edge.shape[1]).bool()

        return values,mask
    


class LOO():
    def __init__(self, args,seed=2023, device='cpu',rel_num=3,class_num=3,train_epoch=200):
        self.seed=seed
        self.args=args
        self.device=device
        self.rel_num=rel_num
        self.class_num=class_num
        self.train_epoch=train_epoch

    def model_train(self,node_feat, edge, train_idx, edge_type, edge_weight, y_train,valid_idx,y_valid):
        args=self.args
        set_random_seed(self.seed)
        gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
                rel_num=self.rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
        classifier = Classifier(args.output_dim, self.class_num)
        model = nn.Sequential(gnn, classifier).to(self.device)

        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay,lr=0.001)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),weight_decay=args.weight_decay,lr=0.01)
        elif args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(),weight_decay=args.weight_decay,lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=1e-2)

        best_acc=0
        for epoch in np.arange(self.train_epoch):
            model.train()
            company_emb=gnn.forward(node_feat, edge, train_idx, edge_type, edge_weight)

            
            res = classifier.forward(company_emb)
            loss = criterion(res, y_train)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            scheduler.step()
            
            prob=torch.softmax(res,dim=-1)
            pred=prob.argmax(dim=1).cpu().detach().numpy()


            del res, loss

            '''
                Valid 
            '''
            model.eval()
            with torch.no_grad():
                company_emb=gnn.forward(node_feat, edge, valid_idx, edge_type, edge_weight)

                res = classifier.forward(company_emb)
                loss = criterion(res,y_valid) 

                prob=torch.softmax(res,dim=-1)
                pred=prob.argmax(dim=1).cpu().detach().numpy()
                y_valid_copy=deepcopy(y_valid).cpu().numpy()

                ac=acc(y_valid_copy,pred)

                if ac > best_acc:
                    best_acc = ac
                    model_state_dict=model.state_dict()

                del res, loss
        return model_state_dict

    def model_test(self,model, node_feat, edge, test_idx, edge_type, edge_weight,y_test):
        args=self.args
        model.eval()
        gnn, classifier = model
        with torch.no_grad():
            company_emb=gnn.forward(node_feat, edge, test_idx, edge_type, edge_weight)
            res = classifier.forward(company_emb)

            prob=torch.softmax(res,dim=-1)
            pred=prob.argmax(dim=1).cpu().detach().numpy()
            y_test_copy=deepcopy(y_test).cpu().numpy()
            ac=acc(y_test_copy,pred)
            
        return ac
        

    def measure_dist(self, node_feat, edge, train_idx, valid_idx, y_train, y_valid,edge_type, edge_weight,base='edge'):
        model_state_dict=self.model_train(node_feat, edge, train_idx, edge_type, edge_weight, y_train,valid_idx,y_valid)
        args=self.args
        gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=self.rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
        classifier = Classifier(args.output_dim, self.class_num)
        model = nn.Sequential(gnn, classifier).to(self.device)
        model.load_state_dict(model_state_dict)
        base_score = self.model_test(model, node_feat, edge, valid_idx, edge_type, edge_weight,y_valid)
        values=torch.zeros(edge.shape[1])
        if base=='edge':
            for i in tqdm(range(edge.shape[1])):
                test_edge=torch.cat((edge[:,:i],edge[:,i+1:]),dim=1)
                test_score=self.model_test(model, node_feat, test_edge, valid_idx, edge_type, edge_weight,y_valid)
                values[i]=test_score-base_score
        elif base == 'node':
            dg=degree(edge[0],num_nodes=48758).cpu()
            nd_values=torch.zeros(node_feat.shape[0])
            for i in tqdm(range(node_feat.shape[0])):
                test_node_feat=deepcopy(node_feat)
                test_node_feat[i]=torch.zeros(node_feat.shape[1])
                test_score=self.model_test(model, test_node_feat, edge, valid_idx, edge_type, edge_weight,y_valid)
                nd_values[i]=test_score-base_score
            src=edge[0].cpu()
            dst=edge[1].cpu()
            values=nd_values[src]/dg[src]+nd_values[src]/dg[dst]
        return values

class GCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,class_num=3, n_layer=2):
        super(GCN, self).__init__()
        self.n_layer=n_layer
        self.proj=nn.Linear(input_dim,hidden_dim)
        self.norm=nn.BatchNorm1d(hidden_dim)
        self.gat=nn.ModuleList()
        for i in range(n_layer):
            if i==0:
                self.gat.append(GCNLayer(hidden_dim, output_dim ))
            else:
                self.gat.append(GCNLayer(output_dim, output_dim ))
        self.linear = nn.Linear(output_dim, class_num)

    def forward(self,x,edge_index):
        x=self.norm(self.proj(x))
        for i in range(self.n_layer):
            x= self.gat[i](x, edge_index)
        x = self.linear(x).squeeze()
        return x

class GNNEX():
    def __init__(self, args,seed=2023, device='cpu',rel_num=3,class_num=3,train_epoch=200):
        self.seed=seed
        self.args=args
        self.device=device
        self.rel_num=rel_num
        self.class_num=class_num
        self.train_epoch=train_epoch

    def model_train(self,node_feat, edge, train_idx, edge_type, edge_weight, y_train,valid_idx,y_valid,test_idx,y_test):
        args=self.args
        set_random_seed(self.seed)
        model=GCN(args.input_dim,args.hidden_dim,args.output_dim,class_num=3, n_layer=2).to(self.device)

        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay,lr=0.001)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),weight_decay=args.weight_decay,lr=0.01)
        elif args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(),weight_decay=args.weight_decay,lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=1e-2)

        best_acc=0
        for epoch in np.arange(self.train_epoch):
            model.train()

            res=model.forward(node_feat, edge)[train_idx]
            loss = criterion(res, y_train)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            scheduler.step()
            
            prob=torch.softmax(res,dim=-1)
            pred=prob.argmax(dim=1).cpu().detach().numpy()



            del res, loss

            '''
                Valid 
            '''
            model.eval()
            with torch.no_grad():
                res=model.forward(node_feat, edge)[valid_idx]
                loss = criterion(res,y_valid) 

                prob=torch.softmax(res,dim=-1)
                pred=prob.argmax(dim=1).cpu().detach().numpy()
                y_valid_copy=deepcopy(y_valid).cpu().numpy()

                ac=acc(y_valid_copy,pred)

                if ac > best_acc:
                    best_acc = ac
                    model_state_dict=model.state_dict()

                del res, loss
        return model_state_dict
    
    def measure_dist(self, node_feat, edge, train_idx, valid_idx, y_train, y_valid,edge_type, edge_weight,test_idx,y_test):
        
        model_state_dict=self.model_train(node_feat, edge, train_idx, edge_type, edge_weight, y_train,valid_idx,y_valid,test_idx,y_test)
        args=self.args
        model=GCN(args.input_dim,args.hidden_dim,args.output_dim,class_num=3, n_layer=2).to(self.device)
        model.load_state_dict(model_state_dict)
        model.eval()

        print('start explain ...')
        

        explainer = GNNE(model=model,epochs=200,lr=0.01)
        
        node_feat_mask, edge_mask=explainer.explain_graph(x=node_feat,edge_index=edge)
                                                    # train_idx=train_idx, edge_type=edge_type, edge_weight=edge_weight)
        return edge_mask


