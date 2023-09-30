# coding: utf-8
import torch
from os import times
import pandas as pd
import torch
import pickle
import time
from torch import nn
import numpy as np

# from gnn_dgl import GNNConv
from gnn import GNNConv
from utils import Classifier, set_random_seed, FocalLoss, weighted_CELoss

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

from torch_geometric.utils import degree
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

import random

from config import args
import os
from copy import deepcopy
seed=2023
set_random_seed(seed)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
# device=torch.device("cpu")
criterion = nn.CrossEntropyLoss()



#load data
data=args.data_dir+'CED.pkl'
with open(data, "rb") as f:
    dataset = pickle.load(f)

node_feat,graph,idx ,label = dataset

edge,edge_type,edge_weight= graph
train_data_idx, valid_data_idx, test_data_idx= idx


train_idx = list(set(list(train_data_idx)) & set(label.keys()))
valid_idx = list(set(list(valid_data_idx)) & set(label.keys()))
test_idx = list(set(test_data_idx) & set(label.keys()))

y_train = [label[i] for i in train_idx if i in label]
y_valid = [label[i] for i in valid_idx if i in label]
y_test = [label[i] for i in test_idx if i in label]





rel_num=3


extrain_idx = list(set(list(train_data_idx)+list(valid_data_idx)) & set(label.keys()))
extest_idx = list(set(test_data_idx) & set(label.keys()))

exx_train = node_feat[torch.LongTensor(extrain_idx)].numpy()
exy_train = [label[i] for i in extrain_idx if i in label]
exx_test = node_feat[torch.LongTensor(extest_idx)].numpy()
exy_test = [label[i] for i in extest_idx if i in label]
from shapley import Dist,GNNEX,LOO



class Sampler():
    def __init__(self, edge, edge_type, rel_weight):
        dg= degree(edge[0],num_nodes=48758).numpy()
        md=np.median(dg)
        dg_list = []
        for i in dg:
            if i == md:
                dg_list +=[1]
            else:
                dg_list+= [1/np.abs(i-md)]
        dg =dg_list

        type_dict=defaultdict(list)
        sample_seeds=train_idx+valid_idx
        for i in sample_seeds:
            tp = label[i]
            type_dict[tp] += [i]
        for i in test_idx:
            type_dict['t'] += [i]


        edge= edge.transpose(0,1).tolist()
        edge_type=edge_type.tolist()

        graph=defaultdict(lambda:defaultdict(list))

        for i, e in enumerate(edge):
            src,dst=e
            rel_type=edge_type[i]
            graph[src][dst] += [[src, dst, rel_weight[rel_type]*dg[dst],rel_type,i]]

        self.graph=graph
        self.type_dict=type_dict
        self.sample_seeds=sample_seeds
        

    def sample_subgraph(self, budget=10, seed=2023, layer=3, sample_ratio=0.25):
        set_random_seed(seed)
        graph = self.graph

        graph_list=[]
        for ep in tqdm(range(budget)):
            seeds= []
            for tp in self.type_dict:
                sample_set=self.type_dict[tp]
                if tp == 't':
                    ratio = 0.8 
                else:
                    ratio = 1-len(sample_set)*2/len(self.sample_seeds)     
                seeds+= random.choices(sample_set, weights=None, cum_weights=None,  k=int(ratio*len(sample_set)))

            new_edge_case=set()
            for l in tqdm(range(layer)): 
                temp_edge_list=[]
                for src in seeds:
                    if src in graph:
                        for dst in graph[src]:
                            temp_edge_list+=graph[src][dst]
                weights= [ i[2] for i in temp_edge_list]
                w_min= min(weights)
                w_max=max(weights)
                weights= [(i-w_min)/(w_max-w_min) for i in weights]
                w_sum=sum(weights)
                weights=[i/w_sum for i in weights]
                a=np.arange(len(temp_edge_list))
                choosed_edge=np.random.choice(a, size=max(int(sample_ratio*len(temp_edge_list)),len(seeds)),replace=False, p=weights)
                for i in choosed_edge:
                    case=temp_edge_list[i]
                    new_edge_case.add((case[0],case[1],case[3],case[4]))
                seeds=[temp_edge_list[i][1] for i in choosed_edge]

            new_edge=[]
            new_edge_type=[]
            new_edge_case= list(new_edge_case)
            edge_idx=[]
            for i in new_edge_case:
                new_edge+=[[i[0],i[1]]]
                new_edge_type+=[i[2]]
                edge_idx+=[i[3]]
            new_edge = torch.LongTensor(new_edge).transpose(0,1)
            new_edge_type = torch.LongTensor(new_edge_type)
            edge_idx = torch.LongTensor(edge_idx)
            graph_list += [[new_edge,new_edge_type,edge_idx]]

        return graph_list

budget=10
measure='dist'
# budget=0
rel_weight={0:1,1:1,2:0.1}

N=edge.shape[1]
if measure == 'dist':
    dist=Dist()
    values,mask=dist.measure_dist(node_feat, edge, extrain_idx,extest_idx, exy_train,exy_test)
    
    if budget == 0:
        print('assign random values')
        values= torch.randn(N)
    elif budget ==1:
        print('assign values based on once calculation')
        values = values
    else:
        print('assign values using results of %s times sampling'%budget)
        sampler= Sampler(edge, edge_type, rel_weight)
        graph_list=sampler.sample_subgraph(budget=budget, seed=seed, layer=3, sample_ratio=0.6)
        dist=Dist()
        value_matrix=torch.zeros(edge.shape[1],budget).double()
        for i,g in enumerate(graph_list):
            new_edge,new_edge_type,edge_idx=g
            v,mask=dist.measure_dist(node_feat, new_edge, extrain_idx,extest_idx, exy_train,exy_test)
            edge_idx = torch.LongTensor(edge_idx)
            value_matrix[edge_idx,i]=v
        value_matrix=torch.cat((value_matrix,values.unsqueeze(1)),dim=1)
        value_matrix=value_matrix.tolist()
        final_value=[]
        for i in range(len(value_matrix)):
            j=[v for v in value_matrix[i] if v!=0]
            final_value+=[np.mean(j)]
        values=torch.FloatTensor(final_value)


node_feat=node_feat.to(device)
edge=edge.to(device)
test_idx=torch.LongTensor(test_idx).to(device)
edge_type=edge_type.to(device)
y_train=torch.LongTensor(y_train).to(device)
y_test=torch.LongTensor(y_test).to(device)
y_valid=torch.LongTensor(y_valid).to(device)


if measure == 'gnnex':
    dist=GNNEX(args,seed=2023, device=args.cuda,rel_num=3,class_num=3,train_epoch=200)
    values = dist.measure_dist(node_feat, edge, train_idx, valid_idx, y_train, y_valid,edge_type, edge_weight, test_idx,y_test)
elif measure == 'loo':
    dist = LOO(args,seed=2023, device=args.cuda,rel_num=3,class_num=3,train_epoch=200)
    values = dist.measure_dist(node_feat, edge, train_idx, valid_idx, y_train, y_valid,edge_type, edge_weight,base='node')


def test(input=None):
    if input==None:
        gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
        classifier = Classifier(args.output_dim, 3)
        model = nn.Sequential(gnn, classifier).to(device)
        model.load_state_dict(torch.load('./model_save/%s.pkl'%(args.conv_name)))
    else:
        model=input
    model.eval()
    gnn, classifier = model
    with torch.no_grad():
        company_emb=gnn.forward(node_feat, edge, test_idx, edge_type, edge_weight)
        res = classifier.forward(company_emb)

        prob=torch.softmax(res,dim=-1)
        pred=prob.argmax(dim=1).cpu().detach().numpy()
        y_test_copy=deepcopy(y_test).cpu().numpy()
        ac=acc(y_test_copy,pred)
        pr=pre(y_test_copy,pred,average='macro')
        re=rec(y_test_copy,pred,average='macro')
        f=f1(y_test_copy,pred,average='macro')
        rc=roc(y_test_copy,prob.cpu().detach().numpy(),average='macro',multi_class='ovo')
        txt='Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f \
              Best Test F1: %.4f Best Test ROC: %.4f' % (ac,pr,re,f,rc)
        print(txt)
    return ac, pr,re,f,rc
        


def train(seed=2023):
    set_random_seed(seed)
    gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
    classifier = Classifier(args.output_dim, 3)
    model = nn.Sequential(gnn, classifier).to(device)

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
    for epoch in np.arange(200):
        '''
            Train 
        '''
        model.train()
        company_emb=gnn.forward(node_feat, edge, train_idx, edge_type, edge_weight)

        
        res = classifier.forward(company_emb)
        loss = criterion(res, y_train)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # train_losses = loss.cpu().detach().tolist()
        scheduler.step()
        
        prob=torch.softmax(res,dim=-1)
        pred=prob.argmax(dim=1).cpu().detach().numpy()
        y_train_copy=deepcopy(y_train).cpu().numpy()

        # train_acc=acc(y_train_copy,pred)


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
                torch.save(model.state_dict(), './model_save/%s.pkl'%(args.conv_name))


            del res, loss




vorder=torch.sort(values,descending=True)[1]
l=[]
granularity=20
ori_edge=edge
ori_edge_type=edge_type
ori_edge_weight=edge_weight
for i in tqdm(range(granularity+1)):
    if i < granularity:
        mask = vorder[:int(N/granularity*i)]
    else:
        mask = vorder[:]
    edge = ori_edge.transpose(0,1)[mask].transpose(0,1)
    edge_type = ori_edge_type[mask]
    edge_weight=None
    print(edge.shape)
    train(seed=seed)
    ac, pr,re,f,rc= test()
    l+=[[ac,pr,re,f,rc,i]]


plot_dict=defaultdict(list)
for i in range(granularity+1):
    if i < granularity:
        plot_dict['x']+=[int(N/granularity*i)] 
    else:
        plot_dict['x']+=[len(vorder)] 
    plot_dict['y']+=[l[i][0]]
    plot_dict['type']+=['test']

sns.lineplot(x='x', y='y',  hue="type", style="type", data=plot_dict)


plt.legend()

plt.xlabel('Number of edge inspected', fontsize=16)
y_label_name = 'Accuracy'
plt.ylabel(y_label_name, fontsize=16)


plt.savefig('./figures/explain_figure_%s_%s.png'%(args.conv_name,measure))
plt.figure().clear()


res_log='./figures/explain_log_%s_%s.pkl' %(args.conv_name,measure)
with open(res_log, 'wb') as f:
    pickle.dump(l, f)


