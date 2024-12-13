# coding: utf-8
import os
os.chdir('./DBLP/')
import torch
from os import times
import pandas as pd
import torch
import pickle
import time
from torch import nn
import numpy as np

from gnn import GNNConv
from utils import Classifier, set_random_seed, FocalLoss, weighted_CELoss

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

import warnings
warnings.filterwarnings("ignore")


from config import args


from copy import deepcopy



if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
# device=torch.device("cpu")
criterion = nn.CrossEntropyLoss()



#load data
homo=args.data_dir+'dblp_homo.pkl'
hete=args.data_dir+'dblp_hete.pkl'

dims= {'author':334,'paper':4231,'term':50,'conference':4231}


if args.conv_name=="hgt" or args.conv_name=="han":
    with open(hete, "rb") as f:
        hete_data = pickle.load(f)
        node_dict,edge,label,train_idx,valid_idx,test_idx=hete_data  
        edge = {i:edge[i].to(device) for i in edge}
        edge_type=None
else:
    with open(homo, "rb") as f:
        homo_data = pickle.load(f)
        node_dict,edge,edge_type,label,train_idx,valid_idx,test_idx =homo_data
        edge = edge.to(device)
        edge_type=edge_type.to(device)


if args.conv_name=="sehgnn":
    with open('./data/metapath_feat.pkl','rb') as f:
        dims,node_dict = pickle.load(f)

label=label.tolist()

# for reproducing data
# from sklearn.model_selection import train_test_split
# def split_data(n, ratio=[0.6,0.2,0.2],seed=2023):
#     set_random_seed(2023)
#     companies_index= np.arange(n)
#     train_data_idx, rest_test_data_idx, _, _ = train_test_split(companies_index, companies_index, train_size=ratio[0],
#                                                                 random_state=seed)
#     valid_data_idx, test_data_idx, _, _ = train_test_split(rest_test_data_idx, rest_test_data_idx, test_size=(1-ratio[0])/2,
#                                                            random_state=seed)

#     return train_data_idx,valid_data_idx, test_data_idx
# train_idx,valid_idx,test_idx=split_data(4057, ratio=[0.8,0.1,0.1],seed=2023)
# with open('./data/index.pkl','wb') as f:
#     idxs=[train_idx,valid_idx,test_idx]
#     pickle.dump(idxs,f)
set_random_seed(2028)
with open('./data/index.pkl', "rb") as f:
    idxs = pickle.load(f)
train_idx,valid_idx,test_idx=idxs


y_train = torch.LongTensor([label[i]  for i in train_idx])
y_valid = torch.LongTensor([label[i]  for i in valid_idx])
y_test = torch.LongTensor([label[i]  for i in test_idx])

rel_num=3
label_num = 4

gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, dims=dims, n_heads=args.n_heads, dropout=args.dropout)


classifier = Classifier(args.output_dim, label_num)
model = nn.Sequential(gnn, classifier).to(device)

train_idx=torch.LongTensor(train_idx).to(device)
valid_idx=torch.LongTensor(valid_idx).to(device)
test_idx=torch.LongTensor(test_idx).to(device)

y_train=torch.LongTensor(y_train).to(device)
y_test=torch.LongTensor(y_test).to(device)
y_valid=torch.LongTensor(y_valid).to(device)
edge_weight= None


if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay,lr=0.001)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),weight_decay=args.weight_decay,lr=0.01)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(),weight_decay=args.weight_decay,lr=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=1e-2)



def test(input=None):
    if input==None:
        gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, dims=dims, n_heads=args.n_heads, dropout=args.dropout)
        classifier = Classifier(args.output_dim, label_num)
        model = nn.Sequential(gnn, classifier).to(device)
        model.load_state_dict(torch.load('./model_save/%s.pkl'%(args.conv_name)))
    else:
        model=input
    model.eval()
    gnn, classifier = model
    with torch.no_grad():
        company_emb=gnn.forward(node_dict, edge, test_idx, edge_type,edge_weight)
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
        



def train():
    best_f=0
    count=0
    up=0
    for epoch in np.arange(args.n_epoch):
        st=time.time()
        '''
            Train 
        '''
        model.train()
        company_emb=gnn.forward(node_dict, edge, train_idx, edge_type,edge_weight)

        
        res = classifier.forward(company_emb)
        loss = criterion(res, y_train)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses = loss.cpu().detach().tolist()
        scheduler.step()
        
        prob=torch.softmax(res,dim=-1)
        pred=prob.argmax(dim=1).cpu().detach().numpy()
        y_train_copy=deepcopy(y_train).cpu().numpy()

        train_acc=acc(y_train_copy,pred)


        del res, loss

        '''
            Valid 
        '''
        model.eval()
        with torch.no_grad():
            company_emb=gnn.forward(node_dict, edge, valid_idx, edge_type,edge_weight)

            res = classifier.forward(company_emb)
            loss = criterion(res,y_valid) 

            prob=torch.softmax(res,dim=-1)
            pred=prob.argmax(dim=1).cpu().detach().numpy()
            y_valid_copy=deepcopy(y_valid).cpu().numpy()

            ac=acc(y_valid_copy,pred)
            pr=pre(y_valid_copy,pred, average='macro')
            re=rec(y_valid_copy,pred, average='macro')
            f=f1(y_valid_copy,pred,average='macro')
            rc=roc(y_valid_copy,prob.cpu().detach().numpy(),average='macro',multi_class='ovo')

            if f > best_f:
                best_f = f
                torch.save(model.state_dict(), './model_save/%s.pkl'%(args.conv_name))

                print('UPDATE!!!')

            et = time.time()
            print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.4f  Train Acc: %.2f  Valid Loss: %.4f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f"  ) % \
                (epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses), train_acc, \
                loss.cpu().detach().tolist(), ac,pr,re,f,rc))

            del res, loss

train()     
test()

import csv
model.load_state_dict(torch.load('./model_save/%s.pkl'%(args.conv_name)))
model.eval()
gnn, classifier = model
with torch.no_grad():
    company_emb=gnn.forward(node_dict, edge, test_idx, edge_type,edge_weight)
    res = classifier.forward(company_emb)
    prob=torch.softmax(res,dim=-1)
    pred=prob.argmax(dim=1).cpu().detach().numpy()
    y_test_copy=deepcopy(y_test).cpu().numpy()

    ac=acc(y_test_copy,pred)
    pr=pre(y_test_copy,pred,average='macro')
    re=rec(y_test_copy,pred,average='macro')
    f1_score=f1(y_test_copy,pred,average='macro')
    rc=roc(y_test_copy,prob.cpu().detach().numpy(),average='macro',multi_class='ovo')
    txt='Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f Best Test F1: %.4f Best Test ROC: %.4f' % (ac,pr,re,f1_score,rc)
    with open('./log/%s.txt'%(args.conv_name),'a') as f:
            f.write('configure info- n_heads: %s, n_layers: %s, n_epoch: %s, dropout: %s, clip: %s, weight_decay: %s.'%\
                    (args.n_heads, args.n_layers,args.n_epoch, args.dropout,args.clip,args.weight_decay,))
            f.write('\n')
            f.write(str(txt))
            f.write('\n')


    data = ['Best Test Acc', 'Best Test Pre', 'Best Test Recall', 'Best Test F1',  'Best Test ROC']
    indicators=[ac,pr,re,f1_score,rc]
    with open('./log/%s.csv'%(args.conv_name), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(indicators)
