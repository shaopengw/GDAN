# coding: utf-8
import torch
import torch
import pickle
import time
from torch import nn
import numpy as np

# from gnn_dgl import GNNConv
from gnn import GNNConv
from utils import Classifier, set_random_seed

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

import warnings
warnings.filterwarnings("ignore")


from config import args
import os
from copy import deepcopy

set_random_seed(2023)

# if args.cuda != -1:
#     device = torch.device("cuda:" + str(args.cuda))
# else:
#     device = torch.device("cpu")
device=torch.device("cpu")
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


# graph =edge,edge_type,edge_weight
# idx =train_data_idx, valid_data_idx, test_data_idx
# dataset =node_feat,graph,idx ,label 
# dt=args.data_dir+'CED.pkl'
# with open(dt, 'wb') as f:
#     pickle.dump(dataset, f)




rel_num=3


gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)


classifier = Classifier(args.output_dim, 3)
model = nn.Sequential(gnn, classifier).to(device)
node_feat=node_feat.to(device)
edge=edge.to(device)
test_idx=torch.LongTensor(test_idx).to(device)
edge_type=edge_type.to(device)
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
            rel_num=rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
        classifier = Classifier(args.output_dim, 3)
        model = nn.Sequential(gnn, classifier).to(device)
        model.load_state_dict(torch.load('./model_save/%s.pkl'%(args.conv_name)))
    else:
        model=input
    model.eval()
    gnn, classifier = model
    with torch.no_grad():
        company_emb=gnn.forward(node_feat, edge, test_idx, edge_type,edge_weight)
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
    best_acc=0
    up=0
    for epoch in np.arange(args.n_epoch):

        st=time.time()
        '''
            Train 
        '''
        model.train()
        company_emb=gnn.forward(node_feat, edge, train_idx, edge_type,edge_weight)

        
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
            company_emb=gnn.forward(node_feat, edge, valid_idx, edge_type,edge_weight)

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

            if ac > best_acc:
                best_acc = ac
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
    company_emb=gnn.forward(node_feat, edge, test_idx, edge_type,edge_weight)
    res = classifier.forward(company_emb)
    prob=torch.softmax(res,dim=-1)
    # print(prob.cpu().detach().tolist())
    pred=prob.argmax(dim=1).cpu().detach().numpy()
    y_test_copy=deepcopy(y_test).cpu().numpy()
    # print(y_test_copy)
    # print(pred)

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