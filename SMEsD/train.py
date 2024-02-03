# coding: utf-8
import torch
from os import times
import pandas as pd
import torch
import time
from torch import nn
import numpy as np

from gnn import GNNConv
from utils import Classifier, set_random_seed,initializae_company_info

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")


from config import args
import os
from copy import deepcopy

set_random_seed(2023)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()


#load data
path = './data/'
train_data=pd.read_pickle('%strain_data.pkl'%path)
valid_data=pd.read_pickle('%svalidate_data.pkl'%path)
test_data=pd.read_pickle('%stest_data.pkl'%path)
split_data_idx=pd.read_pickle('%ssplit_data_idx.pkl'%path)


###justification dict
total_company_num=3976
person_num=2405
court_type=4
category=4
time_label_num=5

train_company_num=2816
valid_company_num=721
test_company_num=491

rel_num=7
cause_type_num=11

train_risk_data,train_company_attr,train_hete_graph,train_hyp_graph,train_label=train_data
valid_risk_data,valid_company_attr,valid_hete_graph,valid_hyp_graph,valid_label=valid_data
test_risk_data,test_company_attr,test_hete_graph,test_hyp_graph,test_label=test_data
train_idx,valid_idx,test_idx=split_data_idx

x_train=initializae_company_info(train_risk_data,train_company_attr,train_company_num,cause_type_num,court_type,category,train_idx)
x_valid=initializae_company_info(valid_risk_data,valid_company_attr,valid_company_num,cause_type_num,court_type,category,valid_idx)
x_test=initializae_company_info(test_risk_data,test_company_attr,test_company_num,cause_type_num,court_type,category,test_idx)


hete_graph = pd.read_pickle('./data/hete_graph.pkl')
g_train, g_valid, g_test = hete_graph



x = torch.zeros((total_company_num, 23))
x[train_idx] = torch.FloatTensor(x_train)
x[valid_idx] = torch.FloatTensor(x_valid)
x[test_idx] = torch.FloatTensor(x_test)




if args.conv_name in ['lr','svm','dt','gbdt']:
    if args.conv_name=='lr':
        model=LogisticRegression(solver='lbfgs', max_iter=5000,C=0.1).fit(x_train, train_label)
        prob=model.predict_proba(x_test)
        res=[np.argmax(i) for i in prob]

        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' % 
            (acc(test_label,res),pre(test_label,res),rec(test_label,res),f1(test_label,res),roc(test_label,np.array(prob)[:,1])))
    elif args.conv_name=='svm':
        model = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
        model.fit(x_train, train_label)
        res=model.predict(x_test)

        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' % 
            (acc(test_label,res),pre(test_label,res),rec(test_label,res),f1(test_label,res),roc(test_label,res)))
    elif args.conv_name=='dt':
        model = DecisionTreeClassifier(random_state=2022,max_features=20,max_depth=10) 
        model.fit(x_train, train_label) 

        res = model.predict(x_test)
        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' % 
            (acc(test_label,res),pre(test_label,res),rec(test_label,res),f1(test_label,res),roc(test_label,res)))
    elif args.conv_name=='gbdt':
        model=GradientBoostingClassifier(learning_rate=0.01, min_samples_split=10,min_samples_leaf=2,max_depth=3,max_features='sqrt', 
        subsample=0.8,random_state=2024)

        model.fit(x_train, train_label)
        res=model.predict(x_test)
        predprob = model.predict_proba(x_test)[:, 1]
        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' % 
            (acc(test_label,res),pre(test_label,res),rec(test_label,res),f1(test_label,res),roc(test_label,predprob)))

    exit()

gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)


classifier = Classifier(args.output_dim, 2)
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



if device != 'cpu':
    node_feat=x.to(device)
    train_edge_index, train_edge_type =  g_train[0].to(device),g_train[1].to(device)
    valid_edge_index, valid_edge_type =g_valid[0].to(device),g_valid[1].to(device)
    test_edge_index, test_edge_type =g_test[0].to(device),g_test[1].to(device)

    edge = torch.cat([train_edge_index, valid_edge_index, test_edge_index], dim=1)
    edge_type = torch.cat([train_edge_type, valid_edge_type, test_edge_type], dim=0)

    train_idx = torch.LongTensor(train_idx).to(device)
    valid_idx = torch.LongTensor(valid_idx).to(device)
    test_idx = torch.LongTensor(test_idx).to(device)
    y_train = torch.LongTensor(train_label).to(device)
    y_valid = torch.LongTensor(valid_label).to(device)
    y_test = torch.LongTensor(test_label).to(device)
edge_weight = None


def test(input=None):
    if input==None:
        gnn=GNNConv(args.conv_name, args.input_dim, args.hidden_dim, args.output_dim, \
            rel_num=rel_num, n_layer=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
        classifier = Classifier(args.output_dim, 2)
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
        rc=roc(y_test_copy,prob.cpu().detach().numpy()[:,1])
        # rc=0
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
            rc=roc(y_valid_copy,prob.cpu().detach().numpy()[:,1])

            if ac > best_acc:
                best_acc = ac
                torch.save(model.state_dict(), './model_save/%s.pkl'%(args.conv_name))

                print('UPDATE!!!')
                up=1

            et = time.time()
            print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.4f  Train Acc: %.2f  Valid Loss: %.4f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f"  ) % \
                (epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses), train_acc, \
                loss.cpu().detach().tolist(), ac,pr,re,f,rc))
            del res, loss

train()
test()
