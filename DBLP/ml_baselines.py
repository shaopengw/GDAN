from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc
import pickle
import torch
import argparse
import numpy as np
import os
os.chdir('./DBLP/')

parser = argparse.ArgumentParser(description='Training GNN')
'''
    Dataset arguments
'''
parser.add_argument('--ml_model', type=str, default='svm',
                    choices=['lr','svm','dt','gbdt'],
                    help='machine learning method.')
parser.add_argument('--dataset', type=str, default='credit',
                    choices=['credit','trend'],
                    help='dataset to use.')

args = parser.parse_args()


hete='./data/'+'dblp_hete.pkl'


with open(hete, "rb") as f:
    hete_data = pickle.load(f)
node_dict,edge,label,train_idx,valid_idx,test_idx=hete_data  
with open('./data/index.pkl', "rb") as f:
    idxs = pickle.load(f)
train_idx,valid_idx,test_idx=idxs

x_train = node_dict['author'][train_idx].numpy()
x_test = node_dict['author'][test_idx].numpy()
y_train = label[train_idx].numpy()
y_test =label[test_idx].numpy()


label_num = 4



if args.ml_model in ['lr','svm','dt','gbdt']:
    if args.ml_model=='lr':
        model = LogisticRegression(random_state=2023).fit(x_train, y_train)
        prob=model.predict_proba(x_test)
        res=[np.argmax(i) for i in prob]
        print(res)
        print(y_test)

        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f ' % 
            (acc(y_test,res),pre(y_test,res,average='macro'),\
             rec(y_test,res,average='macro'),\
                f1(y_test,res, average='macro',)))
        print('Test roc: %.4f '%roc(y_test,prob,average='macro',multi_class='ovo'))
    elif args.ml_model=='svm':
        model = svm.SVC(probability=True,random_state=2023)
        model.fit(x_train, y_train)
        res=model.predict(x_test)
        print(res)
        print(y_test)

        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f ' % 
            (acc(y_test,res),pre(y_test,res,average='macro'),\
             rec(y_test,res,average='macro'),\
                f1(y_test,res, average='macro',)))
        print('Test roc: %.4f '%roc(y_test,model.predict_proba(x_test),average='macro',multi_class='ovo'))
    elif args.ml_model=='dt':
        model = DecisionTreeClassifier(random_state=2022,max_features=20,max_depth=20) 
        model.fit(x_train, y_train) 

        res = model.predict(x_test)
        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f ' % 
            (acc(y_test,res),pre(y_test,res,average='macro'),\
             rec(y_test,res,average='macro'),\
                f1(y_test,res, average='macro')))
        print('Test roc: %.4f '%roc(y_test,model.predict_proba(x_test),average='macro',multi_class='ovo'))
    elif args.ml_model=='gbdt':
        model = GradientBoostingClassifier(random_state=2023)

        model.fit(x_train, y_train)
        res=model.predict(x_test)
        print(res)
        print(y_test)
        predprob = model.predict_proba(x_test)[:, 1]
        print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f ' % 
            (acc(y_test,res),pre(y_test,res,average='macro'),\
             rec(y_test,res,average='macro'),\
                f1(y_test,res, average='macro',)))
        print('Test roc: %.4f '%roc(y_test,model.predict_proba(x_test),average='macro',multi_class='ovo'))

    exit()