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


result_path = './DatasetProcessingCode/res/'


data=result_path+'CED.pkl'

with open(data, "rb") as f:
    dataset = pickle.load(f)

node_feat,graph,idx ,label = dataset

edge,edge_type,edge_weight= graph
train_data_idx, valid_data_idx, test_data_idx= idx


train_idx = list(set(list(train_data_idx)+list(valid_data_idx)) & set(label.keys()))
test_idx = list(set(test_data_idx) & set(label.keys()))

x_train = node_feat[torch.LongTensor(train_idx)].numpy()
y_train = [label[i] for i in train_idx if i in label]
x_test = node_feat[torch.LongTensor(test_idx)].numpy()
y_test = [label[i] for i in test_idx if i in label]



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