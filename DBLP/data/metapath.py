# coding: utf-8
import os
os.chdir('./DBLP/data/')
import torch
from os import times
import pandas as pd
import torch
import pickle
import time
from torch import nn
import numpy as np



from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

# from torch import scatter
from torch_geometric.utils import scatter

import warnings
warnings.filterwarnings("ignore")


# from config import args


from copy import deepcopy



#load data
homo='dblp_homo.pkl'
hete='dblp_hete.pkl'

dims= {'author':334,'paper':4231,'term':50,'conference':4231}
# dims=[334,4231,7723,50]


with open(hete, "rb") as f:
    hete_data = pickle.load(f)
    # edge: dict
    node_dict,edge,label,train_idx,valid_idx,test_idx=hete_data  
    edge = {i:edge[i] for i in edge}
    edge_type=None

# with open(homo, "rb") as f:
#     homo_data = pickle.load(f)
#     # edge: torch.LongTensor
#     node_dict,edge,edge_type,label,train_idx,valid_idx,test_idx =homo_data
#     edge = edge
#     edge_type=edge_type

# # A,AP,APA,APC,APT
PA=edge['paper','to','author']

AP=edge['author','to','paper']

PT=edge['paper','to','term']

PC=edge['paper','to','conference']

au=node_dict['author']
pp=node_dict['paper']
te=node_dict['term']
co=node_dict['conference']

ap=scatter(pp[AP[1]],AP[0],dim_size=4057,reduce='mean')
pa=scatter(au[PA[1]],PA[0],dim_size=14328,reduce='mean')
apa=scatter(pa[AP[1]],AP[0],dim_size=4057,reduce='mean')
pt=scatter(te[PT[1]],PT[0],dim_size=14328,reduce='mean')
apt=scatter(pt[AP[1]],AP[0],dim_size=4057,reduce='mean')
pc=scatter(co[PC[1]],PC[0],dim_size=14328,reduce='mean')
apc=scatter(pc[AP[1]],AP[0],dim_size=4057,reduce='mean')

feat=[au,ap,apa,apc,apt]
# print([i.shape for  i in feat])
feat_dim={'A':334,'AP':4231,'APA':334,'APC':4231,'APT':50}
feat = {'A':au,'AP':ap,'APA':apa,'APC':apc,'APT':apt}

with open('metapath_feat.pkl', 'wb') as f:
    pickle.dump([feat_dim,feat], f)