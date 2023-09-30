import torch
from torch_geometric.datasets import DBLP
import scipy.sparse as sp
import numpy as np
import pickle

data=DBLP(root='./data/').data
node_dict={}
edge_dict={}
for nd in ['paper','author','term', 'conference']:
    if nd == 'conference':
        edge=data['conference', 'to', 'paper']['edge_index']
        v=torch.ones(edge.shape[1]).numpy()
        mx=sp.coo_matrix(arg1=(v,edge.numpy()),shape=(20,14328),dtype=np.float32)
        mx=mx.todense()
        edge_dict['conference', 'to', 'paper']=edge
        edge_dict['paper', 'to', 'conference']=torch.stack((edge[1],edge[0]),dim=0)
        emb=np.matmul(mx,data['paper'].x.numpy())
        dg=mx.sum(axis=1)
        node_dict['conference']=torch.from_numpy(emb/dg)
    else:
        node_dict[nd]=data[nd].x

label=data['author'].y
train_idx=torch.nonzero(data['author']['train_mask'].bool())[:,0]
valid_idx=torch.nonzero(data['author']['val_mask'].bool())[:,0]
test_idx=torch.nonzero(data['author']['test_mask'].bool())[:,0]

edge_dict['author', 'to', 'paper']=data['author', 'to', 'paper']['edge_index']
edge_dict['paper', 'to', 'author']=data['paper', 'to', 'author']['edge_index']
edge_dict['paper', 'to', 'term']=data['paper', 'to', 'term']['edge_index']
edge_dict['term', 'to', 'paper']=data['term', 'to', 'paper']['edge_index']



with open('./data/dblp_hete.pkl', 'wb') as f:
    res= [node_dict,edge_dict,label,train_idx,valid_idx,test_idx]
    pickle.dump(res, f)

num=0
num_dict={}
for i in ['author', 'paper', 'term', 'conference']:
    num_dict[i]=num
    num+=node_dict[i].shape[0]


edge_type=[]

edge_dict['author', 'to', 'paper'][1]+=num_dict['paper']
edge_type+=[torch.ones(edge_dict['author', 'to', 'paper'].shape[1])*0]

edge_dict['paper', 'to', 'author'][0]+=num_dict['paper']
edge_type+=[torch.ones(edge_dict['paper', 'to', 'author'].shape[1])*0]


edge_dict['paper', 'to', 'term'][0]+=num_dict['paper']
edge_dict['paper', 'to', 'term'][1]+=num_dict['term']
edge_type+=[torch.ones(edge_dict['paper', 'to', 'term'].shape[1])*1]


edge_dict['term', 'to', 'paper'][0]+=num_dict['term']
edge_dict['term', 'to', 'paper'][1]+=num_dict['paper']
edge_type+=[torch.ones(edge_dict['term', 'to', 'paper'].shape[1])*1]


edge_dict['conference', 'to', 'paper'][0]+=num_dict['conference']
edge_dict['conference', 'to', 'paper'][1]+=num_dict['paper']
edge_type+=[torch.ones(edge_dict['conference', 'to', 'paper'].shape[1])*2]


edge_dict['paper', 'to', 'conference'][0]+=num_dict['paper']
edge_dict['paper', 'to', 'conference'][1]+=num_dict['conference']
edge_type+=[torch.ones(edge_dict['paper', 'to', 'conference'].shape[1])*2]

edge_homo=[edge_dict[i] for i in edge_dict]
edge_homo=torch.concat(edge_homo,dim=1)
edge_type=torch.concat(edge_type)
print(edge_homo.shape)
print(edge_type.shape)


with open('./data/dblp_homo.pkl', 'wb') as f:
    res= [node_dict,edge_homo,edge_type,label,train_idx,valid_idx,test_idx]
    pickle.dump(res, f)