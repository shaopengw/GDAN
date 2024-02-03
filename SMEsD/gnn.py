import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import softmax, scatter
import math
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm



class GNNConv(nn.Module):
    def __init__(self,conv_name, input_dim,hidden_dim,output_dim,rel_num, n_layer, n_heads=1,dropout=0):
        super(GNNConv, self).__init__()
        self.conv_name=conv_name
        self.base_conv = RiskGNN(input_dim,hidden_dim,output_dim,rel_num, n_layer, h=n_heads)
    def forward(self,x,edge_index,idx,edge_type,edge_weight):
        return self.base_conv(x,edge_index,idx,edge_type,edge_weight)

#DAN (dimension attention networks)
class RiskGNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,rel_num, n_layer=2, h=3):
        super(RiskGNN, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.rel_num=rel_num
        self.n_layer=n_layer
        self.head=h

        self.norm=nn.BatchNorm1d(hidden_dim)
        self.proj=nn.Linear(input_dim,hidden_dim)

        self.gnn_layer=nn.ModuleList()
        for i in range(n_layer):
            if i==0:
                self.gnn_layer.append(DAN(hidden_dim,output_dim,rel_num, h=3))
            else:
                self.gnn_layer.append(DAN(output_dim,output_dim,rel_num, h=3))


        self.alpha=torch.ones((1))
        self.drop=nn.Dropout(0.5)


    def forward(self,x,edge_index,idx,edge_type,edge_weight):
        x=F.relu(self.norm(self.proj(x)))
        contagion_risk=x

        for i in range(self.n_layer):
            contagion_risk=self.gnn_layer[i](contagion_risk,edge_index,edge_type,edge_weight)
        contagion_risk= self.drop(F.gelu(contagion_risk))

        return contagion_risk[idx]


class DAN(nn.Module):
    def __init__(self,hidden_dim,output_dim,rel_num, h=3):
        super(DAN, self).__init__()
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.rel_num=rel_num
        self.head=h

        self.W_rel=nn.Parameter(torch.empty(rel_num,hidden_dim, hidden_dim))
        init.kaiming_uniform_(self.W_rel, a=math.sqrt(5))
        self.proj_gcn= nn.Linear(hidden_dim, hidden_dim)
        self.norm_gcn = nn.LayerNorm(hidden_dim)

        self.W_input=nn.Linear(hidden_dim,hidden_dim,bias=True)
        self.contagion_out=nn.Linear(hidden_dim,output_dim,bias=True)

        self.norm=nn.LayerNorm(output_dim)

        self.norm_weight=None

    def forward(self,contagion_risk,edge_index,edge_type,edge_weight):
        x=self.W_input(contagion_risk)
        x_j=x[edge_index[1]]

        _, norm_weight = gcn_norm(edge_index, num_nodes=3976,add_self_loops=False)
        norm_weight= norm_weight.unsqueeze(1)
        if edge_weight != None:
            edge_weight_sum = scatter(edge_weight,index=edge_index[0], dim=0, reduce='sum')
            edge_weight_sum = edge_weight_sum[edge_index[0]]
            edge_weight = (edge_weight/edge_weight_sum).unsqueeze(1)
            msg=x_j* (self.norm_weight+edge_weight)
            msg_gcn=scatter(msg,edge_index[0],dim=0,dim_size=3976)  #    nxd
        else:
            msg_gcn=self.norm_gcn(F.relu(self.proj_gcn(x))) + scatter(x_j* norm_weight,edge_index[0],dim=0,dim_size=3976)  #    nxd

        res=torch.zeros_like(x_j)
        for i in range(self.rel_num):
            mask=(edge_type==i)
            res[mask]=x_j[mask] @ self.W_rel[i]
        x_j=res

        weig=softmax(x_j,index=edge_index[0], dim=0)
        msg=scatter(x_j* weig,edge_index[0],dim=0,dim_size=3976)  #    nxd

        contagion_risk=self.contagion_out(msg_gcn + 0.1*F.relu(msg))


        return self.norm(contagion_risk)

