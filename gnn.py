from pickle import FALSE
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,RGCNConv,HGTConv,HANConv
from torch import softmax, scatter
import math

from torch_geometric.utils import softmax
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_scatter import scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm



class GNNConv(nn.Module):
    def __init__(self,conv_name, input_dim,hidden_dim,output_dim,rel_num, n_layer, n_heads=1,dropout=0):
        super(GNNConv, self).__init__()
        self.conv_name=conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCN(input_dim,hidden_dim,output_dim,n_layer=n_layer)
        elif self.conv_name == 'gat':
            self.base_conv = GAT(input_dim, hidden_dim, output_dim,n_heads=n_heads,dropout=dropout,n_layer=n_layer)
        elif self.conv_name=='rgcn':
            self.base_conv = RGCN(input_dim,hidden_dim, output_dim, rel_num, n_layer)
        elif self.conv_name=='han':
            self.base_conv = HAN(input_dim,hidden_dim, output_dim,rel_num, n_layer, heads=n_heads)
        elif self.conv_name=='hgt':
            self.base_conv = HGT(input_dim,hidden_dim, output_dim,rel_num, n_layer, heads=n_heads)
        elif self.conv_name=='riskgnn':
            self.base_conv = RiskGNN(input_dim,hidden_dim,output_dim,rel_num, n_layer, h=n_heads)
        elif self.conv_name == 'sehgnn':
            self.base_conv= SeHGNN(input_dim,hidden_dim, output_dim,rel_num)

    def forward(self,x,edge_index,idx,edge_type,edge_weight):
        if self.conv_name == 'gcn':
            return self.base_conv(x,edge_index,idx)
        elif self.conv_name == 'gat':
            return self.base_conv(x,edge_index,idx)
        elif self.conv_name=='rgcn':
            return self.base_conv(x,edge_index,idx,edge_type)
        elif self.conv_name=='han':
            return self.base_conv(x,edge_index,idx,edge_type)
        elif self.conv_name=='hgt':
            return self.base_conv(x,edge_index,idx,edge_type)
        elif self.conv_name=='riskgnn':
            return self.base_conv(x,edge_index,idx,edge_type,edge_weight)
        elif self.conv_name == 'sehgnn':
            return self.base_conv(x,edge_index, edge_type, idx)

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

        self.dim_proj=nn.Linear(hidden_dim,output_dim)
        self.W_intra=nn.ModuleList()

        self.contagion=nn.Linear(input_dim,hidden_dim,bias=True)


        self.gnn_layer=nn.ModuleList()
        
        for i in range(n_layer):
            if i==0:
                self.gnn_layer.append(DAN(hidden_dim,output_dim,rel_num, h=3))
            else:
                self.gnn_layer.append(DAN(output_dim,output_dim,rel_num, h=3))
            

        self.alpha=torch.ones((1))
        self.drop=nn.Dropout(0.5)
        

    def forward(self,x,edge_index,idx,edge_type,edge_weight):
        contagion_risk=F.relu(self.norm(self.proj(x)))

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

        self.W_input=nn.Linear(hidden_dim,hidden_dim,bias=True)
        self.contagion_out=nn.Linear(hidden_dim,output_dim,bias=True)

        self.norm=nn.LayerNorm(output_dim)

        self.norm_weight=None
    

    def forward(self,contagion_risk,edge_index,edge_type,edge_weight):
        x=self.W_input(contagion_risk)
        x_j=x[edge_index[1]]

        _, norm_weight = gcn_norm(edge_index, num_nodes=48758,add_self_loops=False)
        norm_weight= norm_weight.unsqueeze(1)
        msg_gcn=scatter(x_j* norm_weight,edge_index[0],dim=0,dim_size=48758)  #    nxd



        res=torch.zeros_like(x_j)
        for i in range(self.rel_num):
            mask=(edge_type==i)
            res[mask]=x_j[mask] @ self.W_rel[i]
        x_j=res
        
        weig=softmax(x_j,index=edge_index[0], dim=0)
        msg=scatter(x_j* weig,edge_index[0],dim=0,dim_size=48758)  #    nxd
        
        contagion_risk=self.contagion_out(msg_gcn + 0.1*F.relu(msg))

        return contagion_risk


        
       
class HGT(nn.Module):
    def __init__(self,input_dim,hidden_dim, output_dim,rel_num, n_layer=2, heads=2):
        super(HGT, self).__init__()
        self.n_layer=n_layer
        self.rel_num=rel_num
        self.proj=nn.Linear(input_dim,hidden_dim)
        self.norm=nn.BatchNorm1d(hidden_dim)
        self.gat=nn.ModuleList()
        edge_list=[]
        for i in range(rel_num):
            edge_list.append(['person',str(i),'person'])
        metadata=(['person'],edge_list)

        for i in range(n_layer):
            if i==0:
                self.gat.append(HGTConv(hidden_dim, output_dim , heads=heads,metadata=metadata))
            else:
                self.gat.append(HGTConv(output_dim, output_dim , heads=heads,metadata=metadata))

    def forward(self,x,edge_index,idx,edge_type):
        x=self.norm(self.proj(x))

        x_dict={'person':x}
        edge_index_dict={}
        for i in range(self.rel_num):
            mask=(edge_type==i)
            edge=(edge_index.transpose(0,1)[mask]).transpose(0,1)
            edge_index_dict['person',str(i),'person']=edge
        for i in range(self.n_layer):
            x_dict= self.gat[i](x_dict, edge_index_dict)
        return x_dict['person'][idx]
    

class HAN(nn.Module):
    def __init__(self,input_dim,hidden_dim, output_dim,rel_num, n_layer=2, heads=2):
        super(HAN, self).__init__()
        self.n_layer=n_layer
        self.rel_num=rel_num
        self.proj=nn.Linear(input_dim,hidden_dim)
        self.norm=nn.BatchNorm1d(hidden_dim)
        self.gat=nn.ModuleList()
        edge_list=[]
        for i in range(rel_num):
            edge_list.append(['person',str(i),'person'])
        metadata=(['person'],edge_list)

        for i in range(n_layer):
            if i==0:
                self.gat.append(HANConv(hidden_dim, output_dim , heads=heads,metadata=metadata))
            else:
                self.gat.append(HANConv(output_dim, output_dim , heads=heads,metadata=metadata))

    def forward(self,x,edge_index,idx,edge_type):
        x=self.norm(self.proj(x))

        x_dict={'person':x}
        edge_index_dict={}
        for i in range(self.rel_num):
            mask=(edge_type==i)
            edge=(edge_index.transpose(0,1)[mask]).transpose(0,1)
            edge_index_dict['person',str(i),'person']=edge
        for i in range(self.n_layer):
            x_dict= self.gat[i](x_dict, edge_index_dict)
        return x_dict['person'][idx]







class GAT(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,n_layer,n_heads,dropout=0):
        super(GAT, self).__init__()
        self.n_layer=n_layer
        self.proj=nn.Linear(input_dim,hidden_dim)
        self.norm=nn.BatchNorm1d(hidden_dim)
        self.gat=nn.ModuleList()
        for i in range(n_layer):
            if i==0:
                self.gat.append(GATConv(hidden_dim, output_dim // n_heads, heads=n_heads,dropout=dropout))
            else:
                self.gat.append(GATConv(output_dim, output_dim // n_heads, heads=n_heads,dropout=dropout))

    def forward(self,x,edge_index,idx):
        x=self.norm(self.proj(x))
        for i in range(self.n_layer):
            x= self.gat[i](x, edge_index)
        return x[idx]

class GCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,n_layer=2):
        super(GCN, self).__init__()
        self.n_layer=n_layer
        self.proj=nn.Linear(input_dim,hidden_dim)
        self.norm=nn.BatchNorm1d(hidden_dim)
        self.gat=nn.ModuleList()
        for i in range(n_layer):
            if i==0:
                self.gat.append(GCNConv(hidden_dim, output_dim ))
            else:
                self.gat.append(GCNConv(output_dim, output_dim ))

    def forward(self,x,edge_index,idx):
        x=self.norm(self.proj(x))
        for i in range(self.n_layer):
            x= self.gat[i](x, edge_index)
        return x[idx]


class RGCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,rel_num, n_layer=2):
        super(RGCN, self).__init__()
        self.n_layer=n_layer
        self.proj=nn.Linear(input_dim,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
        self.gat=nn.ModuleList()
        if n_layer==1:
            self.gat.append(RGCNConv(hidden_dim, output_dim, num_relations=rel_num ))
        else:
            for i in range(n_layer):
                if i==0:
                    self.gat.append(RGCNConv(hidden_dim, hidden_dim ,num_relations=rel_num ))
                elif i==n_layer-1:
                    self.gat.append(RGCNConv(hidden_dim, output_dim, num_relations=rel_num ))
                else:
                    self.gat.append(RGCNConv(hidden_dim, hidden_dim, num_relations=rel_num ))

    def forward(self,x,edge_index,idx,edge_type):
        x=self.norm(self.proj(x))
        for i in range(self.n_layer):
            x= self.gat[i](x, edge_index, edge_type)
        return x[idx]



class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, independent_attn=False):
        super(SemanticAttention, self).__init__()
        self.independent_attn = independent_attn

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z): # z: (N, M, D)
        w = self.project(z) # (N, M, 1)
        if not self.independent_attn:
            w = w.mean(0, keepdim=True)  # (1, M, 1)
        beta = torch.softmax(w, dim=1)  # (N, M, 1) or (1, M, 1)

        return (beta * z).sum(1)  # (N, M, D)


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False

class SeHGNN(nn.Module):
    def __init__(self, nfeat, hidden, nclass,rel_num,
                 dropout=0.5, input_drop=0.1, att_dropout=0, label_drop=0,
                 n_layers_1=2, n_layers_2=3, act=None,
                 residual=False, bns=False, data_size=None, drop_metapath=0., num_heads=1,
                 remove_transformer=False, independent_attn=False):
        super(SeHGNN, self).__init__()

        self.residual = residual
        self.num_channels = num_channels=rel_num
        self.rel_num=rel_num
        self.remove_transformer = remove_transformer

        self.embeding=nn.ParameterDict({})
        for i in range(rel_num):
            self.embeding[str(i)] = nn.Parameter(
                torch.Tensor(hidden, hidden).uniform_(-0.5, 0.5))

        self.layers = nn.Sequential(
            Conv1d1x1(hidden, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),

            Conv1d1x1(hidden, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        if self.remove_transformer:
            self.layer_mid = SemanticAttention(hidden, hidden, independent_attn=independent_attn)
            self.layer_final = nn.Linear(hidden, hidden)
        else:
            self.layer_mid = Transformer(hidden, num_heads=num_heads)
            self.layer_final = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(dropout, bns=False):
            return [
                nn.BatchNorm1d(hidden, affine=bns, track_running_stats=bns),
                nn.PReLU(),
                nn.Dropout(dropout)
            ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)]))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.proj=nn.Linear(nfeat,hidden)
        self.norm=nn.BatchNorm1d(hidden)

        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.layer_final.weight, gain=gain)
        nn.init.zeros_(self.layer_final.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    def forward(self, features,edge_index, edge_type, idx):
        features=F.relu(self.norm(self.proj(features)))
        mapped_feats = []
        for i in range(self.rel_num):
            mask=(edge_type==i)
            rel_edge_src=edge_index[0][mask]
            rel_edge_dst=edge_index[1][mask]
            rel_edge= edge_index[:,mask]
            _, norm_weight = gcn_norm(rel_edge, num_nodes=48758,add_self_loops=False)
            norm_weight=norm_weight.unsqueeze(1)
            msg_gcn=scatter(features[rel_edge_dst]* norm_weight,rel_edge_src,dim=0,dim_size=48758)
            mapped_feats += [self.input_drop(msg_gcn) @ self.embeding[str(i)] ]
        features =torch.stack(mapped_feats,dim=1)


        B = features.shape[0]
        
        features = self.layers(features)
        if self.remove_transformer:
            features = self.layer_mid(features)
        else:
            features = self.layer_mid(features, mask=None).transpose(1,2)
        out = self.layer_final(features.reshape(B, -1))

        if self.residual:
            out = out + self.res_fc(features)
        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)
        return out[idx]