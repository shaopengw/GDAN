import torch
import random
import numpy as np
import os
from torch import nn

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)
        # self.l2=nn.Linear(n_hid,n_out)
        # self.relu=nn.ReLU()

    def forward(self, x):
        tx = self.linear(x)
        # tx=self.l2(self.relu(tx))
        # return torch.log_softmax(tx.squeeze(), dim=-1)
        # return torch.softmax(tx.squeeze(), dim=-1)
        return tx.squeeze()

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, device='cuda:0'):
        super(FocalLoss, self).__init__()
        self.device=device
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
 
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
class weighted_CELoss(nn.Module):
    def __init__(self, ):
        super(weighted_CELoss, self).__init__()

    def forward(self, res, y):
        prob=-torch.log_softmax(res,dim=1)
        N = res.size(0)
        C = res.size(1)

        class_mask = res.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = y.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        # w=torch.exp((prob.sum(1)).view(-1,1))
        w=torch.exp((torch.softmax(res,dim=1)*prob).sum(1).view(-1,1))

        probs = (w*prob*class_mask).sum(1).view(-1,1)

        loss = probs.mean()

        return loss


def initializae_company_info(risk_data,company_attr,company_num,cause_type_num,court_type,category,idx=None):
    if idx:
        idx_dict={index:ser for ser,index in enumerate(idx)}
    company_risk=np.zeros((company_num,cause_type_num+court_type+category+1))
    for index in risk_data:
        risk_info=risk_data[index]
        cause_info=[0 for i in range(cause_type_num)]
        court_info=[0 for i in range(court_type)]
        res_info=[0 for i in range(category)]
        time_info=[]
        for i in range(len(risk_info)):
            justify=risk_info[i]
            cause=justify[0]
            court=justify[1]
            res=justify[2]
            time=justify[3]
            cause_info[cause]+=1
            court_info[court]+=1
            res_info[res]+=1
            time_info+=[time]
        time_ave=[np.average(time_info)]
        if idx:
            company_risk[idx_dict[index]]=np.concatenate((cause_info,court_info,res_info,time_ave),axis=0)
        else:
            company_risk[index]=np.concatenate((cause_info,court_info,res_info,time_ave),axis=0)
    company_attr=np.array(company_attr)
    company_info=np.concatenate((company_attr,company_risk),axis=1)
    return company_info