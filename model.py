import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch import Tensor
from torch.nn.modules.module import Module
from torch_geometric.nn.inits import glorot

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.bn = nn.BatchNorm1d(out_ft)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj=None, sparse=False):
        out = self.fc(seq)
        if adj is not None:
            if sparse:
                out = torch.spmm(adj, out)
            else:
                out = torch.mm(adj, out)
        if self.bias is not None:
            out += self.bias
        out = self.bn(out)
        out = self.act(out)
        return out


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h, activation)

    def forward(self, seq, adj):
        feat = self.gcn1(seq, adj)
        return feat

    def reset_parameters(self,):
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb
    
class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = torch.mm(weight, self.p_list)
        return x + p

class Projection(nn.Module):
    def __init__(self, hidden_dim):
        super(Projection, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, feat):
        feat = self.fc1(feat)
        return feat