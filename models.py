""" MLP models """

import torch
from torch import nn
from torch.nn import Parameter

class MLP(nn.Module):
    def __init__(self, init_W_list, activation):
        super(MLP, self).__init__()
        self.W_list = nn.ParameterList([])
        for i in range(len(init_W_list)):
            self.W_list.append(Parameter(init_W_list[i].clone()))
        self.activation = activation

    def forward(self, X):
        h = X.clone()
        for i in range(len(self.W_list)-1):
            h = self.activation(torch.matmul(self.W_list[i], h))

        out = torch.matmul(self.W_list[-1], h)
        return out

    def get_W_list(self):
        W_list = []
        for param in self.W_list:
            W_list.append(param.data.clone())
        return W_list