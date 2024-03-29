# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math

class LinearFloat64(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearFloat64, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return nn.functional.linear(input, self.weight.type(torch.float64), self.bias.type(torch.float64))

linear_layer = LinearFloat64(in_features=10, out_features=5)
def activation_layer(act_name):
    if act_name.lower() == 'sigmoid':
        act_layer = nn.Sigmoid()

    elif act_name.lower() == 'relu':
        act_layer = nn.ReLU(inplace=True)
    elif act_name.lower() == 'tahn':
        act_layer = nn.Tanh()
    elif act_name.lower() == 'prelu':
        act_layer = nn.PReLU()
    elif act_name.lower() == 'elu':
        act_layer = nn.ELU()
    elif act_name.lower() == 'lrelu':
        act_layer = nn.LeakyReLU()
    return act_layer

class Net(nn.Module):
    def __init__(self, config, kinds, inputs_dim):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.kinds = kinds
        if kinds != 0:
            hidden_units = [inputs_dim + config.embedding_dim - 1] + list(config.hidden_units)
        else:
            hidden_units = [inputs_dim] + list(config.hidden_units)

        self.linears = nn.ModuleList(
        [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        if kinds !=0:
            self.embedding = nn.Embedding(kinds,
                                          config.embedding_dim, device=config.device)


            nn.init.normal_(self.embedding.weight, mean=0, std=config.init_std)
        self.activation_layers = nn.ModuleList(
            [activation_layer(config.activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            tensor = tensor.double()
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=config.init_std)
        self.dense = nn.Sigmoid()

    def forward(self, inputs):
        deep_input = inputs

        if self.kinds !=0:
            sparse_input = self.embedding(inputs[:, -1:].long())
            cur_input = sparse_input.squeeze()
            deep_input = deep_input[:, :-1]
            deep_input = torch.cat((deep_input, cur_input), dim=1)


        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        fc = self.dense(deep_input)

        return fc
