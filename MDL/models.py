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
    act_layer = None
    if act_name.lower() == 'sigmoid':
        act_layer = nn.Sigmoid()
    # elif act_name.lower() == 'linear':
    #     act_layer = Identity()
    elif act_name.lower() == 'relu':
        act_layer = nn.ReLU(inplace=True)
    elif act_name.lower() == 'tahn':
        act_layer = nn.Tanh()
    # elif act_name.lower() == 'dice':
    #     assert dice_dim
    #     act_layer = Dice(hidden_size, dice_dim)
    elif act_name.lower() == 'prelu':
        act_layer = nn.PReLU()
    elif act_name.lower() == 'elu':
        act_layer = nn.ELU()
    elif act_name.lower() == 'lrelu':
        act_layer = nn.LeakyReLU()
    return act_layer


class Net(nn.Module):
    def __init__(self, config, inputs_dim):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        hidden_units = [inputs_dim] + list(config.hidden_units)

        self.linears = nn.ModuleList(
        [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(config.activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            tensor = tensor.double()
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=config.init_std)
        self.dense = nn.Sigmoid()

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        fc = self.dense(deep_input)

        return fc

class mulNet(nn.Module):
    def __init__(self, configmul, data, group_index, sparse_group_index):
        super(mulNet, self).__init__()
        self.device = configmul.device
        self.group_index = group_index
        self.sparse_group_index = sparse_group_index
        self.firsts = nn.ModuleList()
        self.group_linears = nn.ModuleList()
        self.group_hidden_units = configmul.group_hidden_units
        self.regularization_weight = []

        group_out = 0
        for i in range(len(group_index)):
            self.firsts.append(nn.Linear(group_index[i][1] - group_index[i][0], self.group_hidden_units[i][0],device=configmul.device))
            units = self.group_hidden_units[i]
            self.group_linears.append(nn.ModuleList([nn.Linear(units[i], units[i + 1],device=configmul.device) for i in range(len(units) - 1)]))
            group_out += units[-1]

        self.group_activation_layers = activation_layer(configmul.group_activation)
        self.dropout = nn.Dropout(configmul.dropout_rate)


        for first in self.firsts:
            nn.init.normal_(first.weight.data, mean=0, std=configmul.init_std)


        for linears in self.group_linears:
            for name, tensor in linears.named_parameters():
                tensor = tensor.double()
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=configmul.init_std)

        self.add_regularization_weight(self.firsts.parameters(), l2=configmul.l2_reg)
        if len(sparse_group_index) != 0:

            self.embedding = nn.Embedding(data.iloc[:, sparse_group_index[0]:sparse_group_index[1]].nunique().values[0], configmul.embedding_dim, device=configmul.device)

            nn.init.normal_(self.embedding.weight, mean=0, std=configmul.init_std)

            self.add_regularization_weight(self.embedding.parameters(), l2=configmul.l2_reg)

            self.sparse_hidden_units = [configmul.embedding_dim] + list(configmul.sparse_hidden_units)
            self.sparse_linears = nn.ModuleList(
                [nn.Linear(self.sparse_hidden_units[i], self.sparse_hidden_units[i + 1], device=configmul.device) for i in
                 range(len(self.sparse_hidden_units) - 1)])

            for name, tensor in self.sparse_linears.named_parameters():
                tensor = tensor.double()
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=configmul.init_std)

            self.add_regularization_weight(self.sparse_linears.parameters(), l2=configmul.l2_reg)
        if len(sparse_group_index) != 0:
            self.concat_hidden_units = [group_out + configmul.sparse_hidden_units[-1]] + list(configmul.concat_hidden_units)
        else:
            self.concat_hidden_units = [group_out] + list(configmul.concat_hidden_units)

        self.concat_linears = nn.ModuleList(
            [nn.Linear(self.concat_hidden_units[i], self.concat_hidden_units[i + 1], device=configmul.device) for i in
             range(len(self.concat_hidden_units) - 1)])

        self.concat_activation_layers = activation_layer(configmul.concat_activation)
        for name, tensor in self.concat_linears.named_parameters():
            tensor = tensor.double()
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=configmul.init_std)
        self.dense = nn.Sigmoid()

        self.add_regularization_weight(self.concat_linears.parameters(), l2=configmul.l2_reg)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss






    def forward(self, inputs):
        groups = []
        for i in range(len(self.group_index)):
            cur_input = self.firsts[i](inputs[:, self.group_index[i][0]:self.group_index[i][1]])
            for j in range(len(self.group_linears[i])):
                fc = self.group_linears[i][j](cur_input)
                fc = self.group_activation_layers(fc)
                fc = self.dropout(fc)
                cur_input = fc
            groups.append(cur_input)
        if len(self.sparse_group_index) != 0:
            sparse_input = self.embedding(inputs[:, self.sparse_group_index[0]:self.sparse_group_index[1]].long())
            cur_input = sparse_input.squeeze()
            for i in range(len(self.sparse_linears)):
                fc = self.sparse_linears[i](cur_input)
                fc = self.group_activation_layers(fc)
                fc = self.dropout(fc)
                cur_input = fc
            groups.append(cur_input)

        input_cat = torch.cat(groups, dim=1)
        cur_input = input_cat
        for i in range(len(self.concat_linears)):
            fc = self.concat_linears[i](cur_input)
            fc = self.concat_activation_layers(fc)
            fc = self.dropout(fc)
            cur_input = fc
        fc = self.dense(cur_input)
        return fc
