from abc import ABC

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module, ABC):
    def __init__(self, input_dim, output_dim, hidden_dims, activation='tanh', last_activation='identity', biases=None):
        super(MLP, self).__init__()
        sizes_list = hidden_dims.copy()
        self.activation = getattr(torch, activation)
        self.last_activation = getattr(torch, last_activation)
        sizes_list.insert(0, input_dim)
        biases = [True] * len(sizes_list) if biases is None else biases.copy()

        layers = []
        if 1 < len(sizes_list):
            for i in range(len(sizes_list) - 1):
                layers.append(nn.Linear(sizes_list[i], sizes_list[i + 1], bias=biases[i]))
        self.last_layer = nn.Linear(sizes_list[-1], output_dim)
        self.layers = nn.ModuleList(layers)

    # noinspection PyTypeChecker
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        x = self.last_activation(x)
        return x

    # noinspection PyTypeChecker
    def init(self, init_fn, last_init_fn):
        for layer in self.layers:
            init_fn(layer)
        last_init_fn(self.last_layer)


def soft_update(source_model: nn.Module, target_model: nn.Module, tau):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def init(module, weight_init=None, bias_init=None):
    if weight_init:
        weight_init(module.weight.data)
    if bias_init:
        bias_init(module.bias.data)


def get_flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

