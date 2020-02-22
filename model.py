from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
# Provide multiple model


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        xavier_normal(m.weight.data)
        xavier_normal(m.bias.data)


def weights_init_1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)



class Model(nn.Module):
    def __init__(self, name, obs_dim, action_dim, hidden = [512,512],activation = nn.ReLU,out_activation =nn.Identity):
        super(Model, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden= hidden
        self.name = name
        net_sizes = [self.obs_dim] +list(hidden)+[self.action_dim]
        self.net = mlp(net_sizes,activation,out_activation)


    def act(self, obs):
        print("Disable backprob")
        return self.forward(obs)


class QModel(Model):
    def __init__(self, name, obs_dim, action_dim):
        super().__init__(name, obs_dim, action_dim)

    def forward(self, obs):
        return "Q(a|s)"


class PiModel(Model):
    def __init__(self, name, obs_dim, action_dim):
        super().__init__(name, obs_dim, action_dim)

    def forward(self, obs):
        return "P(a|s)"
