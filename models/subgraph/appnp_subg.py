"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""

import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from dgl.nn.pytorch.conv import APPNPConv


class APPNP_subg(nn.Module):
    def __init__(
        self,
        in_size,
        hid_size,
        out_size,
        num_layer,
        heads,
        activation=F.relu,
        feat_drop=0.5,
        edge_drop=0.5,
        alpha=0.1,
        k=10,
    ):
        super(APPNP_subg, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hid_size))
        for i in range(1, num_layer - 1):
            self.layers.append(nn.Linear(hid_size, hid_size))
        self.layers.append(nn.Linear(hid_size, out_size))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        h = self.propagate(g, h)
        return h
