import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import tqdm

# Adapted from DGL's GAT example: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer=2, heads=[8,1]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
        )

        for i in range(num_layer-2):
            self.layers.append(dglnn.GATConv(hid_size * heads[i], out_size, heads[i+1], feat_drop=0.6, attn_drop=0.6, activation=None, allow_zero_in_degree=True))

        self.layers.append(
            dglnn.GATConv(
                hid_size * heads[num_layer-2],
                out_size,
                heads[num_layer-1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
                allow_zero_in_degree=True,
            )
        )

        self.hid_size = hid_size
        self.out_size = out_size
        self.dropout = nn.Dropout(0.5)
        self.heads = heads


    def forward(self, blocks, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i == 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            if l == 0:
                y = torch.empty(g.num_nodes(), self.heads[0]*self.hid_size,
                                device=buffer_device, pin_memory=pin_memory)
            elif l == len(self.layers) - 1:
                y = torch.empty(g.num_nodes(), self.out_size,
                                device=buffer_device, pin_memory=pin_memory)
            else:
                y = torch.empty(
                    g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                    device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l == 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)

                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class GAT_FULL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer=2, heads=[8, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
        )
        for i in range(num_layer - 2):
            self.layers.append(
                dglnn.GATConv(
                    hid_size * heads[i],
                    hid_size,
                    heads[i + 1],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.layers.append(
            dglnn.GATConv(
                hid_size * heads[num_layer - 2],
                out_size,
                heads[num_layer - 1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
                allow_zero_in_degree=True,
            )
        )
        self.hid_size = hid_size
        self.out_size = out_size
        self.dropout = nn.Dropout(0.5)
        self.heads = heads

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i == len(self.layers) - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h
