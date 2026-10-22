import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import dgl.nn as dglnn
import tqdm


class GIN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(
            nn.Sequential(
                nn.Linear(in_size, hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, hid_size),
            ),
            aggregator_type='sum',
            init_eps=0,
            learn_eps=True,
        ))
        for i in range(num_layer - 2):
            self.layers.append(dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(hid_size, hid_size),
                    nn.ReLU(),
                    nn.Linear(hid_size, hid_size),
                ),
                aggregator_type='sum',
                init_eps=0,
                learn_eps=True,
            ))
        self.layers.append(dglnn.GINConv(
            nn.Sequential(
                nn.Linear(hid_size, out_size),
            ),
            aggregator_type='sum',
            init_eps=0,
            learn_eps=True,
        ))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.num_layer = num_layer

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class GIN_FULL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(
            nn.Sequential(
                nn.Linear(in_size, hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, hid_size),
            ),
            aggregator_type='sum',
            init_eps=0,
            learn_eps=True,
        ))
        for i in range(num_layer - 2):
            self.layers.append(dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(hid_size, hid_size),
                    nn.ReLU(),
                    nn.Linear(hid_size, hid_size),
                ),
                aggregator_type='sum',
                init_eps=0,
                learn_eps=True,
            ))
        self.layers.append(dglnn.GINConv(
            nn.Sequential(
                nn.Linear(hid_size, out_size),
            ),
            aggregator_type='sum',
            init_eps=0,
            learn_eps=True,
        ))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.num_layer = num_layer

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
