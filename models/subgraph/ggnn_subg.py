import torch
import tqdm

import torch.nn.functional as F

from torch import nn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

from dgl.nn.pytorch import GatedGraphConv


class GGNN_subg(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, heads, num_steps=5):
        super(GGNN_subg, self).__init__()

        self.layers = nn.ModuleList()
        self.lin = nn.ModuleList()
        self.out_size = out_size

        if in_size <= out_size:
            self.layers.append(GatedGraphConv(in_size, out_size, num_steps, n_etypes=1))
        elif in_size > out_size:
            print(f"in dim {in_size} > out dim {out_size}")
            self.layers = nn.ModuleList()
            self.layers.append(GatedGraphConv(in_size, in_size, num_steps, n_etypes=1))
            self.lin.append(nn.Linear(in_size, out_size, bias=False))
        else:
            print("Error in GGNN layer dimensions")

    def forward(self, g, h):
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(g, h))

        if len(self.lin) > 0:
            for i, layer in enumerate(self.lin):
                h = layer(h)

        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        sampler = None
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
                h = F.relu(layer(blocks[0], x))
                if len(self.lin) > 0:
                    for i, lay in enumerate(self.lin):
                        h = lay(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
