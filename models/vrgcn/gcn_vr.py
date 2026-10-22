import dgl
import torch
import torch.nn as nn
import tqdm
import dgl.function as fn
import torch.nn.functional as F


# Adapted from DGL's VR-GCN example: https://github.com/dmlc/dgl/blob/master/examples/pytorch/vrgcn/train_cv.py


class SAGEConvWithCV(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, block, H, HBar=None):
        if self.training:
            with block.local_scope():
                H_src, H_dst = H
                HBar_src, agg_HBar_dst = HBar
                block.dstdata["agg_hbar"] = agg_HBar_dst
                block.srcdata["hdelta"] = H_src - HBar_src
                block.update_all(fn.copy_u("hdelta", "m"), fn.mean("m", "hdelta_new"))
                h_neigh = block.dstdata["agg_hbar"] + block.dstdata["hdelta_new"]
                h = self.W(torch.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h
        else:
            with block.local_scope():
                H_src, H_dst = H
                block.srcdata["h"] = H_src
                block.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_new"))
                h_neigh = block.dstdata["h_new"]
                h = self.W(torch.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h


class GCN_VR(nn.Module):
    # self, in_size, hid_size, out_size, num_layer, heads
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.activation = F.relu
        self.layers.append(SAGEConvWithCV(in_feats, n_hidden, self.activation))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConvWithCV(n_hidden, n_hidden, self.activation))
        self.layers.append(SAGEConvWithCV(n_hidden, n_classes, None))

    def forward(self, blocks):
        h = blocks[0].srcdata["feat"]
        updates = []
        for layer, block in zip(self.layers, blocks):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.number_of_dst_nodes()]
            hbar_src = block.srcdata["hist"]
            agg_hbar_dst = block.dstdata["agg_hist"]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst), (hbar_src, agg_hbar_dst))
            block.dstdata["h_new"] = h
        return h

    def inference(self, g, device, batch_size):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        nodes = torch.arange(g.num_nodes())
        ys = []
        x = g.ndata["feat"]
        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            )

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                block = block.int().to(device)
                induced_nodes = block.srcdata[dgl.NID]

                h = x[induced_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))

                y[start:end] = h.cpu()

            ys.append(y)
            x = y
        return y, ys
