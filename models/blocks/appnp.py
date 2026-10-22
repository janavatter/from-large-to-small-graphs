import dgl
import torch
import tqdm
import numpy as np

from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

import torch.nn.functional as F

from dgl import function as fn
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm


# Workaround for DGL block-to-homogeneous conversion; see https://github.com/dmlc/dgl/issues/6296.
def block_to_homogeneous(block, source_ndata, source_ndata_name):
    """Convert a DGLBlock to a homogeneous graph.

    This function performs a two-step process to convert a given block into a
    homogeneous graph. It first transforms the block into a heterogeneous graph,
    treating its source nodes as heterogeneous nodes. Then, this heterogeneous
    graph is further converted to a homogeneous graph. The function outputs
    includes the homogeneous graph, the mapping from node type to node type ID,
    node type count offsets, and the number of destination nodes for each node
    type. Notably, the `source_ndata` of the original block is retained and
    stored in the `source_ndata_name` attribute of the homogeneous graph.

    Parameters
    ----------
    block : dgl.Block
        The block to convert.
    source_ndata : dict[str, Tensor]
        The source node features of the block.
    source_ndata_name : str
        The name of the source node features in the homogeneous graph.

    Returns
    -------
    DGL.Graph
        The converted homogeneous graph.
    dict[str, int]
        The mapping from node type to node type ID.
    list[int]
        The node type count offsets.
    dict[str, int]
        The number of destination nodes for each node type.
    """
    num_dst_nodes_dict = {}
    num_src_nodes_dict = {}
    for ntype in block.dsttypes:
        num_dst_nodes_dict[ntype] = block.number_of_dst_nodes(ntype)
    for ntype in block.srctypes:
        num_src_nodes_dict[ntype] = block.number_of_src_nodes(ntype)

    hetero_edges = {}
    for srctpye, etype, dsttype in block.canonical_etypes:
        src, dst = block.all_edges(etype=etype, order="eid")
        hetero_edges[(srctpye, etype, dsttype)] = (src, dst)
    hetero_g = dgl.heterograph(
        hetero_edges,
        num_nodes_dict=num_src_nodes_dict,
        idtype=block.idtype,
        device=block.device,
    )
    ntype_to_id = {ntype: hetero_g.get_ntype_id(ntype) for ntype in hetero_g.ntypes}
    hetero_g.ndata[source_ndata_name] = source_ndata
    homo_g, ntype_counts, _ = dgl.to_homogeneous(
        hetero_g, ndata=[source_ndata_name], return_count=True
    )
    ntype_offset = np.insert(np.cumsum(ntype_counts), 0, 0)
    return homo_g, ntype_to_id, ntype_offset, num_dst_nodes_dict


"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""

# APPNP implementation adapted from DGL's example: https://github.com/dmlc/dgl/blob/master/examples/pytorch/appnp/appnp.py
# Mini-batch design follows the DGL guide: https://www.dgl.ai/dgl_docs/guide/minibatch-nn.html#guide-minibatch-custom-gnn-module

import torch.nn as nn

from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
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
        super(APPNP, self).__init__()
        self.hid_size = hid_size
        self.out_size = out_size
        self.num_layer = num_layer
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

    def forward(self, blocks, features):
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        for block in blocks:
            homo_g, ntype_to_id, ntype_offset, num_dst_nodes_dict = (
                block_to_homogeneous(block, block.srcdata["feat"], "feat")
            )
            h = self.propagate(homo_g, h)
            h = h[: block.number_of_dst_nodes()]
            block.dstdata["h_new"] = h
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

        h = feat
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            homo_g, ntype_to_id, ntype_offset, num_dst_nodes_dict = (
                block_to_homogeneous(blocks[0], blocks[0].srcdata["feat"], "feat")
            )
            src_ids = blocks[0].srcdata["_ID"]
            dst_ids = blocks[0].dstdata["_ID"]
            feat = h[src_ids]
            feat = self.propagate(homo_g, feat)
            h[dst_ids] = feat[: blocks[0].number_of_dst_nodes()]
        return h


class APPNP_FULL(nn.Module):
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
        super().__init__()
        self.hid_size = hid_size
        self.out_size = out_size
        self.num_layer = num_layer
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


class APPNPConv2(nn.Module):
    r"""Approximate Personalized Propagation of Neural Predictions layer from `Predict then
    Propagate: Graph Neural Networks meet Personalized PageRank
    <https://arxiv.org/pdf/1810.05997.pdf>`__

    # math::
        H^{0} &= X

        H^{l+1} &= (1-\alpha)\left(\tilde{D}^{-1/2}
        \tilde{A} \tilde{D}^{-1/2} H^{l}\right) + \alpha H^{0}

    where :math:`\tilde{A}` is :math:`A` + :math:`I`.

    Parameters
    ----------
    k : int
        The number of iterations :math:`K`.
    alpha : float
        The teleport probability :math:`\alpha`.
    edge_drop : float, optional
        The dropout rate on edges that controls the
        messages received by each node. Default: ``0``.

    tensor([[0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536,
            0.8536],
            [0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268,
            0.9268],
            [0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634,
            0.9634],
            [0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268, 0.9268,
            0.9268],
            [0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634, 0.9634,
            0.9634],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
            0.5000]])
    """

    def __init__(self, k, alpha, edge_drop=0.0):
        super(APPNPConv2, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)`. :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        with graph.local_scope():
            if edge_weight is None:
                src_norm = torch.pow(graph.out_degrees().to(feat).clamp(min=1), -0.5)
                shp = src_norm.shape + (1,) * (feat.dim() - 1)
                src_norm = torch.reshape(src_norm, shp).to(feat.device)
                dst_norm = torch.pow(graph.in_degrees().to(feat).clamp(min=1), -0.5)
                shp = dst_norm.shape + (1,) * (feat.dim() - 1)
                dst_norm = torch.reshape(dst_norm, shp).to(feat.device)
            else:
                edge_weight = EdgeWeightNorm("both")(graph, edge_weight)
            feat_0 = feat
            for _ in range(self._k):
                if edge_weight is None:
                    feat = feat * src_norm
                graph.ndata["h"] = feat
                w = (
                    torch.ones(graph.num_edges(), 1)
                    if edge_weight is None
                    else edge_weight
                )
                graph.edata["w"] = self.edge_drop(w).to(feat.device)
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
                feat = graph.ndata.pop("h")
                if edge_weight is None:
                    feat["_N_dst"] = feat["_N_dst"] * dst_norm
                feat["_N_src"] = (1 - self._alpha) * feat[
                    "_N_src"
                ] + self._alpha * feat_0
            return feat["_N_src"]
