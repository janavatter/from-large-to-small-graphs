import dgl
import torch
import tqdm

import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import init
from dgl import function as fn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

# Mini-batch GGNN implementation follows the DGL guide: https://www.dgl.ai/dgl_docs/guide/minibatch-nn.html#guide-minibatch-custom-gnn-module


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


class GGNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, heads, num_steps=5):
        super(GGNN, self).__init__()

        self.layers = nn.ModuleList()
        self.lin = nn.ModuleList()
        self.out_size = out_size

        if in_size <= out_size:
            self.layers.append(
                GatedGraphConv2(in_size, out_size, num_steps, n_etypes=1)
            )
        elif in_size > out_size:
            print(f"in dim {in_size} > out dim {out_size}")
            self.layers = nn.ModuleList()
            self.layers.append(GatedGraphConv2(in_size, in_size, num_steps, n_etypes=1))
            self.lin.append(nn.Linear(in_size, out_size, bias=False))
        else:
            print("Error in GGNN layer dimensions")

    def forward(self, blocks, h):
        for i, layer in enumerate(self.layers):
            block = blocks[i]
            h = F.relu(layer(block, h))

        if len(self.lin) > 0:
            for i, layer in enumerate(self.lin):
                h = layer(h)
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


class GGNN_FULL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, heads, num_steps=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.lin = nn.ModuleList()
        self.out_size = out_size
        if in_size <= out_size:
            self.layers.append(
                GatedGraphConv2(in_size, out_size, num_steps, n_etypes=1)
            )
        else:
            self.layers.append(GatedGraphConv2(in_size, in_size, num_steps, n_etypes=1))
            self.lin.append(nn.Linear(in_size, out_size, bias=False))

    def forward(self, g, h):
        dst_nodes = torch.arange(g.num_nodes(), device=g.device)
        for i, layer in enumerate(self.layers):
            block = dgl.to_block(g, dst_nodes)
            h = F.relu(layer(block, h))
        if len(self.lin) > 0:
            for layer in self.lin:
                h = layer(h)
        return h


"""Torch Module for Gated Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop


class GatedGraphConv2(nn.Module):
    r"""Gated Graph Convolution layer from `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__

    # math::
        h_{i}^{0} &= [ x_i \| \mathbf{0} ]

        a_{i}^{t} &= \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} &= \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`x_i`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(t+1)}`.
    n_steps : int
        Number of recurrent steps; i.e, the :math:`t` in the above formula.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self, in_feats, out_feats, n_steps, n_etypes, bias=True):
        super(GatedGraphConv2, self).__init__()
        assert in_feats <= out_feats, "out_feats must be not less than in_feats"
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain("relu")
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, block, feat, etypes=None):
        """

        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        block : DGL Block
            The block.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor, or None
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph. When there's only one edge type,
            this argument can be skipped

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with block.local_scope():
            assert block.is_block, "not a block; convert it with to_block "
            if self._n_etypes != 1:
                assert (
                    etypes.min() >= 0 and etypes.max() < self._n_etypes
                ), "edge type indices out of range [0, {})".format(self._n_etypes)

            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            feat = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                if self._n_etypes == 1 and etypes is None:

                    # Fast path when graph has only one edge type

                    h_dst = feat[: block.number_of_dst_nodes()]

                    block.srcdata["h"] = self.linears[0](feat)
                    block.dstdata["h"] = h_dst

                    aggregate_fn = fn.copy_u("h", "m")

                    block.update_all(aggregate_fn, fn.sum(msg="m", out="a"))
                    a = block.dstdata["a"]
                    block.dstdata.pop("a", None)
                else:
                    block.ndata["h"] = feat
                    for i in range(self._n_etypes):
                        eids = (
                            torch.nonzero(etypes == i, as_tuple=False)
                            .view(-1)
                            .type(block.idtype)
                        )
                        if len(eids) > 0:
                            block.apply_edges(
                                lambda edges: {
                                    "W_e*h": self.linears[i](edges.src["h"])
                                },
                                eids,
                            )
                    block.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
                    a = block.ndata.pop("a")  # (N, D)
                h_dst = self.gru(a, block.dstdata["h"])
                feat = block.srcdata["h"]
                feat[: block.number_of_dst_nodes()] = h_dst

            return feat[: block.number_of_dst_nodes()]
