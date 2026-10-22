import dgl
import torch
from dgl.dataloading.base import *


# Adapted from DGL's VR-GCN example: https://github.com/dmlc/dgl/blob/master/examples/pytorch/vrgcn/train_cv.py


class VRGCNSampler(Sampler):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seed_nodes = torch.LongTensor(seed_nodes)
        blocks = []
        hist_blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seed_nodes, fanout)
            hist_frontier = dgl.in_subgraph(self.g, seed_nodes)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seed_nodes)
            hist_block = dgl.to_block(hist_frontier, seed_nodes)
            # Obtain the seed nodes for next layer.
            seed_nodes = block.srcdata[dgl.NID]

            blocks.insert(0, block)
            hist_blocks.insert(0, hist_block)
        # seed nodes, output nodes, blocks
        return blocks, hist_blocks

    def sample(
        self, g, seed_nodes, exclude_eids=None
    ):  # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        blocks, histblocks = self.sample_blocks(
            g, seed_nodes, exclude_eids=exclude_eids
        )
        return blocks, histblocks
