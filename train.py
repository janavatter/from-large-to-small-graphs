import wandb
import torch.nn.functional as F
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
    ClusterGCNSampler,
    SAINTSampler,
    ShaDowKHopSampler,
    MultiLayerFullNeighborSampler,
)
from dgl.dataloading import LaborSampler

from sampler.LADIESSampler import LADIESSampler
from sampler.FastGCNSampler import FastGCNSampler
from sampler.VRGCN import VRGCNSampler
from evaluate import (
    evaluate_blocks,
    evaluate_subgraph_shadow,
    evaluate_subgraph,
    evaluate_vrgcn,
    evaluate_full,
)

from utils import *
from time import perf_counter


def train_blocks(model, train_dataloader, opt, sampling, args):
    block_table = wandb.Table(columns=["sampler", "block"])
    total_loss = 0
    log_nodes_edges = True
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        block_table.add_data(sampling, str(blocks))
        numnodes = 0
        numedges = 0
        first = True

        subgs = []

        for block in blocks:
            subgs.append(dgl.block_to_graph(block))
            if first:
                numnodes += block.number_of_src_nodes()
                first = False
            numnodes += block.number_of_dst_nodes()
            numedges += block.number_of_edges()

        if log_nodes_edges:
            wandb.log({"sampled_nodes": numnodes, "sampled_edges": numedges})
            log_nodes_edges = False

        block_table.add_data(
            sampling, str(f"Numnodes: {numnodes}, numedges: {numedges}")
        )

        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = model(blocks, x)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)

        y = y.to("cpu" if args.mode == "cpu" else "cuda:0")

        if args.data in ["german", "bail", "credit", "pokec-z", "pokec-n"]:
            y = torch.unsqueeze(y, 1).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return it, total_loss, block_table


def train_vrgcn(model, train_dataloader, opt, sampling, args):
    block_table = wandb.Table(columns=["sampler", "block"])
    total_loss = 0
    log_nodes_edges = True
    for it, (blocks, hist_blocks) in enumerate(train_dataloader):
        block_table.add_data(sampling, str(blocks))
        numnodes = 0
        numedges = 0
        first = True
        for block in blocks:
            if first == True:
                numnodes += block.number_of_src_nodes()
                first == False
            numnodes += block.number_of_dst_nodes()
            numedges += block.number_of_edges()

        block_table.add_data(
            sampling, str(f"Numnodes: {numnodes}, numedges: {numedges}")
        )

        if log_nodes_edges:
            wandb.log({"sampled_nodes": numnodes, "sampled_edges": numedges})
            log_nodes_edges = False

        g = args.g
        labels = g.ndata["label"]

        blocks, hist_blocks = load_subtensor(
            g, labels, blocks, hist_blocks, args.mode, True
        )

        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        if args.model in ["ggnn", "sage", "appnp", "gin"]:
            y_hat = model(blocks, x)
        else:
            y_hat = model(blocks)

        # update history
        update_history(g, blocks)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)

        y = y.to("cpu" if args.mode == "cpu" else "cuda:0")

        if args.data in ["german", "bail", "credit", "pokec-z", "pokec-n"]:
            y = torch.unsqueeze(y, 1).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    return it, total_loss, block_table


def train_subgraph(model, train_dataloader, opt, sampling, args):
    subgraph_table = wandb.Table(columns=["sampler", "block"])
    total_loss = 0
    log_nodes_edges = True
    for it, subgraph in enumerate(train_dataloader):
        subgraph_table.add_data(sampling, str(subgraph))

        subgraph_table.add_data(sampling, str(subgraph))
        numnodes = subgraph.number_of_nodes()
        numedges = subgraph.number_of_edges()
        if log_nodes_edges:
            wandb.log({"sampled_nodes": numnodes, "sampled_edges": numedges})
            log_nodes_edges = False

        x = subgraph.ndata["feat"]
        y = subgraph.ndata["label"]
        y_hat = model(subgraph, x)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)

        y = y.to("cpu" if args.mode == "cpu" else "cuda:0")

        if args.data in ["german", "bail", "credit", "pokec-z", "pokec-n"]:
            y = torch.unsqueeze(y, 1).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return it, total_loss, subgraph_table


def train_subgraph_shadow(model, train_dataloader, opt, sampling, args):
    subgraph_table = wandb.Table(columns=["sampler", "block"])
    total_loss = 0
    log_nodes_edges = True
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        subgraph_table.add_data(sampling, str(blocks))
        numnodes = blocks.number_of_nodes()
        numedges = blocks.number_of_edges()
        if log_nodes_edges:
            wandb.log({"sampled_nodes": numnodes, "sampled_edges": numedges})
            log_nodes_edges = False

        x = blocks.ndata["feat"]
        y = blocks.ndata["label"]
        if args.data in ["pokec-n", "pokec-z"]:
            y = torch.where(y > 0, 1, 0)
        y_hat = model(blocks, x)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)

        y = y.to("cpu" if args.mode == "cpu" else "cuda:0")

        if args.data in ["german", "bail", "credit", "pokec-z", "pokec-n"]:
            y = torch.unsqueeze(y, 1).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return it, total_loss, subgraph_table


def train_full(model, g, idx_train, opt, sampling, args):
    total_loss = 0
    model.train()
    y_hat = model(g, g.ndata["feat"])
    y = g.ndata["label"]
    if args.data == "yelp":
        y = y.type(torch.float64)
    else:
        y = y.type(torch.LongTensor)

    y = y.to("cpu" if args.mode == "cpu" else "cuda:0")

    if args.data in ["german", "bail", "credit", "pokec-z", "pokec-n"]:
        y_train = torch.unsqueeze(y[idx_train], 1).float()
        loss = F.binary_cross_entropy_with_logits(y_hat[idx_train], y_train)
    else:
        loss = F.cross_entropy(y_hat[idx_train], y[idx_train])
    opt.zero_grad()
    loss.backward()
    opt.step()
    total_loss += loss.item()
    return 0, total_loss, None


def init_history(g, model, dev_id, args):
    with torch.no_grad():
        if args.model in ["ggnn", "sage", "appnp", "gin"]:
            history = model.inference(g, dev_id, args.batchsize)
        else:
            history = model.inference(g, dev_id, args.batchsize)[1]
        for layer in range(args.layer + 1):
            if layer > 0:
                hist_col = "hist_%d" % layer
                if args.layer == 1:
                    g.ndata["hist_%d" % layer] = history
                else:
                    if args.model in ["sage", "appnp", "gin"]:
                        g.ndata["hist_%d" % layer] = history
                    else:
                        g.ndata["hist_%d" % layer] = history[layer - 1]


def update_history(g, blocks):
    with torch.no_grad():
        for i, block in enumerate(blocks):
            ids = block.dstdata[dgl.NID].cpu()
            hist_col = "hist_%d" % (i + 1)

            h_new = block.dstdata["h_new"].cpu()
            g.ndata[hist_col][ids] = h_new


def train(args, device, g, idx_train, idx_val, model, sampling):
    args.g = g

    start_sampling_time = perf_counter()
    fanout = args.fanout
    fastladies = args.fastladies
    if (
        sampling in ["neighbor", "labor", "shadow", "vrgcn"]
        and len(args.fanout) != args.layer
    ):
        if len(args.fanout) == 1:
            fanout = []
            for i in range(args.layer):
                fanout.append(str(args.fanout[0]))
        else:
            print(
                "Fanout {0} and number of layers {1} do not match.".format(
                    args.fanout, args.layer
                )
            )
            exit()

    elif sampling in ["fastgcn", "ladies"] and len(args.fastladies) != args.layer:
        if len(args.fastladies) == 1:
            fastladies = []
            for i in range(args.layer):
                fastladies.append(str(args.fastladies[0]))
        else:
            print(
                f"Fanout {args.fastladies} and number of layers {args.layer} do not match."
            )
            exit()

    if sampling == "neighbor":
        sampler = NeighborSampler(
            fanout,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
    elif sampling == "labor":
        sampler = LaborSampler(
            fanout, prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )
    elif sampling == "cluster":
        sampler = ClusterGCNSampler(
            g,
            args.num_parts,
            prefetch_ndata=["feat", "label"],
            cache_path=args.cache_path,
        )
        idx_train = torch.arange(args.num_parts)
        idx_val = torch.arange(args.num_parts)
    elif sampling == "saint":
        sampler = SAINTSampler("node", args.budget, prefetch_ndata=["feat", "label"])
    elif sampling == "shadow":
        sampler = ShaDowKHopSampler(
            fanout, prefetch_node_feats=["feat"]
        )
    elif sampling == "ladies":
        sampler = LADIESSampler(
            g, fastladies, prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )
    elif sampling == "fastgcn":
        sampler = FastGCNSampler(
            g, fastladies, prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )
    elif sampling == "vrgcn":
        sampler = VRGCNSampler(g, fanout)
        model.eval()
        init_history(g, model, device, args)
    elif sampling == "none":
        sampler = MultiLayerFullNeighborSampler(
            args.layer, prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )
    elif sampling == "full":
        sampler = None
    else:
        sampler = NeighborSampler(
            fanout,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )

    use_uva = args.mode == "mixed"

    if sampling == "full":
        train_dataloader = None
        val_dataloader = idx_val
    else:
        train_dataloader = DataLoader(
            g,
            idx_train,
            sampler,
            device=device,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=use_uva,
        )

        val_dataloader = DataLoader(
            g,
            idx_val,
            sampler,
            device=device,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=use_uva,
        )

    end_sampling_time = perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    if sampling == "full":
        train_func = lambda model, dl, _opt, s, args: train_full(
            model, g, idx_train, opt, s, args
        )
    else:
        train_func = {
            "neighbor": train_blocks,
            "labor": train_blocks,
            "ladies": train_blocks,
            "fastgcn": train_blocks,
            "shadow": train_subgraph_shadow,
            "saint": train_subgraph,
            "cluster": train_subgraph,
            "vrgcn": train_vrgcn,
            "none": train_blocks,
        }[sampling]

    eval_func = {
        "neighbor": evaluate_blocks,
        "labor": evaluate_blocks,
        "ladies": evaluate_blocks,
        "fastgcn": evaluate_blocks,
        "shadow": evaluate_subgraph_shadow,
        "saint": evaluate_subgraph,
        "cluster": evaluate_subgraph,
        "vrgcn": evaluate_vrgcn,
        "none": evaluate_blocks,
        "full": evaluate_full,
    }[sampling]

    start_time = perf_counter()

    current_epoch_time = 0
    for epoch in range(args.epoch):
        start_epoch = perf_counter()
        model.train()
        it, total_loss, table = train_func(model, train_dataloader, opt, sampling, args)

        acc, f1, auroc = eval_func(
            model, g, val_dataloader, args.out_size, args.data, args.model
        )

        wandb.log(
            {"accuracy": acc, "f1": f1, "auroc": auroc, "loss": total_loss / (it + 1)}
        )
        end_epoch = perf_counter()
        print(
            f"Epoch {epoch} | Accuracy {acc} | F1 {f1} | AUROC {auroc} | loss {total_loss/(it+1)} |"
        )

        current_epoch_time += end_epoch - start_epoch

    end_time = perf_counter()


    table_name = (
        args.model
        + "-"
        + args.data
        + "-"
        + args.sampler
        + "-"
        + args.mode
        + "-"
        + str(torch.get_num_threads())
        + "-"
        + str(args.epoch)
        + "-"
        + str(args.layer)
        + "-"
        + str(args.hidden)
        + "-"
        + str(args.num_parts)
        + "-"
        + str(args.batchsize)
        + "-"
        + str(args.fanout)
        + "-"
        + str(args.budget)
    )
    wandb.log(
        {
            "sampling time": end_sampling_time - start_sampling_time,
            "overall training time": end_time - start_time,
            "time per epoch": current_epoch_time / args.epoch,
            "sampling size " + table_name: table,
            "modelpath": args.modelpath,
        }
    )
