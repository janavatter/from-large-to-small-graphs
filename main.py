import torch
import argparse
from time import perf_counter
import wandb
import dgl
import numpy as np
from dgl.data import (
    AsNodePredDataset,
    CoraGraphDataset,
    RedditDataset,
    FlickrDataset,
    YelpDataset,
    PubmedGraphDataset,
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CiteseerGraphDataset,
    CoauthorCSDataset,
    CoauthorPhysicsDataset,
    ChameleonDataset,
    SquirrelDataset,
    ActorDataset,
)
from ogb.nodeproppred import DglNodePropPredDataset
from models.blocks.gcn import GCN, GCN_FULL
from models.blocks.gat import GAT, GAT_FULL
from models.blocks.ggnn import GGNN, GGNN_FULL
from models.blocks.appnp import APPNP, APPNP_FULL
from models.blocks.sage import SAGE, SAGE_FULL
from models.blocks.gin import GIN, GIN_FULL
from models.subgraph.gcn_subg import GCN_subg
from models.subgraph.gat_subg import GAT_subg
from models.subgraph.ggnn_subg import GGNN_subg
from models.subgraph.appnp_subg import APPNP_subg
from models.subgraph.sage_subg import SAGE_subg
from models.subgraph.gin_subg import GIN_subg
from models.vrgcn.gcn_vr import GCN_VR
from train import train
from layerwise_inference import layerwise_infer
from datetime import datetime
from utils import load_bail, load_credit, load_german, load_pokec, load_heterophilous_dataset, feature_norm
import psutil
import os
import shutil
import json


def get_user_info():
    """Return an optional local W&B entity, or use W&B's default entity."""
    if not os.path.exists("config.json"):
        return None

    with open("config.json", "r") as file:
        config = json.load(file)
    return config.get("entity") or None


def select_first_split(g):
    g.ndata["train_mask"] = g.ndata["train_mask"][:, 0]
    g.ndata["val_mask"] = g.ndata["val_mask"][:, 0]
    g.ndata["test_mask"] = g.ndata["test_mask"][:, 0]
    return g


def train_gnn(args):

    start_time = perf_counter()
    # load and preprocess dataset
    dataset = args.data
    start_dataloading_time = perf_counter()
    if args.data not in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
        if args.data in ["ogbn-products", "ogbn-arxiv", "ogbn-papers100M"]:
            dgldataset = DglNodePropPredDataset(dataset)
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "flickr":
            dgldataset = FlickrDataset()
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "yelp":
            dgldataset = YelpDataset()
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "cora":
            dgldataset = CoraGraphDataset()
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "reddit":
            dgldataset = RedditDataset()
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "pubmed":
            dgldataset = PubmedGraphDataset()
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "amazoncomputer":
            dgldataset = AmazonCoBuyComputerDataset()
            graphdata = AsNodePredDataset(dgldataset, [0.8, 0.1, 0.1])
        elif args.data == "amazonphoto":
            dgldataset = AmazonCoBuyPhotoDataset()
            graphdata = AsNodePredDataset(dgldataset, [0.8, 0.1, 0.1])
        elif args.data == "citeseer":
            dgldataset = CiteseerGraphDataset()
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "coauthorcs":
            dgldataset = CoauthorCSDataset()
            graphdata = AsNodePredDataset(dgldataset, [0.8, 0.1, 0.1])
        elif args.data == "coauthorph":
            dgldataset = CoauthorPhysicsDataset()
            graphdata = AsNodePredDataset(dgldataset, [0.8, 0.1, 0.1])
        elif args.data == "chameleon":
            dgldataset = ChameleonDataset(transform=select_first_split)
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "squirrel":
            dgldataset = SquirrelDataset(transform=select_first_split)
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "actor":
            dgldataset = ActorDataset(transform=select_first_split)
            graphdata = AsNodePredDataset(dgldataset)
        elif args.data == "romanempire":
            graphdata = load_heterophilous_dataset("roman_empire")
        elif args.data == "amazonratings":
            graphdata = load_heterophilous_dataset("amazon_ratings")
        elif args.data == "minesweeper":
            graphdata = load_heterophilous_dataset("minesweeper")
        elif args.data == "tolokers":
            graphdata = load_heterophilous_dataset("tolokers")
        elif args.data == "questions":
            graphdata = load_heterophilous_dataset("questions")
        idx_test = graphdata.test_idx
        idx_train = graphdata.train_idx
        idx_val = graphdata.val_idx

        g = graphdata[0]
        if args.data in ["ogbn-products", "ogbn-arxiv"]:
            g = dgl.add_self_loop(g)
        if args.data == "flickr" and args.model == "gat":
            g = dgl.add_self_loop(g)
        if args.data in ["fraudamazon", "fraudyelp"]:
            g.ndata["feat"] = g.ndata["feature"]

        out_size = graphdata.num_classes
        args.out_size = out_size

    # Credit, German, and Bail/Recidivism dataset distribution: https://github.com/chirag-agarwall/nifty
    # Credit, German, and Bail/Recidivism preprocessing layout: https://github.com/yushundong/EDITS/tree/main/dataset
    else:
        if args.data == "credit":
            sens_attr = "Age"  # column number after feature process is 1
            sens_idx = 1
            predict_attr = "NoDefaultNextMonth"
            label_number = 6000
            path_credit = "./dataset/credit"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(
                args.data,
                sens_attr,
                predict_attr,
                path=path_credit,
                label_number=label_number,
            )
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
            src, dst = np.nonzero(adj)
            g = dgl.graph((src, dst))
            g.ndata["feat"] = features
            g.ndata["label"] = labels
            out_size = labels.unique().shape[0] - 1
        elif args.data == "german":
            sens_attr = "Gender"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "GoodCustomer"
            label_number = 100
            path_german = "./dataset/german"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(
                args.data,
                sens_attr,
                predict_attr,
                path=path_german,
                label_number=label_number,
            )
            src, dst = np.nonzero(adj)
            g = dgl.graph((src, dst))
            g.ndata["feat"] = features
            g.ndata["label"] = labels
            out_size = labels.unique().shape[0] - 1
        elif args.data == "bail":
            sens_attr = "WHITE"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "RECID"
            label_number = 100
            path_bail = "./dataset/bail"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(
                args.data,
                sens_attr,
                predict_attr,
                path=path_bail,
                label_number=label_number,
            )
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
            src, dst = np.nonzero(adj)
            g = dgl.graph((src, dst))
            g.ndata["feat"] = features
            g.ndata["label"] = labels
            out_size = labels.unique().shape[0] - 1
        # Pokec dataset distribution: https://github.com/WuGL-CS/FairGNN
        # Pokec preprocessing reference: https://github.com/ZzoomD/FairGKD/tree/master/datasets
        elif args.data == "pokec-z":
            dataset = "region_job"
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 4000
            sens_number = 200
            sens_idx = 3
            seed = 20
            path = "./dataset/pokec/"
            test_idx = False
            (
                adj,
                features,
                labels,
                idx_train,
                idx_val,
                idx_test,
                sens,
                idx_sens_train,
            ) = load_pokec(
                dataset,
                sens_attr,
                predict_attr,
                path=path,
                label_number=label_number,
                sens_number=sens_number,
                seed=seed,
                test_idx=test_idx,
            )
            labels[labels > 1] = 1
            src, dst = np.nonzero(adj)
            g = dgl.graph((src, dst))
            g.ndata["feat"] = features
            g.ndata["label"] = labels
            out_size = 1
        elif args.data == "pokec-n":
            dataset = "region_job_2"
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 3500
            sens_number = 200
            sens_idx = 3
            seed = 20
            path = "./dataset/pokec/"
            test_idx = False
            (
                adj,
                features,
                labels,
                idx_train,
                idx_val,
                idx_test,
                sens,
                idx_sens_train,
            ) = load_pokec(
                dataset,
                sens_attr,
                predict_attr,
                path=path,
                label_number=label_number,
                sens_number=sens_number,
                seed=seed,
                test_idx=test_idx,
            )
            labels[labels > 1] = 1
            src, dst = np.nonzero(adj)
            g = dgl.graph((src, dst))
            g.ndata["feat"] = features
            g.ndata["label"] = labels
            out_size = 1
        print(f"Use {args.data} graph, {g}")
        if args.sampler == "cluster":
            print("self loop")
            g = dgl.add_self_loop(g)
        args.out_size = out_size
        args.sens = sens

    g = g.to("cuda:0" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda:0")
    args.device = device
    end_dataloading_time = perf_counter()

    in_size = g.ndata["feat"].shape[1]
    hidden_size = args.hidden
    num_layer = args.layer

    gnn = {
        ("gcn", "neighbor"): GCN,
        ("gcn", "labor"): GCN,
        ("gcn", "ladies"): GCN,
        ("gcn", "fastgcn"): GCN,
        ("gcn", "cluster"): GCN_subg,
        ("gcn", "saint"): GCN_subg,
        ("gcn", "shadow"): GCN_subg,
        ("gcn", "vrgcn"): GCN_VR,
        ("gcn", "none"): GCN,
        ("gcn", "full"): GCN_FULL,
        ("gat", "neighbor"): GAT,
        ("gat", "labor"): GAT,
        ("gat", "ladies"): GAT,
        ("gat", "fastgcn"): GAT,
        ("gat", "cluster"): GAT_subg,
        ("gat", "saint"): GAT_subg,
        ("gat", "shadow"): GAT_subg,
        ("gat", "vrgcn"): GAT,
        ("gat", "none"): GAT,
        ("gat", "full"): GAT_FULL,
        ("ggnn", "neighbor"): GGNN,
        ("ggnn", "labor"): GGNN,
        ("ggnn", "ladies"): GGNN,
        ("ggnn", "fastgcn"): GGNN,
        ("ggnn", "cluster"): GGNN_subg,
        ("ggnn", "saint"): GGNN_subg,
        ("ggnn", "shadow"): GGNN_subg,
        ("ggnn", "vrgcn"): GGNN,
        ("ggnn", "none"): GGNN,
        ("ggnn", "full"): GGNN_FULL,
        ("appnp", "neighbor"): APPNP,
        ("appnp", "labor"): APPNP,
        ("appnp", "ladies"): APPNP,
        ("appnp", "fastgcn"): APPNP,
        ("appnp", "cluster"): APPNP_subg,
        ("appnp", "saint"): APPNP_subg,
        ("appnp", "shadow"): APPNP_subg,
        ("appnp", "vrgcn"): APPNP,
        ("appnp", "none"): APPNP,
        ("appnp", "full"): APPNP_FULL,
        ("sage", "neighbor"): SAGE,
        ("sage", "labor"): SAGE,
        ("sage", "ladies"): SAGE,
        ("sage", "fastgcn"): SAGE,
        ("sage", "cluster"): SAGE_subg,
        ("sage", "saint"): SAGE_subg,
        ("sage", "shadow"): SAGE_subg,
        ("sage", "vrgcn"): SAGE,
        ("sage", "none"): SAGE,
        ("sage", "full"): SAGE_FULL,
        ("gin", "neighbor"): GIN,
        ("gin", "labor"): GIN,
        ("gin", "ladies"): GIN,
        ("gin", "fastgcn"): GIN,
        ("gin", "cluster"): GIN_subg,
        ("gin", "saint"): GIN_subg,
        ("gin", "shadow"): GIN_subg,
        ("gin", "vrgcn"): GIN,
        ("gin", "none"): GIN,
        ("gin", "full"): GIN_FULL,
    }[(args.model, args.sampler)]

    model = gnn(in_size, hidden_size, out_size, num_layer, heads=[8, 1]).to(device)

    train(args, device, g, idx_train, idx_val, model, args.sampler)

    start_inference_time = perf_counter()

    acc, f1, auroc, parity, equality = layerwise_infer(
        device,
        g,
        idx_test,
        model,
        args.data,
        batch_size=args.testbatchsize,
        out_size=out_size,
        args=args,
    )

    end_time = perf_counter()

    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_usage_full = psutil.Process().memory_full_info()
    process_cpu_times = psutil.Process().cpu_times()
    system_cpu_times = psutil.cpu_times()
    system_cpu_stats = psutil.cpu_stats()

    wandb.log(
        {
            "test-acc": acc.item(),
            "test-f1": f1.item(),
            "test-auroc": auroc,
            "parity": parity,
            "equality": equality,
        }
    )

    wandb.log(
        {
            "num_nodes": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "end-to-end time": end_time - start_time,
            "dataloading time": end_dataloading_time - start_dataloading_time,
            "inference time": end_time - start_inference_time,
            "used memory": memory_usage,
            "used memory full": memory_usage_full,
            "process cpu times": process_cpu_times,
            "system cpu times": system_cpu_times,
            "system cpu stats": system_cpu_stats,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode (default: cpu). Use 'mixed' for CPU-GPU mixed training or "
             "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--model", default="gcn", choices=["gcn", "gat", "ggnn", "appnp", "sage", "gin"]
    )
    parser.add_argument(
        "--data",
        default="ogbn-arxiv",
        choices=[
            "ogbn-products",
            "ogbn-arxiv",
            "ogbn-papers100M",
            "cora",
            "reddit",
            "flickr",
            "yelp",
            "pubmed",
            "amazoncomputer",
            "amazonphoto",
            "citeseer",
            "coauthorcs",
            "coauthorph",
            "credit",
            "german",
            "bail",
            "pokec-z",
            "pokec-n",
            "chameleon",
            "squirrel",
            "actor",
            "romanempire",
            "amazonratings",
            "minesweeper",
            "tolokers",
            "questions",
        ],
    )
    parser.add_argument(
        "--sampler",
        default="neighbor",
        choices=[
            "neighbor",
            "cluster",
            "saint",
            "labor",
            "shadow",
            "ladies",
            "fastgcn",
            "vrgcn",
            "none",
            "full",
        ],
    )
    parser.add_argument(
        "--fanout",
        type=int,
        nargs="+",
        default=[10],
        help="Neighbors sampled per layer; a single value is reused for every layer (default: 10).",
    )
    parser.add_argument("--fastladies", type=int, nargs="+")
    parser.add_argument("--num_parts", default=100, type=int)
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--hidden", default=256, type=int)
    parser.add_argument("--layer", default=2, type=int)
    parser.add_argument("--wandb", default="sampling-study")
    parser.add_argument("--cache_path", default="cluster_gcn.pkl")
    parser.add_argument("--batchsize", default=1024, type=int)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--testbatchsize", type=int, default=4096)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for PyTorch and NumPy (default: 0).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "ggnn":
        args.layer = 1
        if (
                args.sampler in ["neighbor", "vrgcn", "labor", "shadow"]
                and len(args.fanout) > 1
        ):
            args.fanout = [args.fanout[0]]
        if args.sampler in ["fastgcn", "ladies"] and len(args.fastladies) > 1:
            args.fastladies = [args.fastladies[0]]

    print("the number of cpu threads: {}".format(torch.get_num_threads()))
    num_threads = torch.get_num_threads()

    config = {
        "architecture": args.model,
        "dataset": args.data,
        "sampler": args.sampler,
        "mode": args.mode,
        "epochs": args.epoch,
        "layer": args.layer,
        "hidden": args.hidden,
        "num_parts": args.num_parts,
        "fastladies": args.fastladies,
        "num_threads": num_threads,
        "batch_size": args.batchsize,
        "fanout": args.fanout,
        "budget": args.budget,
        "run": args.run,
        "seed": args.seed,
    }
    run_name = (
            args.model
            + "-"
            + args.data
            + "-"
            + args.sampler
            + "-"
            + args.mode
            + "-"
            + str(num_threads)
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
            + str(args.fastladies)
            + "-"
            + str(args.budget)
    )
    wandb.init(
        project=args.wandb,
        entity=get_user_info(),
        name=run_name,
        config=config,
        mode="offline",
    )
    wandb.log({"seed": args.seed})

    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")

    modelpath = (
            str(now)
            + "-"
            + args.model
            + "-"
            + args.data
            + "-"
            + args.sampler
            + "-"
            + args.mode
            + "-"
            + str(num_threads)
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
            + str(args.fastladies)
            + "-"
            + str(args.budget)
            + ".pkl"
    )
    args.modelpath = modelpath

    train_gnn(args)

    run_dir = os.path.dirname(wandb.run.dir) if wandb.run.dir else None
    wandb.finish()
    if run_dir and os.path.exists(run_dir):
        shutil.rmtree(run_dir)
