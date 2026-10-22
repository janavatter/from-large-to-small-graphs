import torch
import torchmetrics.functional as MF
from utils import *


def evaluate_blocks(model, graph, dataloader, out_size, dataset, modelname):
    model.eval()
    ys = []
    yhatsauroc = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            if dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
                ys.append(torch.unsqueeze(blocks[-1].dstdata["label"], 1))
                tmp = torch.sigmoid(model(blocks, x))
                yhatsauroc.append(tmp)
                tmp = torch.where(tmp >= 0.5, 1, 0)
                y_hats.append(tmp)
            else:
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, x))

    if dataset == "yelp":
        acc = MF.classification.multilabel_accuracy(
            torch.cat(y_hats), torch.cat(ys), num_labels=int(out_size)
        )
        f1 = MF.classification.multilabel_f1_score(
            torch.cat(y_hats), torch.cat(ys), num_labels=int(out_size)
        )
        auroc = None
    elif dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
        acc = MF.accuracy(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        f1 = MF.f1_score(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        auroc = MF.auroc(
            torch.cat(yhatsauroc), torch.cat(ys), task="binary", num_classes=int(1)
        )
    else:
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        f1 = MF.f1_score(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        auroc = None

    return acc, f1, auroc


def evaluate_vrgcn(model, graph, dataloader, out_size, dataset, modelname):
    model.eval()
    ys = []
    y_hats = []
    yhatsauroc = []
    for it, (blocks, hist_blocks) in enumerate(dataloader):
        with torch.no_grad():
            labels = graph.ndata["label"]
            blocks, hist_blocks = load_subtensor(
                graph, labels, blocks, hist_blocks, "cpu", True
            )
            x = blocks[0].srcdata["feat"]
            if dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
                ys.append(torch.unsqueeze(blocks[-1].dstdata["label"], 1))
                tmp = torch.sigmoid(model(blocks))
                yhatsauroc.append(tmp)
                tmp = torch.where(tmp >= 0.5, 1, 0)
                y_hats.append(tmp)
            elif modelname in ["ggnn", "sage", "appnp", "gin"]:
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, x))
            else:
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks))

    if dataset == "yelp":
        acc = MF.classification.multilabel_accuracy(
            torch.cat(y_hats), torch.cat(ys), num_labels=int(out_size)
        )
        f1 = MF.classification.multilabel_f1_score(
            torch.cat(y_hats), torch.cat(ys), num_labels=int(out_size)
        )
        auroc = None
    elif dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
        acc = MF.accuracy(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        f1 = MF.f1_score(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        auroc = MF.auroc(
            torch.cat(yhatsauroc), torch.cat(ys), task="binary", num_classes=int(1)
        )
    else:
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        f1 = MF.f1_score(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        auroc = None

    return acc, f1, auroc


def evaluate_subgraph_shadow(model, graph, dataloader, out_size, dataset, modelname):
    model.eval()
    ys = []
    y_hats = []
    yhatsauroc = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks.ndata["feat"]
            if dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
                label = torch.unsqueeze(blocks.ndata["label"], 1)
                if dataset in ["pokec-z", "pokec-n"] and torch.min(label) < 0:
                    label = torch.where(label > 0, 1, 0)
                ys.append(label)
                tmp = torch.sigmoid(model(blocks, x))
                yhatsauroc.append(tmp)
                tmp = torch.where(tmp >= 0.5, 1, 0)
                y_hats.append(tmp)
            else:
                ys.append(blocks.ndata["label"])
                y_hats.append(model(blocks, x))

    if dataset == "yelp":
        acc = MF.classification.multilabel_accuracy(
            torch.cat(y_hats), torch.cat(ys), num_labels=int(out_size)
        )
        f1 = MF.f1_score(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multilabel",
            num_labels=int(out_size),
        )
        auroc = None
    elif dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
        acc = MF.accuracy(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        f1 = MF.f1_score(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        auroc = MF.auroc(
            torch.cat(yhatsauroc), torch.cat(ys), task="binary", num_classes=int(1)
        )
    else:
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        f1 = MF.f1_score(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        auroc = None

    return acc, f1, auroc


def evaluate_subgraph(model, graph, dataloader, out_size, dataset, modelname):
    model.eval()
    ys = []
    y_hats = []
    yhatsauroc = []
    for it, subgraph in enumerate(dataloader):
        with torch.no_grad():
            x = subgraph.ndata["feat"]

            if dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
                label = torch.unsqueeze(subgraph.ndata["label"], 1)
                if dataset in ["pokec-z", "pokec-n"] and torch.min(label) < 0:
                    label = torch.where(label > 0, 1, 0)
                ys.append(label)
                tmp = torch.sigmoid(model(subgraph, x))
                yhatsauroc.append(tmp)
                tmp = torch.where(tmp >= 0.5, 1, 0)
                y_hats.append(tmp)
            else:
                ys.append(subgraph.ndata["label"])
                y_hats.append(model(subgraph, x))

    if dataset == "yelp":
        acc = MF.classification.multilabel_accuracy(
            torch.cat(y_hats), torch.cat(ys), num_labels=int(out_size)
        )
        f1 = MF.f1_score(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multilabel",
            num_labels=int(out_size),
        )
        auroc = None
    elif dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
        acc = MF.accuracy(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        f1 = MF.f1_score(
            torch.cat(y_hats), torch.cat(ys), task="binary", num_classes=int(1)
        )
        auroc = MF.auroc(
            torch.cat(yhatsauroc), torch.cat(ys), task="binary", num_classes=int(1)
        )
    else:
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        f1 = MF.f1_score(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=int(out_size),
        )
        auroc = None

    return acc, f1, auroc


def evaluate_full(model, graph, mask, out_size, dataset, modelname):
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata["feat"])
        logits = logits[mask]
        labels = graph.ndata["label"][mask]
        if dataset == "yelp":
            acc = MF.classification.multilabel_accuracy(
                logits, labels, num_labels=int(out_size)
            )
            f1 = MF.f1_score(
                logits, labels, task="multilabel", num_labels=int(out_size)
            )
            auroc = None
        elif dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
            logits = logits.squeeze(-1)
            acc = MF.accuracy(logits, labels, task="binary", num_classes=int(1))
            f1 = MF.f1_score(logits, labels, task="binary", num_classes=int(1))
            auroc = MF.auroc(logits, labels, task="binary", num_classes=int(1))
        else:
            acc = MF.accuracy(
                logits, labels, task="multiclass", num_classes=int(out_size)
            )
            f1 = MF.f1_score(
                logits, labels, task="multiclass", num_classes=int(out_size)
            )
            auroc = None

        return acc, f1, auroc
