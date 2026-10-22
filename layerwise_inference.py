import torch
import torchmetrics.functional as MF
from utils import fair_metric


def layerwise_infer(device, graph, nid, model, dataset, batch_size, out_size, args):
    model.eval()
    parity = None
    equality = None
    auroc = None
    with torch.no_grad():
        if args.sampler == "vrgcn" and args.model not in ["ggnn", "gin"]:
            pred, tmp = model.inference(
                graph, device, batch_size
            )
        elif args.sampler == "full":
            pred = model(graph, graph.ndata["feat"])
        elif args.model in ["ggnn", "appnp", "gin"] and args.sampler in ["cluster", "shadow", "saint"]:
            pred = model(graph, graph.ndata["feat"])
        else:
            pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)

        if dataset == "yelp":
            acc = MF.classification.multilabel_accuracy(
                pred, label, num_labels=int(out_size)
            )
            f1 = MF.classification.multilabel_f1_score(
                pred, label, num_labels=int(out_size)
            )
        elif dataset in ["german", "credit", "bail", "pokec-z", "pokec-n"]:
            label = torch.unsqueeze(label, 1)
            acc = MF.accuracy(pred, label, task="binary")
            f1 = MF.f1_score(pred, label, task="binary")
            auroc = MF.auroc(pred, label, task="binary")
            parity, equality = fair_metric(
                pred, label, torch.unsqueeze(args.sens[nid], 1)
            )
            return acc, f1, auroc, parity, equality
        else:
            acc = MF.accuracy(pred, label, task="multiclass", num_classes=int(out_size))
            f1 = MF.f1_score(pred, label, task="multiclass", num_classes=int(out_size))

    return acc, f1, auroc, parity, equality
