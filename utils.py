import dgl
import dgl.function as fn
import torch
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.spatial import distance_matrix


# Credit, German, and Bail/Recidivism dataset distribution: https://github.com/chirag-agarwall/nifty
# Credit, German, and Bail/Recidivism preprocessing layout: https://github.com/yushundong/EDITS/tree/main/dataset
# Pokec dataset distribution: https://github.com/WuGL-CS/FairGNN
# Pokec preprocessing reference: https://github.com/ZzoomD/FairGKD/tree/master/datasets


def load_subtensor(g, labels, blocks, hist_blocks, dev_id, aggregation_on_device=False):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    blocks[0].srcdata["feat"] = g.ndata["feat"][blocks[0].srcdata[dgl.NID]]
    blocks[-1].dstdata["label"] = labels[blocks[-1].dstdata[dgl.NID]]
    ret_blocks = []
    ret_hist_blocks = []
    for i, (block, hist_block) in enumerate(zip(blocks, hist_blocks)):
        hist_col = "feat" if i == 0 else "hist_%d" % i
        block.srcdata["hist"] = g.ndata[hist_col][block.srcdata[dgl.NID]]

        hist_block.srcdata["hist"] = g.ndata[hist_col][hist_block.srcdata[dgl.NID]]
        if aggregation_on_device:
            hist_block = hist_block.to(dev_id)
        hist_block.update_all(fn.copy_u("hist", "m"), fn.mean("m", "agg_hist"))

        block = block.int().to(dev_id)
        if not aggregation_on_device:
            hist_block = hist_block.to(dev_id)
        block.dstdata["agg_hist"] = hist_block.dstdata["agg_hist"]
        ret_blocks.append(block)
        ret_hist_blocks.append(hist_block)
    return ret_blocks, ret_hist_blocks


def load_credit(
    dataset,
    sens_attr="Age",
    predict_attr="NoDefaultNextMonth",
    path="./dataset/credit/",
    label_number=1000,
):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove("Single")

    if os.path.exists(f"{path}/{dataset}_edges.txt"):
        edges_unordered = np.genfromtxt(f"{path}/{dataset}_edges.txt").astype("int")
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f"{path}/{dataset}_edges.txt", edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)) :],
        label_idx_1[int(0.75 * len(label_idx_1)) :],
    )

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_bail(
    dataset,
    sens_attr="WHITE",
    predict_attr="RECID",
    path="../dataset/bail/",
    label_number=1000,
):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f"{path}/{dataset}_edges.txt"):
        edges_unordered = np.genfromtxt(f"{path}/{dataset}_edges.txt").astype("int")
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f"{path}/{dataset}_edges.txt", edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(
        label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)) :],
        label_idx_1[int(0.75 * len(label_idx_1)) :],
    )

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(
    dataset,
    sens_attr="Gender",
    predict_attr="GoodCustomer",
    path="../dataset/german/",
    label_number=1000,
):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove("OtherLoansAtStore")
    header.remove("PurposeOfLoan")

    idx_features_labels["Gender"].loc[idx_features_labels["Gender"] == "Female"] = 1
    idx_features_labels["Gender"].loc[idx_features_labels["Gender"] == "Male"] = 0

    if os.path.exists(f"{path}/{dataset}_edges.txt"):
        edges_unordered = np.genfromtxt(f"{path}/{dataset}_edges.txt").astype("int")
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f"{path}/{dataset}_edges.txt", edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)) :],
        label_idx_1[int(0.75 * len(label_idx_1)) :],
    )

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec(
    dataset,
    sens_attr,
    predict_attr,
    path="./dataset/pokec/",
    label_number=1000,
    sens_number=500,
    seed=19,
    test_idx=False,
):
    """Load data"""

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int
    )

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random

    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)) :]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns
    )
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random

        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)

    return idx_map


def load_heterophilous_dataset(name, raw_dir="./dataset/heterophilous"):
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{name}.npz")

    if not os.path.exists(path):
        url = f"https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}.npz"
        print(f"Downloading {url}...")
        import urllib.request

        urllib.request.urlretrieve(url, path)

    data = np.load(path)
    src = torch.from_numpy(data["edges"][:, 0])
    dst = torch.from_numpy(data["edges"][:, 1])
    features = torch.from_numpy(data["node_features"])
    if name in ["minesweeper", "questions"]:
        print("Normalizing...")
        features = feature_norm(features)
    labels = torch.from_numpy(data["node_labels"])
    train_masks = torch.from_numpy(data["train_masks"].T)
    val_masks = torch.from_numpy(data["val_masks"].T)
    test_masks = torch.from_numpy(data["test_masks"].T)

    num_nodes = len(labels)
    num_classes = len(labels.unique())

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.to_bidirected(g)
    g.ndata["feat"] = features
    g.ndata["label"] = labels
    g.ndata["train_mask"] = train_masks
    g.ndata["val_mask"] = val_masks
    g.ndata["test_mask"] = test_masks

    train_mask = g.ndata["train_mask"][:, 0]
    val_mask = g.ndata["val_mask"][:, 0]
    test_mask = g.ndata["test_mask"][:, 0]

    idx_train = torch.where(train_mask)[0]
    idx_val = torch.where(val_mask)[0]
    idx_test = torch.where(test_mask)[0]

    return _HeterophilousWrapper(g, num_classes, idx_train, idx_val, idx_test)


class _HeterophilousWrapper:
    """Duck-typed compat wrapper mimicking AsNodePredDataset interface."""

    def __init__(self, g, num_classes, idx_train, idx_val, idx_test):
        self._g = g
        self.num_classes = num_classes
        self.train_idx = idx_train
        self.val_idx = idx_val
        self.test_idx = idx_test

    def __getitem__(self, idx):
        return self._g


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1

def fair_metric(pred, labels, sens, *, from_logits=True, threshold=0.5):
    pred = np.asarray(pred)
    labels = np.asarray(labels).reshape(-1)
    sens = np.asarray(sens).reshape(-1)

    # Convert model output to binary predictions y-hat ∈ {0, 1}.
    if pred.ndim == 2 and pred.shape[1] > 1:
        # Two-class (or multi-class) logits/probabilities: choose the predicted class.
        pred_binary = np.argmax(pred, axis=1)
    elif from_logits:
        # One logit per example. sigmoid(logit) >= 0.5 is equivalent to logit >= 0.
        pred_binary = (pred.reshape(-1) >= 0).astype(int)
    else:
        # One probability per example.
        pred_binary = (pred.reshape(-1) >= threshold).astype(int)

    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = idx_s0 & (labels == 1)
    idx_s1_y1 = idx_s1 & (labels == 1)

    if not (idx_s0.any() and idx_s1.any()
            and idx_s0_y1.any() and idx_s1_y1.any()):
        raise ValueError("Each sensitive group needs at least one sample; "
                         "each group also needs at least one true positive.")

    parity = abs(pred_binary[idx_s0].mean() - pred_binary[idx_s1].mean())
    equality = abs(
        pred_binary[idx_s0_y1].mean() - pred_binary[idx_s1_y1].mean()
    )

    return float(parity), float(equality)


def fair_metric_orig(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(
        sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
    )
    return parity.item(), equality.item()
