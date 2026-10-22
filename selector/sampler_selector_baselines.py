from lightgbm import LGBMClassifier
from torchaudio.models.squim import objective

import wandb
import dgl
import torch
import time
import os

import networkx as nx
import scipy.sparse as sp
import scipy

import pandas as pd
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
)
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    RidgeClassifier,
    SGDClassifier,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


from catboost import CatBoostRegressor, CatBoostClassifier


from utils import load_bail, load_credit, load_german, load_pokec, feature_norm

import hydra

from omegaconf import DictConfig

from sklearn.utils.extmath import softmax

GRAPH_CHARACTERISTICS = {
    "ogbn-arxiv": {
        "num_nodes": 169343,
        "num_edges": 1335586,
        "feat_dim": 128,
        "num_classes": 40,
        "avg_degree": 7.886868476867676,
        "density": 4.6573612352572874e-05,
        "num_components": 1,
        "largest_component_size": 169343,
        "avg_clustering_coeff": 0.22612914062470202,
        "transitivity": 0.01620149151351999,
        "assortativity": -0.03946015874996083,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
    "ogbn-products": {
        "num_nodes": 2449029,
        "num_edges": 126167309,
        "feat_dim": 100,
        "num_classes": 47,
        "avg_degree": 51.51728057861328,
        "density": 2.1035806238464124e-05,
        "avg_clustering_coeff": 0.4107879837939035,
        "transitivity": 0.1300615866399033,
        "assortativity": -0.03898114257343303,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
    "cora": {
        "num_nodes": 2708,
        "num_edges": 10556,
        "feat_dim": 1433,
        "num_classes": 7,
        "avg_degree": 3.8980798721313477,
        "density": 0.0014399999126942077,
        "num_components": 78,
        "largest_component_size": 2485,
        "avg_clustering_coeff": 0.24067329850193728,
        "transitivity": 0.09349725626661058,
        "assortativity": -0.06587087427227857,
        "diameter": 19,
        "avg_shortest_path_len": 6.310998681298742,
    },
    "flickr": {
        "num_nodes": 89250,
        "num_edges": 899756,
        "feat_dim": 500,
        "num_classes": 7,
        "avg_degree": 10.081299781799316,
        "density": 0.00011295700478311192,
        "num_components": 1,
        "largest_component_size": 89250,
        "avg_clustering_coeff": 0.03295705875811584,
        "transitivity": 0.003909422012143272,
        "assortativity": -0.04410306340496158,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
    "pubmed": {
        "num_nodes": 19717,
        "num_edges": 88651,
        "feat_dim": 500,
        "num_classes": 3,
        "avg_degree": 4.496170997619629,
        "density": 0.00022804680549104377,
        "num_components": 1,
        "largest_component_size": 19717,
        "avg_clustering_coeff": 0.060175209437523615,
        "transitivity": 0.0537076280274887,
        "assortativity": -0.043630637555888356,
        "diameter": 18,
        "avg_shortest_path_len": 6.336870027999047,
    },
    "amazoncomputer": {
        "num_nodes": 13752,
        "num_edges": 491722,
        "feat_dim": 767,
        "num_classes": 10,
        "avg_degree": 35.75639724731445,
        "density": 0.0026002762758509414,
        "num_components": 314,
        "largest_component_size": 13381,
        "avg_clustering_coeff": 0.3441263874533229,
        "transitivity": 0.10773900322765097,
        "assortativity": -0.056487705164197634,
        "diameter": 10,
        "avg_shortest_path_len": 3.3845095934500526,
    },
    "amazonphoto": {
        "num_nodes": 7650,
        "num_edges": 238163,
        "feat_dim": 745,
        "num_classes": 8,
        "avg_degree": 31.132417678833008,
        "density": 0.004070129206517662,
        "num_components": 136,
        "largest_component_size": 7487,
        "avg_clustering_coeff": 0.40397927439375425,
        "transitivity": 0.17733148009494923,
        "assortativity": -0.04494238741477897,
        "diameter": 11,
        "avg_shortest_path_len": 4.050996792338352,
    },
    "citeseer": {
        "num_nodes": 3327,
        "num_edges": 9228,
        "feat_dim": 3703,
        "num_classes": 6,
        "avg_degree": 2.773669958114624,
        "density": 0.0008339356503152743,
        "num_components": 438,
        "largest_component_size": 2120,
        "avg_clustering_coeff": 0.14147102442629086,
        "transitivity": 0.13006166877182554,
        "assortativity": 0.04965646607078052,
        "diameter": 28,
        "avg_shortest_path_len": 9.329714532486845,
    },
    "coauthorcs": {
        "num_nodes": 18333,
        "num_edges": 163788,
        "feat_dim": 6805,
        "num_classes": 15,
        "avg_degree": 8.934053421020508,
        "density": 0.00048734744416454727,
        "num_components": 1,
        "largest_component_size": 18333,
        "avg_clustering_coeff": 0.34252338196985527,
        "transitivity": 0.18255559535165766,
        "assortativity": 0.11260854329931977,
        "diameter": 24,
        "avg_shortest_path_len": 5.427693716383878,
    },
    "coauthorph": {
        "num_nodes": 34493,
        "num_edges": 495924,
        "feat_dim": 8415,
        "num_classes": 5,
        "avg_degree": 14.37752628326416,
        "density": 0.0004168365381774087,
        "num_components": 1,
        "largest_component_size": 34493,
        "avg_clustering_coeff": 0.37762293908452615,
        "transitivity": 0.18742917153412708,
        "assortativity": 0.2010313810362111,
        "diameter": 17,
        "avg_shortest_path_len": 5.163814898581291,
    },
    "yelp": {
        "num_nodes": 716847,
        "num_edges": 13954819,
        "feat_dim": 300,
        "num_classes": 100,
        "avg_degree": 19.466941833496094,
        "density": 2.7156379528149294e-05,
        "num_components": 6288,
        "largest_component_size": 703655,
        "avg_clustering_coeff": 0.0914540538025344,
        "transitivity": 0.05943817413280867,
        "assortativity": 0.10312919177566342,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
    "reddit": {
        "num_nodes": 232965,
        "num_edges": 114615892,
        "feat_dim": 602,
        "num_classes": 41,
        "avg_degree": 491.98760986328125,
        "density": 0.0021118610741919248,
        "avg_clustering_coeff": 0.5793221228011032,
        "transitivity": 0.24432952325159296,
        "assortativity": 0.10647139431064821,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
    "credit": {
        "num_nodes": 30000,
        "num_edges": 304754,
        "feat_dim": 13,
        "num_classes": 1,
        "avg_degree": 10.158466339111328,
        "density": 0.0003386268431169928,
        "num_components": 120,
        "largest_component_size": 29460,
        "avg_clustering_coeff": 0.5793221228011032,
        "transitivity": 0.7979412022560566,
        "assortativity": 0.65170237999375,
    },
    "german": {
        "num_nodes": 1000,
        "num_edges": 44484,
        "feat_dim": 27,
        "num_classes": 1,
        "avg_degree": 44.48400115966797,
        "density": 0.04452852852852853,
        "num_components": 1,
        "largest_component_size": 1000,
        "avg_clustering_coeff": 0.5793221228011032,
        "transitivity": 0.274344063336954,
        "assortativity": -0.09043898661353802,
        "diameter": 4,
        "avg_shortest_path_len": 2.459785785785786,
    },
    "bail": {
        "num_nodes": 18876,
        "num_edges": 642616,
        "feat_dim": 18,
        "num_classes": 1,
        "avg_degree": 34.0440788269043,
        "density": 0.0018036597157608596,
        "num_components": 1,
        "largest_component_size": 18876,
        "avg_clustering_coeff": 0.5793221228011032,
        "transitivity": 0.19786272602998836,
        "assortativity": -0.014930778151438754,
        "diameter": 24,
        "avg_shortest_path_len": 6.844287511805875,
    },
    "pokec-z": {
        "num_nodes": 67796,
        "num_edges": 1303712,
        "feat_dim": 277,
        "num_classes": 1,
        "avg_degree": 19.22992515563965,
        "density": 0.0002836481314156741,
        "num_components": 170,
        "largest_component_size": 67435,
        "avg_clustering_coeff": 0.5793221228011032,
        "transitivity": 0.07458937793863618,
        "assortativity": 0.2108367398059526,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
    "pokec-n": {
        "num_nodes": 66569,
        "num_edges": 1100663,
        "feat_dim": 266,
        "num_classes": 1,
        "avg_degree": 16.534168243408203,
        "density": 0.00024838011590356873,
        "num_components": 226,
        "largest_component_size": 66082,
        "avg_clustering_coeff": 0.5793221228011032,
        "transitivity": 0.07120961731924773,
        "assortativity": 0.19775507809045031,
        "diameter": None,
        "avg_shortest_path_len": None,
    },
}

TABLE_RULES = (
    {
        "degree": (50, 500),
        "clustering": (0.41, 0.60),
        "time_per_epoch": ("neighbor", "labor", "cluster"),
        "test_acc": ("saint",),
        "used_memory": ("cluster", "saint", "shadow"),
    },
    {
        "degree": (34, 36),
        "clustering": (0.30, 0.41),
        "time_per_epoch": ("cluster",),
        "test_acc": ("vrgcn",),
        "used_memory": ("neighbor", "ladies", "labor"),
    },
    {
        "degree": (8, 15),
        "clustering": (0.30, 0.44),
        "time_per_epoch": ("cluster",),
        "test_acc": ("vrgcn", "saint", "shadow"),
        "used_memory": ("cluster", "saint", "shadow"),
    },
    {
        "degree": (2.8, 5),
        "clustering": (0.06, 0.25),
        "time_per_epoch": ("saint",),
        "test_acc": ("cluster",),
        "used_memory": ("neighbor", "fastgcn", "saint", "shadow"),
    },
)


fanout_dict = {
    "fastgcn": "fastladies",
    "ladies": "fastladies",
    "saint": "budget",
    "cluster": "num_parts",
}

# Define mapping dictionaries
arch_map = {"gcn": 0, "gat": 1, "sage": 2, "ggnn": 3, "appnp": 4}

sampler_map = {
    "neighbor": 0,
    "labor": 1,
    "shadow": 2,
    "saint": 3,
    "cluster": 4,
    "ladies": 5,
    "fastgcn": 6,
    "vrgcn": 7,
}

FEATURE_COLS = [
    "architecture",
    "fanout",
    "layer",
    "epochs",
    "hidden",
    "sampler",
    "batch_size",  # config parameters
    "num_nodes",
    "num_edges",
    "feat_dim",
    "num_classes",
    "avg_degree",
    "density",
    "avg_clustering_coeff",
    "transitivity",
    "assortativity",
]

# all config-defining columns except "sampler" and "test_accuracy"/"time_per_epoch"
group_cols = [
    "architecture",
    "fanout",
    "layer",
    "epochs",
    "hidden",
    "batch_size",
    "num_nodes",
    "num_edges",
    "feat_dim",
    "num_classes",
    "avg_degree",
    "density",
    "avg_clustering_coeff",
    "transitivity",
    "assortativity",
]


def load_results(file="full_results_bias.pkl"):
    results = pd.read_pickle(file)
    return results


def assign_new_fanouts(df, dataset, sampler, architecture=None):

    mask = (df["dataset"] == dataset) & (df["sampler"] == sampler) & (df["layer"] == 2)
    if architecture in ["gcn", "gat"]:
        mask = (
            (df["dataset"] == dataset)
            & (df["sampler"] == sampler)
            & (df["layer"] == 2)
            & (df["architecture"] == architecture)
        )

    col = fanout_dict[sampler]
    df.loc[mask & df[col].isna(), col] = df.loc[mask & df[col].isna(), "fanout"]

    # Step 1: get unique fastladies values in the masked rows
    unique_val = df.loc[mask, fanout_dict[sampler]].unique()

    # Step 2: sort them ascending
    unique_val.sort()

    # If sampler is "cluster", sort descending
    if sampler == "cluster":
        unique_val = unique_val[::-1]  # reverse the sorted array

    # Step 3: create corresponding fanout values from 10 to 50
    # Use linspace to generate evenly spaced values
    fanout_values = np.linspace(10, 50, num=len(unique_val))

    # Step 4: map fanout values back to the dataframe
    for fl_value, fo_value in zip(unique_val, fanout_values):
        df.loc[mask & (df[fanout_dict[sampler]] == fl_value), "fanout"] = fo_value

    return df


def prepare_data(df, objective=None):
    df = df.copy()
    # Fanout is reassigned to evenly spaced fractional values below.
    df["fanout"] = df["fanout"].astype(float)

    datasets = [
        "ogbn-arxiv",
        "ogbn-products",
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
    ]

    # Set fanout to 10 where layer > 2
    df.loc[df["layer"] != 2, "fanout"] = 10

    # Remove rows where dataset is citeseer and num_parts == 37000
    df = df[~((df["dataset"] == "citeseer") & (df["num_parts"] == 37000))].copy()

    mask_saint = df["sampler"] != "saint"
    df.loc[mask_saint, "budget"] = np.nan

    mask_cluster = df["sampler"] != "cluster"
    df.loc[mask_cluster, "num_parts"] = np.nan

    mask_fastladies = ~df["sampler"].isin(["fastgcn", "ladies"])
    df.loc[mask_fastladies, "fastladies"] = np.nan

    for sampler in ["fastgcn", "ladies", "saint", "cluster"]:
        for dataset in datasets:
            if dataset == "yelp" and sampler in ["fastgcn", "ladies"]:
                for architecture in ["gcn", "gat"]:
                    df = assign_new_fanouts(df, dataset, sampler, architecture)
            else:
                df = assign_new_fanouts(df, dataset, sampler)

    drop_cols = [
        "budget",
        "fastladies",
        "num_parts",
        "overall_training_time",
        "end_to_end_time",
        "inference_time",
        "loss",
        "dataloading_time",
        "mode",
        "num_threads",
        "sampling_time",
        "accuracy",
        "f1",
        "test_f1",
    ]

    df = df.drop(columns=drop_cols)

    return df


def build_data_with_graph_characteristics(
    df, graph_characteristics, objective, merge_runs
):

    objectives = ["test_acc", "time_per_epoch", "used_memory"]
    # Drop all other objective columns except the selected one
    columns_to_drop = [col for col in objectives if col != objective]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    rows = []
    for _, row in df.iterrows():
        dataset = row["dataset"]
        if dataset in ["ogbn-papers100M"]:
            continue
        gchar = graph_characteristics[dataset]
        merged = {**row.to_dict(), **gchar}
        rows.append(merged)

    data = pd.DataFrame(rows)

    if merge_runs:
        exclude_cols = ["run", objective]
    else:
        exclude_cols = [objective]

    # List of columns to group by (everything except 'run' and the target column)
    group_cols = [c for c in data.columns if c not in exclude_cols]

    data[group_cols] = data[group_cols].fillna(-1)

    processed = data.groupby(group_cols, as_index=False)[objective].mean()

    return processed


"""
Enhanced Sampler Selection Pipeline - Focused on Accuracy Improvement
====================================================================
"""


def advanced_feature_engineering(data, use_regression_approach):
    """Create more sophisticated features for better prediction."""
    print("Advanced feature engineering...")

    data = data.copy()

    # Architecture encoding (one-hot might be better than label encoding)
    if "architecture" in data.columns:
        arch_dummies = pd.get_dummies(data["architecture"], prefix="arch")
        data = pd.concat([data, arch_dummies], axis=1)

    # Sampler encoding
    if "sampler" in data.columns and not use_regression_approach:
        data = data.drop("sampler", axis=1)

    # Graph complexity tiers
    if "num_nodes" in data.columns:
        data["graph_size_tier"] = pd.cut(
            data["num_nodes"],
            bins=[0, 5000, 50000, 200000, float("inf")],
            labels=["small", "medium", "large", "huge"],
        )
        graph_size_dummies = pd.get_dummies(data["graph_size_tier"], prefix="size")
        data = pd.concat([data, graph_size_dummies], axis=1)

    # Model complexity features
    if "hidden" in data.columns and "layer" in data.columns:
        data["model_complexity"] = data["hidden"] * data["layer"]
        data["hidden_per_layer"] = data["hidden"] / data["layer"]

    # Graph structural features
    if all(
        col in data.columns for col in ["avg_degree", "density", "avg_clustering_coeff"]
    ):
        data["structure_score"] = (
            data["avg_degree"] * data["density"] * data["avg_clustering_coeff"]
        )

    # Interaction features between model and graph
    if all(col in data.columns for col in ["hidden", "num_nodes"]):
        data["hidden_to_nodes_ratio"] = data["hidden"] / np.log(data["num_nodes"] + 1)

    if all(col in data.columns for col in ["layer", "avg_degree"]):
        data["layers_degree_interaction"] = data["layer"] * np.log(
            data["avg_degree"] + 1
        )

    print(f"Feature engineering complete. New shape: {data.shape}")
    return data


def evaluate_with_detailed_metrics(y_true, y_pred, class_names=None, verbose=False):
    """Return accuracy, optionally printing detailed classification diagnostics."""
    y_true = np.array(y_true).ravel()  # Flatten to 1D
    y_pred = np.array(y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    if verbose:
        labels = sorted(set(y_true) | set(y_pred))
        class_names = [str(c) for c in class_names]
        class_report = classification_report(
            y_true,
            y_pred,
            labels=list(range(8)),
            target_names=class_names,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        label_to_class = {int(float(name)): name for name in class_names}
        per_class_acc = np.divide(
            cm.diagonal(),
            cm.sum(axis=1),
            out=np.zeros(len(cm), dtype=float),
            where=cm.sum(axis=1) != 0,
        )
        print(f"True labels: {y_true}, predicted labels: {y_pred}")
        print(f"Overall Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        print(
            pd.DataFrame(
                cm,
                index=[label_to_class.get(int(i), f"Unknown_{i}") for i in labels],
                columns=[label_to_class.get(int(i), f"Unknown_{i}") for i in labels],
            )
        )
        print("\nPer-class Accuracy:")
        for label, acc in zip(labels, per_class_acc):
            print(f"{label_to_class.get(int(label), f'Unknown_{label}')}: {acc:.3f}")

    return accuracy


def contains_nan(x):
    """Return True if x contains any NaN values."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.isnull().values.any()
    else:
        x = np.asarray(x)
        return np.isnan(x).any()


def evaluate_relative_score(y_test, y_pred, ranked_labels, ranked_values, objective):

    K = 7

    # --- NaN CHECKING ---
    if any(contains_nan(arr) for arr in (y_test, y_pred, ranked_labels, ranked_values)):
        print("NaN detected — returning None for all outputs.")
        return None, None, None, None, None

    common_idx = y_test.index & ranked_labels.index & ranked_values.index

    pred_values = []
    best_values = []
    pred_ranks = []

    for idx in common_idx:
        pred = y_pred[idx]
        ranked_l = ranked_labels.loc[idx]
        ranked_v = ranked_values.loc[idx]
        pred_rank = ranked_l.index(pred)

        pred_values.append(ranked_v[pred_rank])
        best_values.append(ranked_v[0])
        pred_ranks.append(pred_rank + 1)

    pred_values = np.array(pred_values)
    best_values = np.array(best_values)
    pred_ranks = np.array(pred_ranks)

    if objective == "test_acc":
        relative_utility_ratio = (
            pred_values / best_values
        ) * 100  # predicted value, best value
        relative_utility_loss_score = (1 - (pred_values / best_values)) * 100
        realtive_penalized_loss_score = ((1 - (pred_values / best_values)) ** 2) * 100
        relative_log_loss_utility_score = -np.log(pred_values / best_values) * 100
    elif objective in ("time_per_epoch", "used_memory"):
        relative_utility_ratio = (
            best_values / pred_values
        ) * 100  # predicted value, best value
        relative_utility_loss_score = (1 - (best_values / pred_values)) * 100
        realtive_penalized_loss_score = ((1 - (best_values / pred_values)) ** 2) * 100
        relative_log_loss_utility_score = -np.log(best_values / pred_values) * 100

    rank_based_score = 1 - (
        (pred_ranks - 1) / (K - 1)
    )  # predicted rank, K number classes

    return (
        np.mean(relative_utility_ratio),
        np.mean(relative_utility_loss_score),
        np.mean(realtive_penalized_loss_score),
        np.mean(relative_log_loss_utility_score),
        np.mean(rank_based_score),
    )


def prepare_classification_labels(processed_data, group_cols, objective):
    # Store original index and order
    original_index = processed_data.index
    processed_data = processed_data.reset_index(drop=True)

    processed_data = processed_data.drop_duplicates()

    # Determine ascending or descending ranking
    ascending = objective != "test_acc"  # test_acc → maximize, others → minimize

    # Rank within each group
    processed_data["rank"] = processed_data.groupby(group_cols)[objective].rank(
        method="first", ascending=ascending
    )

    # Sort to extract ranking lists
    ranked = (
        processed_data.sort_values(group_cols + ["rank"])
        .groupby(group_cols)
        .agg(
            ranked_labels=("sampler", lambda x: list(x)),
            ranked_values=(objective, lambda x: list(x)),
        )
        .reset_index()
    )

    # Compute best (rank 1)
    best_idx = (
        processed_data.groupby(group_cols)[objective].idxmin()
        if ascending
        else processed_data.groupby(group_cols)[objective].idxmax()
    )

    best_samplers = processed_data.loc[
        best_idx, group_cols + ["sampler", objective]
    ].rename(columns={"sampler": "label", objective: "best_value"})

    # Merge best label + rankings back into dataset
    labeled_data = processed_data.merge(best_samplers, on=group_cols, how="left").merge(
        ranked, on=group_cols, how="left"
    )

    # Restore original order using the stored index
    labeled_data = labeled_data.set_index(original_index)

    target_col = "label"

    # Print stats
    id_to_name = {v: k for k, v in sampler_map.items()}
    print(f"Value counts for best samplers according to {objective}:")
    for label, count in best_samplers["label"].value_counts().items():
        name = id_to_name.get(label, label)
        print(f" {label} - {name}: {count}")

    return labeled_data, target_col


def prepare_regression_labels(processed_data, group_cols, objective):
    # Determine ascending or descending ranking
    ascending = "test_acc" not in objective

    # Create a copy to avoid modifying the original during operations
    temp_data = processed_data.copy()

    # Rank within each group
    temp_data["rank"] = temp_data.groupby(group_cols)[objective].rank(
        method="first", ascending=ascending
    )

    # Sort to extract ranking lists (this sorting is only for the aggregation)
    ranked = (
        temp_data.sort_values(group_cols + ["rank"])
        .groupby(group_cols)
        .agg(
            ranked_labels=("sampler", lambda x: list(x)),
            ranked_values=(objective, lambda x: list(x)),
        )
        .reset_index()
    )

    # Merge only the ranking columns back into the original dataset
    # This preserves the original order of processed_data
    labeled_data = processed_data.merge(ranked, on=group_cols, how="left")

    # Print stats (optional)
    id_to_name = {v: k for k, v in sampler_map.items()}
    print(f"Value counts for best samplers according to {objective}:")

    # Get best samplers for stats
    best_idx = (
        processed_data.groupby(group_cols)[objective].idxmin()
        if ascending
        else processed_data.groupby(group_cols)[objective].idxmax()
    )
    best_samplers = processed_data.loc[best_idx, ["sampler"]]

    for label, count in best_samplers["sampler"].value_counts().items():
        name = id_to_name.get(label, label)
        print(f" {label} - {name}: {count}")

    return labeled_data


def normalize_data(processed_data, objective, method="zscore"):
    print(f"Normalizing with {method}...")
    df = processed_data.copy()
    col = df[[objective]].values

    objective_original = objective + "_original"
    df[objective_original] = df[objective]

    if method == "minmax":
        scaler = MinMaxScaler()
        df[objective] = scaler.fit_transform(col)

    elif method == "zscore":
        scaler = StandardScaler()
        df[objective] = scaler.fit_transform(col)

    elif method == "mean":
        x = col.flatten()
        df[objective] = (x - np.mean(x)) / (np.max(x) - np.min(x))

    elif method == "robust":
        scaler = RobustScaler()
        df[objective] = scaler.fit_transform(col)

    elif method == "log":
        x = col.flatten()
        df[objective] = np.log1p(x - np.min(x))

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df, objective_original


def remove_duplicates(x, y, target_col, runs):
    print(f"Removing duplicates...")
    ranked_values = None
    ranked_labels = None
    dataset = None

    tmp_df = x.copy()
    tmp_df[target_col] = y.values

    original_cols = x.columns.tolist()
    print(f"original cols: {original_cols}")
    if "ranked_values" in original_cols:
        original_cols.remove("ranked_values")
    if "ranked_labels" in original_cols:
        original_cols.remove("ranked_labels")
    if "dataset" in original_cols:
        original_cols.remove("dataset")
    drop_duplicates_cols = original_cols.copy()
    if target_col not in drop_duplicates_cols:
        drop_duplicates_cols.append(target_col)

    if runs is not None:
        tmp_df["runs"] = runs
        drop_duplicates_cols.append("runs")

    print(f"drop duplicates cols: {drop_duplicates_cols}")
    df_unique = tmp_df.drop_duplicates(subset=drop_duplicates_cols, keep="first")
    x = df_unique[original_cols].reset_index(drop=True)
    y = df_unique[target_col].reset_index(drop=True)

    if "ranked_values" in df_unique.columns:
        ranked_values = df_unique["ranked_values"].reset_index(drop=True)
    if "ranked_labels" in df_unique.columns:
        ranked_labels = df_unique["ranked_labels"].reset_index(drop=True)
    if "dataset" in df_unique.columns:
        dataset = df_unique["dataset"].reset_index(drop=True)

    return x, y, ranked_values, ranked_labels, dataset


def is_perfect_ascending(nums):
    return nums == list(range(1, len(nums) + 1))


def build_random_classification_preds(y_test):
    n = len(y_test)

    y_pred_random = pd.Series(
        [list(np.random.permutation(8)) for _ in range(n)], index=y_test.index
    )

    first_elements = y_pred_random.apply(lambda x: x[0])

    tmp_pred = y_pred_random.reset_index(drop=True)

    def ranking_to_probabilities(rank_list):
        n = len(rank_list)
        # descending weights: n, n-1, ..., 1
        weights = np.arange(n, 0, -1)

        # probability vector initialized with zeros
        probs = np.zeros(n)

        # fill in probability vector based on rank positions
        for pos, idx in enumerate(rank_list):
            probs[idx] = weights[pos]

        # normalize
        probs = probs / probs.sum()
        return probs

    # Apply to entire Series
    prob_matrix = tmp_pred.apply(ranking_to_probabilities)
    prob_matrix = np.vstack(prob_matrix.values)

    return first_elements, prob_matrix


def build_random_regression_preds(data, group_cols, objective):
    # --- Step 0: clean group_cols ---
    group_cols = list(group_cols)
    print(f"Group cols: {group_cols}")
    for col in ["label", "sampler"]:
        if col in group_cols:
            group_cols.remove(col)
    if "run" not in data.columns and "run" in group_cols:
        group_cols.remove("run")
    if (
        "hidden" in group_cols
        and "hidden" not in data.columns
        and "hidden_per_layer" in data.columns
    ):
        group_cols = ["hidden_per_layer" if c == "hidden" else c for c in group_cols]

    missing_group_cols = [col for col in group_cols if col not in data.columns]
    if missing_group_cols:
        print(f"Ignoring unavailable grouping columns: {missing_group_cols}")
        group_cols = [col for col in group_cols if col in data.columns]

    keep_cols = group_cols + ["sampler", "ranked_labels", "ranked_values"]

    # --- Step 1: true labels ---
    best_idx = (
        data.groupby(group_cols)[objective].idxmax()
        if objective == "test_acc"
        else data.groupby(group_cols)[objective].idxmin()
    )
    best_samplers = data.loc[best_idx, keep_cols].rename(columns={"sampler": "label"})

    print(best_samplers)

    # --- Step 2: Build random predictions ---
    ALL_SAMPLERS = sorted(data["sampler"].unique())
    n_samplers = len(ALL_SAMPLERS)
    n_groups = len(best_samplers)

    # Generate random rankings for each group
    random_rankings = [list(np.random.permutation(n_samplers)) for _ in range(n_groups)]
    pred_labels = [ALL_SAMPLERS[ranking[0]] for ranking in random_rankings]

    # Convert rankings to probabilities
    def ranking_to_probabilities(rank_list):
        n = len(rank_list)
        weights = np.arange(n, 0, -1)  # descending weights: n, n-1, ..., 1
        probs = np.zeros(n)
        for pos, idx in enumerate(rank_list):
            probs[idx] = weights[pos]
        return probs / probs.sum()

    sampler_proba = np.array(
        [ranking_to_probabilities(ranking) for ranking in random_rankings]
    )

    # --- Step 3: Add predictions to labels ---
    best_samplers["pred_label"] = pred_labels
    best_samplers["sampler_list"] = [ALL_SAMPLERS] * n_groups
    best_samplers["sampler_proba"] = list(sampler_proba)

    # --- Step 4: return values ---
    return (
        best_samplers["label"],  # true labels
        best_samplers["pred_label"],  # predicted labels (random)
        sampler_proba,  # probability matrix
        best_samplers["ranked_labels"],
        best_samplers["ranked_values"],
        data,
    )


def build_majority_preds(y_test, y_train):
    print(f"y_train: {y_train}")
    y_train_val_counts = y_train.value_counts()
    labels_list = y_train_val_counts.index.tolist()
    most_common = y_train_val_counts.idxmax()
    y_pred_majority = pd.Series(most_common, index=y_test.index)

    if len(labels_list) != 8:
        expected = set(range(0, 8))
        missing = sorted(expected - set(labels_list))
        for m in missing:
            labels_list.append(m)

    length = len(y_test)
    y_pred_majority_ranked = pd.Series([labels_list] * length)

    # Convert rankings to probabilities
    def ranking_to_probabilities(rank_list):
        n = len(rank_list)
        weights = np.arange(n, 0, -1)  # descending weights: n, n-1, ..., 1
        probs = np.zeros(n)
        for pos, idx in enumerate(rank_list):
            probs[idx] = weights[pos]
        return probs / probs.sum()

    sampler_proba_majority = np.array(
        [ranking_to_probabilities(ranking) for ranking in y_pred_majority_ranked]
    )

    return (
        y_pred_majority,
        sampler_proba_majority,
    )


def build_table_preds(y_test, y_train, dataset, objective):
    def get_rule(dataset):
        """Return an exact table row, or the closest row when unmatched."""
        characteristics = GRAPH_CHARACTERISTICS[dataset]
        degree = characteristics["avg_degree"]
        clustering = characteristics["avg_clustering_coeff"]

        for rule in TABLE_RULES:
            degree_min, degree_max = rule["degree"]
            clustering_min, clustering_max = rule["clustering"]
            if (
                degree_min <= degree <= degree_max
                and clustering_min <= clustering <= clustering_max
            ):
                return rule

        # The table intervals do not cover every graph in GRAPH_CHARACTERISTICS.
        # Use distance to the interval boundaries as a deterministic fallback so
        # an uncovered dataset does not abort the complete baseline run.
        def interval_distance(value, bounds):
            lower, upper = bounds
            if value < lower:
                return lower - value
            if value > upper:
                return value - upper
            return 0.0

        def rule_distance(rule):
            degree_min, degree_max = rule["degree"]
            clustering_min, clustering_max = rule["clustering"]
            degree_scale = max(degree_max - degree_min, 1.0)
            clustering_scale = max(clustering_max - clustering_min, 0.01)
            return np.hypot(
                interval_distance(degree, rule["degree"]) / degree_scale,
                interval_distance(clustering, rule["clustering"])
                / clustering_scale,
            )

        return min(TABLE_RULES, key=rule_distance)

    rule = get_rule(dataset)
    print(f"Rule {rule}")

    recommended = [sampler_map[name] for name in rule[objective]]
    labels_list = recommended + [
        sampler_id for sampler_id in range(8) if sampler_id not in recommended
    ]
    y_pred_table = pd.Series(labels_list[0], index=y_test.index)
    y_pred_table_ranked = pd.Series([labels_list] * len(y_test), index=y_test.index)

    def ranking_to_probabilities(rank_list):
        n = len(rank_list)
        weights = np.arange(n, 0, -1)
        probs = np.zeros(n)
        for pos, sampler_id in enumerate(rank_list):
            probs[sampler_id] = weights[pos]
        return probs / probs.sum()

    sampler_proba_table = np.array(
        [ranking_to_probabilities(ranking) for ranking in y_pred_table_ranked]
    )

    return (
        y_pred_table,
        sampler_proba_table,
    )


def run_enhanced_sampler_prediction(
    data,
    arch_map,
    sampler_map,
    test_dataset="cora",
    objective="test_acc",
    use_regression_approach=True,
    use_advanced_features=True,
    onehot=False,
    normalize=False,
    remove_dup=False,
    seed=42,
):
    """
    Enhanced pipeline focused on maximizing accuracy.
    """

    print("ENHANCED SAMPLER PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Objective: {objective}")
    print(f"Test dataset: {test_dataset}")
    print(f"Using regression approach: {use_regression_approach}")
    print(f"Using advanced features: {use_advanced_features}")

    if test_dataset == "products":
        test_dataset = "ogbn-products"

    # Preprocess data.
    processed_data = data.copy()
    processed_data = processed_data.dropna(subset=[objective])
    processed_data = processed_data.drop(
        columns=["diameter", "avg_shortest_path_len"], errors="ignore"
    )
    processed_data = processed_data.fillna(0)

    # Apply mappings
    processed_data["architecture"] = processed_data["architecture"].map(arch_map)
    processed_data["sampler"] = processed_data["sampler"].map(sampler_map)

    if test_dataset not in ["coauthorph", "reddit"]:
        processed_data = processed_data[
            ~processed_data["dataset"].isin(["coauthorph", "reddit"])
        ]

        wandb.log(
            {
                "included_datasets": processed_data["dataset"].unique(),
                "excluded_datasets": ["coauthorph", "reddit"],
            }
        )
    else:
        wandb.log(
            {
                "included_datasets": processed_data["dataset"].unique(),
                "excluded_datasets": [],
            }
        )

    print(processed_data["dataset"].unique())

    core_cols = [
        "dataset",
        "architecture",
        "hidden",
        "run",
        "fanout",
        "layer",
        "epochs",
        "sampler",
    ]
    available_core_cols = [col for col in core_cols if col in processed_data.columns]

    # Create labels the traditional way
    group_cols = [
        col for col in available_core_cols if col not in ["sampler", objective]
    ]

    target_col = objective

    if use_regression_approach:
        print("Using regression approach...")
        if normalize:
            labeled_data, original_column = normalize_data(processed_data, objective)
            labeled_data = prepare_regression_labels(
                labeled_data, group_cols, original_column
            )
            labeled_data = labeled_data.drop(original_column, axis=1)
        else:
            labeled_data = processed_data.copy()
            labeled_data = prepare_regression_labels(
                labeled_data, group_cols, objective
            )
    else:
        labeled_data, target_col = prepare_classification_labels(
            processed_data, group_cols, objective
        )

    runs = None
    if "run" in labeled_data.columns:
        runs = labeled_data["run"]
        labeled_data = labeled_data.drop("run", axis=1)

    if use_advanced_features:
        labeled_data = advanced_feature_engineering(
            labeled_data, use_regression_approach
        )

    if onehot:
        print("One-hot encoding...")
        # --- One-hot encode labels ---
        onehot_labels = pd.get_dummies(labeled_data["label"], prefix="label").astype(
            int
        )

        # Concatenate one-hot columns
        labeled_data = pd.concat([labeled_data, onehot_labels], axis=1)

        target_cols = onehot_labels.columns

    train_data = labeled_data[labeled_data["dataset"] != test_dataset].reset_index(
        drop=True
    )
    test_data = labeled_data[labeled_data["dataset"] == test_dataset].reset_index(
        drop=True
    )

    def fill_missing(lst):
        full_set = set(range(8))
        missing = sorted(full_set - set(lst))
        return lst + missing

    def pad_with_value(lst, target_len=8, fill_value=0):
        return lst + [fill_value] * (target_len - len(lst))

    test_data["ranked_labels"] = test_data["ranked_labels"].apply(fill_missing)
    train_data["ranked_labels"] = train_data["ranked_labels"].apply(fill_missing)
    if objective == "test_acc":
        fill_value = 0
    elif objective in ("time_per_epoch", "used_memory"):
        fill_value = 1000000

    test_data["ranked_values"] = test_data["ranked_values"].apply(
        lambda x: pad_with_value(x, target_len=8, fill_value=fill_value)
    )
    train_data["ranked_values"] = train_data["ranked_values"].apply(
        lambda x: pad_with_value(x, target_len=8, fill_value=fill_value)
    )

    print(f"train data columns: {train_data.columns}")

    ranked_labels = test_data["ranked_labels"]
    ranked_values = test_data["ranked_values"]

    ranked_labels_train = train_data["ranked_labels"]
    ranked_values_train = train_data["ranked_values"]

    print("Number of unique datasets in training:", train_data["dataset"].nunique())
    print("Number of samples per dataset:")
    print(train_data["dataset"].value_counts())

    print(f"\nData sizes:")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    wandb.log({"train_size": len(train_data), "test_size": len(test_data)})

    exclude_cols = [
        objective,
        target_col,
        "sampler",
        "predicted_performance",
        "sampler_rank",
        "label_0",
        "label_1",
        "label_2",
        "label_3",
        "label_4",
        "label_5",
        "label_6",
        "label_7",
        "rank",
        "ranked_labels",
        "ranked_values",
        "best_value",
    ]
    if not use_regression_approach:
        exclude_cols.append("label")
    if use_regression_approach:
        if "sampler" in exclude_cols:
            exclude_cols.remove("sampler")

    feature_cols = [col for col in labeled_data.columns if col not in exclude_cols]

    # Remove any remaining non-numeric columns
    numeric_features = []
    for col in feature_cols:
        if train_data[col].dtype in ["int64", "float64", "bool"]:
            numeric_features.append(col)

    print(f"Using {len(numeric_features)} features for training")
    wandb.log({"num_numeric_features": len(numeric_features)})

    X_train = train_data[numeric_features]
    y_train = train_data[target_col]
    X_test = test_data[numeric_features]
    y_test = test_data[target_col]

    groups = train_data["dataset"]
    datasets = train_data["dataset"]

    if remove_dup:
        tmp_X_train = X_train.copy()
        tmp_X_train["ranked_values"] = ranked_values_train
        tmp_X_train["ranked_labels"] = ranked_labels_train
        X_train, y_train, ranked_values_train, ranked_labels_train, groups = (
            remove_duplicates(tmp_X_train, y_train, target_col, runs)
        )
        tmp_X_test = X_test.copy()
        tmp_X_test["ranked_values"] = ranked_values
        tmp_X_test["ranked_labels"] = ranked_labels
        X_test, y_test, ranked_values, ranked_labels, _ = remove_duplicates(
            tmp_X_test, y_test, target_col, runs
        )

    wandb.log({"train_size": len(X_train), "test_size": len(X_test)})

    X_test_fanout_col = X_test["fanout"]
    X_test_layer_col = X_test["layer"]
    X_test_epochs_col = X_test["epochs"]

    X_train_fanout_col = X_train["fanout"]
    X_train_layer_col = X_train["layer"]
    X_train_epochs_col = X_train["epochs"]

    print(X_train.columns)

    if len(numeric_features) > 20:  # Only if we have many features
        selector = SelectKBest(f_classif, k=min(20, len(numeric_features)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        mask = selector.get_support()
        selected_columns = X_train.columns[mask].tolist()

        if use_regression_approach:
            if "sampler" in X_train.columns and "sampler" not in selected_columns:
                selected_columns.append("sampler")
                # Add the actual column
                X_train_selected = np.hstack(
                    [X_train_selected, X_train[["sampler"]].values]
                )
                X_test_selected = np.hstack(
                    [X_test_selected, X_test[["sampler"]].values]
                )
            if (
                "architecture" in X_train.columns
                and "architecture" not in selected_columns
            ):
                selected_columns.append("architecture")
                X_train_selected = np.hstack(
                    [X_train_selected, X_train[["architecture"]].values]
                )
                X_test_selected = np.hstack(
                    [X_test_selected, X_test[["architecture"]].values]
                )
            if "dataset" in X_train.columns and "dataset" not in selected_columns:
                selected_columns.append("dataset")
                X_train_selected = np.hstack(
                    [X_train_selected, X_train[["dataset"]].values]
                )
                X_test_selected = np.hstack(
                    [X_test_selected, X_test[["dataset"]].values]
                )

            # Grouping columns are needed later to recover the best sampler.
            # Feature selection may otherwise discard one of them (for example,
            # ``hidden``), causing the regression baseline to lose its groups.
            for col in group_cols:
                if col in X_train.columns and col not in selected_columns:
                    selected_columns.append(col)
                    X_train_selected = np.hstack(
                        [X_train_selected, X_train[[col]].values]
                    )
                    X_test_selected = np.hstack(
                        [X_test_selected, X_test[[col]].values]
                    )

        print(f"Selected columns: {selected_columns}")

        # Re-wrap as DataFrame with column names
        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=selected_columns,
            index=X_train.index,
        )

        # Transform test data and also re-wrap
        X_test_selected = pd.DataFrame(
            X_test_selected,
            columns=selected_columns,
            index=X_test.index,
        )

        selected_features = [
            numeric_features[i] for i in selector.get_support(indices=True)
        ]
        print(f"Selected top {len(selected_features)} features: {selected_features}")
        wandb.log({"selected_features": selected_features})

        X_train, X_test = X_train_selected, X_test_selected

    if onehot:
        y_train = train_data[target_cols]
        y_test = test_data[target_cols]

    scaler = StandardScaler()

    if use_regression_approach:

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train.drop(columns="sampler")),
            columns=[col for col in X_train.columns if col != "sampler"],
            index=X_train.index,
        )

        # Add back the 'sampler' column unchanged
        X_train_scaled["sampler"] = X_train["sampler"]

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test.drop(columns="sampler")),
            columns=[col for col in X_train.columns if col != "sampler"],
            index=X_test.index,
        )

        X_test_scaled["sampler"] = X_test["sampler"]

    else:
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(
            scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index
        )

    X_train_scaled["dataset"] = datasets

    # numeric mapping for dataset
    ds_map = {d: i for i, d in enumerate(train_data["dataset"].unique())}
    ds_ids = train_data["dataset"].map(ds_map).values
    corrs = {
        col: np.corrcoef(train_data[col].astype(float).fillna(0), ds_ids)[0, 1]
        for col in numeric_features
    }
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("Top features correlated with dataset id (abs corr):", sorted_corrs)

    name = "random"
    name2 = "majority"
    name3 = "expert"
    results = {}

    relative_utility_ratios = {}
    relative_utility_losses = {}
    relative_penalized_losses = {}
    relative_log_losses = {}
    rank_based_scores = {}

    X_train_scaled["fanout"] = X_train_fanout_col
    X_train_scaled["layer"] = X_train_layer_col
    X_train_scaled["epochs"] = X_train_epochs_col
    X_train_scaled[objective] = y_train
    X_train_scaled["ranked_labels"] = ranked_labels_train
    X_train_scaled["ranked_values"] = ranked_values_train

    X_test_scaled["ranked_labels"] = ranked_labels
    X_test_scaled["ranked_values"] = ranked_values

    np.random.seed(seed)
    if use_regression_approach:
        # Apply the mapping
        X_test_scaled["dataset"] = test_dataset
        X_test_scaled["fanout"] = X_test_fanout_col
        X_test_scaled["layer"] = X_test_layer_col
        X_test_scaled["epochs"] = X_test_epochs_col
        X_test_scaled["sampler"] = X_test["sampler"]
        X_test_scaled[objective] = y_test
        (y_test, y_pred, y_proba, ranked_labels_, ranked_values_, _) = (
            build_random_regression_preds(X_test_scaled, group_cols, objective)
        )
        X_train_scaled["sampler"] = X_train["sampler"]
        X_train_scaled[objective] = y_train
        print(X_train_scaled["dataset"])
        (_, y_train, _, ranked_labels_train_, ranked_values_train_, _) = (
            build_random_regression_preds(X_train_scaled, group_cols, objective)
        )
        print(f"y_train line 2051: {y_train}")
        (
            y_pred_majority,
            y_proba_majority,
        ) = build_majority_preds(y_test, y_train.astype(int))
        (y_pred_table, y_proba_table) = build_table_preds(
            y_test, y_train.astype(int), test_dataset, objective
        )
    else:
        y_pred, y_proba = build_random_classification_preds(y_test)

        (
            y_pred_majority,
            y_proba_majority,
        ) = build_majority_preds(y_test, y_train)
        y_pred_table, y_proba_table = build_table_preds(
            y_test, y_train, test_dataset, objective
        )
        ranked_labels_ = ranked_labels.copy()
        ranked_values_ = ranked_values.copy()

    all_labels = list(range(y_proba.shape[1]))

    print("Evaluating random baseline...", flush=True)
    accuracy = evaluate_with_detailed_metrics(
        y_test.astype(int),
        y_pred.astype(int),
        all_labels,
    )

    print("Evaluating majority baseline...", flush=True)
    accuracy_majority = evaluate_with_detailed_metrics(
        y_test.astype(int), y_pred_majority.astype(int), all_labels
    )

    print("Evaluating expert (table) baseline...", flush=True)
    accuracy_table = evaluate_with_detailed_metrics(
        y_test.astype(int), y_pred_table.astype(int), all_labels
    )

    print("Computing top-k metrics...", flush=True)
    observed_labels = sorted(np.unique(y_test.astype(int)))
    label_to_dense = {label: index for index, label in enumerate(observed_labels)}
    y_test_dense = np.asarray(
        [label_to_dense[label] for label in y_test.astype(int)], dtype=int
    )

    def align_probabilities(probabilities):
        columns_to_remove = [
            label
            for label in range(probabilities.shape[1])
            if label not in observed_labels
        ]
        return np.delete(probabilities, columns_to_remove, axis=1)

    def top_k_scores(probabilities):
        n_classes = probabilities.shape[1]
        if n_classes == 1:
            return {k: 1.0 for k in range(1, 8)}
        if n_classes == 2:
            top_1 = accuracy_score(y_test_dense, probabilities.argmax(axis=1))
            return {k: (top_1 if k == 1 else 1.0) for k in range(1, 8)}

        dense_labels = np.arange(n_classes)
        return {
            k: (
                1.0
                if k >= n_classes
                else top_k_accuracy_score(
                    y_test_dense, probabilities, k=k, labels=dense_labels
                )
            )
            for k in range(1, 8)
        }

    top_accuracies = top_k_scores(align_probabilities(y_proba))
    top_accuracies_majority = top_k_scores(align_probabilities(y_proba_majority))
    top_accuracies_table = top_k_scores(align_probabilities(y_proba_table))

    print(f"top_accuracies: {top_accuracies}")
    print(f"top_accuracies_majority: {top_accuracies_majority}")
    print(f"top_accuracies_expert: {top_accuracies_table}")

    print("Computing utility metrics...", flush=True)
    (
        relative_utility_ratio,
        relative_utility_loss_score,
        relative_penalized_loss_score,
        relative_log_loss_utility_score,
        rank_based_score,
    ) = evaluate_relative_score(
        y_test.astype(int),
        y_pred.astype(int),
        ranked_labels_,
        ranked_values_,
        objective,
    )

    (
        relative_utility_ratio_majority,
        relative_utility_loss_score_majority,
        relative_penalized_loss_score_majority,
        relative_log_loss_utility_score_majority,
        rank_based_score_majority,
    ) = evaluate_relative_score(
        y_test.astype(int),
        y_pred_majority.astype(int),
        ranked_labels_,
        ranked_values_,
        objective,
    )

    (
        relative_utility_ratio_table,
        relative_utility_loss_score_table,
        relative_penalized_loss_score_table,
        relative_log_loss_utility_score_table,
        rank_based_score_table,
    ) = evaluate_relative_score(
        y_test.astype(int),
        y_pred_table.astype(int),
        ranked_labels_,
        ranked_values_,
        objective,
    )

    results[name] = accuracy
    results[name2] = accuracy_majority
    results[name3] = accuracy_table

    results_top_1 = {}
    results_top_2 = {}
    results_top_3 = {}
    results_top_4 = {}
    results_top_5 = {}
    results_top_6 = {}
    results_top_7 = {}
    results[f"accuracy_{name}"] = accuracy
    results[f"accuracy_{name2}"] = accuracy_majority
    results[f"accuracy_{name3}"] = accuracy_table
    results_top_1[f"top_1_accuracy_{name}"] = top_accuracies[1]
    results_top_1[f"top_1_accuracy_{name2}"] = top_accuracies_majority[1]
    results_top_2[f"top_2_accuracy_{name}"] = top_accuracies[2]
    results_top_2[f"top_2_accuracy_{name2}"] = top_accuracies_majority[2]
    results_top_3[f"top_3_accuracy_{name}"] = top_accuracies[3]
    results_top_3[f"top_3_accuracy_{name2}"] = top_accuracies_majority[3]
    results_top_4[f"top_4_accuracy_{name}"] = top_accuracies[4]
    results_top_4[f"top_4_accuracy_{name2}"] = top_accuracies_majority[4]
    results_top_5[f"top_5_accuracy_{name}"] = top_accuracies[5]
    results_top_5[f"top_5_accuracy_{name2}"] = top_accuracies_majority[5]
    results_top_6[f"top_6_accuracy_{name}"] = top_accuracies[6]
    results_top_6[f"top_6_accuracy_{name2}"] = top_accuracies_majority[6]
    results_top_7[f"top_7_accuracy_{name}"] = top_accuracies[7]
    results_top_7[f"top_7_accuracy_{name2}"] = top_accuracies_majority[7]

    results_top_1[f"top_1_accuracy_{name3}"] = top_accuracies_table[1]
    results_top_2[f"top_2_accuracy_{name3}"] = top_accuracies_table[2]
    results_top_3[f"top_3_accuracy_{name3}"] = top_accuracies_table[3]
    results_top_4[f"top_4_accuracy_{name3}"] = top_accuracies_table[4]
    results_top_5[f"top_5_accuracy_{name3}"] = top_accuracies_table[5]
    results_top_6[f"top_6_accuracy_{name3}"] = top_accuracies_table[6]
    results_top_7[f"top_7_accuracy_{name3}"] = top_accuracies_table[7]

    relative_utility_ratios[f"relative_utility_ratio_{name}"] = relative_utility_ratio
    relative_utility_ratios[f"relative_utility_ratio_{name2}"] = (
        relative_utility_ratio_majority
    )
    relative_utility_ratios[f"relative_utility_ratio_{name3}"] = (
        relative_utility_ratio_table
    )
    relative_utility_losses[f"relative_utility_loss_{name}"] = (
        relative_utility_loss_score
    )
    relative_utility_losses[f"relative_utility_loss_{name2}"] = (
        relative_utility_loss_score_majority
    )
    relative_utility_losses[f"relative_utility_loss_{name3}"] = (
        relative_utility_loss_score_table
    )
    relative_penalized_losses[f"relative_penalized_loss_{name}"] = (
        relative_penalized_loss_score
    )
    relative_penalized_losses[f"relative_penalized_loss_{name2}"] = (
        relative_penalized_loss_score_majority
    )
    relative_penalized_losses[f"relative_penalized_loss_{name3}"] = (
        relative_penalized_loss_score_table
    )
    relative_log_losses[f"relative_log_loss_{name}"] = relative_log_loss_utility_score
    relative_log_losses[f"relative_log_loss_{name2}"] = (
        relative_log_loss_utility_score_majority
    )
    relative_log_losses[f"relative_log_loss_{name3}"] = (
        relative_log_loss_utility_score_table
    )
    rank_based_scores[f"rank_based_score_{name}"] = rank_based_score
    rank_based_scores[f"rank_based_score_{name2}"] = rank_based_score_majority
    rank_based_scores[f"rank_based_score_{name3}"] = rank_based_score_table

    results[f"score_{name}"] = relative_utility_ratio
    results[f"score_{name2}"] = relative_utility_ratio_majority
    results[f"score_{name3}"] = relative_utility_ratio_table

    print("-" * 40)

    print("Logging metrics...", flush=True)
    wandb.log(
        {
            **relative_utility_ratios,
            **relative_utility_losses,
            **relative_penalized_losses,
            **relative_log_losses,
            **rank_based_scores,
            **results,
            **results_top_1,
            **results_top_2,
            **results_top_3,
            **results_top_4,
            **results_top_5,
            **results_top_6,
            **results_top_7,
        }
    )

    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Test_Accuracy"])
    results_df = results_df.sort_values("Test_Accuracy", ascending=False)

    scores = {}
    scores["Relative Utility Score"] = relative_utility_ratios
    scores["Relative Utility Loss Score"] = relative_utility_losses
    scores["Relative Penalized Loss Score"] = relative_penalized_losses
    scores["Relative Log Loss Score"] = relative_log_losses
    scores["Rank Based Score"] = rank_based_scores

    print("\nFINAL RESULTS SUMMARY:")
    print("=" * 30)
    print(results_df)

    # Convert dict-of-dicts → DataFrame
    df_scores = pd.DataFrame(scores)

    # Optional: sort rows by a specific score
    df_scores = df_scores.sort_values("Relative Utility Score", ascending=False)

    print("Scores summary:")
    print(df_scores)

    wandb.log(
        {
            "final_results": wandb.Table(dataframe=results_df),
            "final_scores": wandb.Table(dataframe=df_scores.reset_index()),
        }
    )

    print("Baseline evaluation complete.", flush=True)
    return {
        "results": results_df,
        "scores": df_scores,
        "feature_cols": numeric_features,

    }


def analyze_feature_importance_and_leakage(models, feature_names, target_col="is_best"):
    """Analyze what features the models are using to make predictions."""
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Get feature importance from tree-based models
    importance_data = {}

    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        if model_name in models:
            model = models[model_name]
            if hasattr(model, "feature_importances_"):
                importance_data[model_name] = model.feature_importances_

    if importance_data:
        # Create DataFrame of feature importances
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        importance_df["mean_importance"] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values("mean_importance", ascending=False)

        print("Top 10 Most Important Features:")
        print(importance_df.head(10))

        # Check for suspiciously high importance
        top_importance = importance_df.iloc[0]["mean_importance"]
        if top_importance > 0.8:
            print(f"\n⚠️  WARNING: Very high feature importance detected!")
            print(
                f"   Feature '{importance_df.index[0]}' has {top_importance:.3f} importance"
            )
            print("   This might indicate data leakage or overfitting")

        return importance_df
    else:
        print("No tree-based models found for feature importance analysis")
        return None


@hydra.main(config_path=".", config_name="config_baseline")
def main(cfg: DictConfig):
    """Example usage of the sampler selection pipeline."""

    normalize = cfg.normalize
    use_regression_approach = cfg.use_regression_approach
    if not use_regression_approach:
        normalize = False

    objective = cfg.objective
    merge_runs = cfg.merge_runs

    config = {
        "test_dataset": cfg.test_dataset,
        "objective": objective,
        "use_regression_approach": cfg.use_regression_approach,
        "use_advanced_features": cfg.use_advanced_features,
        "onehot": cfg.onehot,
        "normalize": normalize,
        "merge_runs": merge_runs,
        "remove_duplicates": cfg.remove_dup,
        "seed": cfg.seed,
    }

    run = wandb.init(
        project="samplerprediction_baselines",
        name=f"samplerprediction_baseline_{cfg.test_dataset}_{objective}_{cfg.use_regression_approach}_{cfg.use_advanced_features}_{cfg.onehot}_{cfg.normalize}_{cfg.merge_runs}_{cfg.remove_dup}",
        config=config,
        mode="offline",
        reinit="finish_previous",
    )

    try:
        results = load_results(
            file="full_results_bias.pkl",
        )

        preprocessed = prepare_data(results, objective)
        print(f"Preprocessed: {preprocessed}")
        data = build_data_with_graph_characteristics(
            preprocessed, GRAPH_CHARACTERISTICS, objective, merge_runs
        )

        print(f"Data: {data}")

        results = run_enhanced_sampler_prediction(
            data=data,
            arch_map=arch_map,
            sampler_map=sampler_map,
            test_dataset=cfg.test_dataset,
            objective=objective,
            use_regression_approach=cfg.use_regression_approach,
            use_advanced_features=cfg.use_advanced_features,
            onehot=cfg.onehot,
            normalize=normalize,
            remove_dup=cfg.remove_dup,
            seed=cfg.seed,
        )
    finally:
        print("Finalizing W&B run...", flush=True)
        run.finish()
        print("W&B run finalized.", flush=True)


if __name__ == "__main__":
    main()
