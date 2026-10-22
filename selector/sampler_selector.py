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

import shutil

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


def analyze_prediction_difficulty(data, objective="test_acc"):
    """Analyze why certain configurations might be hard to predict."""
    print("=" * 60)
    print("PREDICTION DIFFICULTY ANALYSIS")
    print("=" * 60)

    # Get core config columns
    core_cols = ["architecture", "hidden", "layer", "epochs"]
    available_core_cols = [col for col in core_cols if col in data.columns]

    difficulty_stats = []

    for name, group in data.groupby(available_core_cols):
        if len(group) > 1:
            samplers = group["sampler"].unique()
            performances = []

            for sampler in samplers:
                sampler_data = group[group["sampler"] == sampler]
                if len(sampler_data) > 0:
                    avg_perf = sampler_data[objective].mean()
                    performances.append(avg_perf)

            if len(performances) > 1:
                perf_gap = max(performances) - min(performances)
                perf_std = np.std(performances)
                best_perf = max(performances)

                difficulty_stats.append(
                    {
                        "config": str(name),
                        "perf_gap": perf_gap,
                        "perf_std": perf_std,
                        "best_perf": best_perf,
                        "num_samplers": len(performances),
                    }
                )

    if len(difficulty_stats) > 0:
        df_stats = pd.DataFrame(difficulty_stats)

        print(f"Configuration difficulty analysis:")
        print(f"Mean performance gap: {df_stats['perf_gap'].mean():.4f}")
        print(f"Std of performance gaps: {df_stats['perf_gap'].std():.4f}")

        # Identify hard vs easy configs
        easy_threshold = df_stats["perf_gap"].quantile(0.75)
        hard_configs = df_stats[df_stats["perf_gap"] < easy_threshold]
        easy_configs = df_stats[df_stats["perf_gap"] >= easy_threshold]

        print(
            f"Easy configs (large gaps): {len(easy_configs)} ({len(easy_configs) / len(df_stats) * 100:.1f}%)"
        )
        print(
            f"Hard configs (small gaps): {len(hard_configs)} ({len(hard_configs) / len(df_stats) * 100:.1f}%)"
        )

        return df_stats

    return None


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


class RidgeWithProba(RidgeClassifier):
    def predict_proba(self, X):
        decision = self.decision_function(X)
        return scipy.special.softmax(decision, axis=1)


def train_classification_models(
    X,
    y,
    cv_folds=5,
    random_state=42,
    scoring="accuracy",
    use_group_cv=False,
    groups=None,
):
    """Train classification models with hyperparameter tuning."""

    classes = np.unique(y)
    print(f"Classes: {classes}")
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    # Define models with hyperparameter grids
    models_and_params = {
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=random_state, max_iter=1000, class_weight="balanced"
            ),
            "params": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            },
        },
        "RidgeClassifier": {
            "model": RidgeWithProba(random_state=random_state, class_weight="balanced"),
            "params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        },
        "SGDClassifier": {
            "model": SGDClassifier(random_state=random_state, class_weight="balanced"),
            "params": {
                "loss": ["log_loss", "hinge"],
                "alpha": [0.0001, 0.001, 0.01],
                "penalty": ["l2", "l1", "elasticnet"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=random_state, class_weight="balanced"
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(
                random_state=random_state, class_weight="balanced"
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                random_state=random_state, verbose=0, class_weights=class_weights
            ),
            "params": {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.1],
                "n_estimators": [200, 500],
                "l2_leaf_reg": [1, 3, 5],
            },
        },
    }

    print("Best sampler val counts:")
    print(y.value_counts())

    # Select the cross-validation strategy.
    if use_group_cv:
        if groups is None:
            raise ValueError("`groups` must be provided when use_group_cv=True.")
        cv = LeaveOneGroupOut()
        print(f"Using GroupKFold with {cv_folds} folds.")
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        print(f"Using standard KFold with {cv_folds} folds.")

    best_models = {}
    results = []

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        train_datasets = groups.iloc[train_idx].unique()
        val_datasets = groups.iloc[val_idx].unique()
        print(f"Fold {i + 1}")
        print("  Train datasets:", train_datasets)
        print("  Val dataset:", val_datasets)
        print(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        print("-" * 60)

    print("Training classification models with hyperparameter tuning...")

    for name, config in models_and_params.items():
        print(f"Tuning {name}...")

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )

        if use_group_cv:
            grid_search.fit(X, y, groups=groups)
            test_groups_per_fold = []

            # Re-run the CV splitter only to record which group is test in each fold
            for train_idx, test_idx in cv.split(X, y, groups):
                test_group = np.unique(groups[test_idx])
                assert (
                    len(test_group) == 1
                )  # LeaveOneGroupOut ensures exactly 1 test group
                test_groups_per_fold.append(test_group[0])
        else:
            grid_search.fit(X, y)
            test_groups_per_fold = None

        best_models[name] = grid_search.best_estimator_

        results.append(
            {
                "Model": name,
                "Best_CV_Score": grid_search.best_score_,
                "Best_Params": str(grid_search.best_params_),
            }
        )
        wandb.log(
            {
                f"{name}_best_cv_score": grid_search.best_score_,
                f"{name}_best_params": grid_search.best_params_,
            }
        )

        print(
            f"Mean CV score: {np.mean(grid_search.cv_results_['mean_test_score']):.4f}"
        )
        print(
            f"Std deviation: {np.std(grid_search.cv_results_['mean_test_score']):.4f}"
        )

        # Get per-fold columns dynamically
        fold_cols = [
            col
            for col in grid_search.cv_results_.keys()
            if col.startswith("split") and col.endswith("_test_score")
        ]

        print("\nPer-fold CV performance (held out):")

        if use_group_cv:
            rows = []
            for fold_name, test_group in zip(fold_cols, test_groups_per_fold):
                score = grid_search.cv_results_[fold_name][grid_search.best_index_]
                rows.append({"dataset": test_group, "test_score": score})

            cv_summary = pd.DataFrame(rows).sort_values("test_score", ascending=False)

        else:
            # For normal KFold, you keep the original behavior
            cv_summary = pd.DataFrame(
                {
                    "fold": list(range(len(fold_cols))),
                    "test_score": [
                        grid_search.cv_results_[col][grid_search.best_index_]
                        for col in fold_cols
                    ],
                }
            ).sort_values("test_score", ascending=False)

        print(cv_summary)
        wandb.log({f"cv_summary_{name}": cv_summary})

    results_df = pd.DataFrame(results).sort_values("Best_CV_Score", ascending=False)
    print("\nTuning Results:")
    print(results_df)
    wandb.log({"results_df": results_df})

    return best_models


def train_regression_models(
    X, y, cv_folds=5, random_state=42, scoring="r2", use_group_cv=False, groups=None
):
    """Train regression models with hyperparameter tuning."""
    # Define models with hyperparameter grids
    models_and_params = {
        "LinearRegression": {"model": LinearRegression(), "params": {}},
        "Ridge": {
            "model": Ridge(random_state=random_state),
            "params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        },
        "Lasso": {
            "model": Lasso(random_state=random_state, max_iter=10000),
            "params": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            },
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=random_state),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(random_state=random_state),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        },
        "XGBoost": {
            "model": XGBRegressor(
                random_state=random_state, objective="reg:squarederror"
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
            },
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_state=random_state, silent=True),
            "params": {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [200, 500, 800],
                "l2_leaf_reg": [1, 3, 5, 7],
            },
        },
    }

    # Select the cross-validation strategy.
    if use_group_cv:
        if groups is None:
            raise ValueError("`groups` must be provided when use_group_cv=True.")
        cv = LeaveOneGroupOut()
        print(f"Using LeaveOneGroupOut with {len(groups)} groups.")
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        print(f"Using standard KFold with {cv_folds} folds.")

    best_models = {}
    results = []

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        train_datasets = groups.iloc[train_idx].unique()
        val_datasets = groups.iloc[val_idx].unique()
        print(f"Fold {i + 1}")
        print("  Train datasets:", train_datasets)
        print("  Val dataset:", val_datasets)
        print(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        print("-" * 60)

    print("Training regression models with hyperparameter tuning...")

    for name, config in models_and_params.items():
        print(f"Tuning {name}...")

        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )

        # Fit with group labels when group-based cross-validation is used.
        if use_group_cv:
            grid_search.fit(X, y, groups=groups)
            test_groups_per_fold = []

            # Re-run the CV splitter only to record which group is test in each fold
            for train_idx, test_idx in cv.split(X, y, groups):
                test_group = np.unique(groups[test_idx])
                assert (
                    len(test_group) == 1
                )  # LeaveOneGroupOut ensures exactly 1 test group
                test_groups_per_fold.append(test_group[0])
        else:
            grid_search.fit(X, y)
            test_groups_per_fold = None  # Not used for normal KFold

        best_models[name] = grid_search.best_estimator_

        results.append(
            {
                "Model": name,
                "Best_CV_Score": grid_search.best_score_,
                "Best_Params": str(grid_search.best_params_),
            }
        )

        wandb.log(
            {
                f"{name}_best_cv_score": grid_search.best_score_,
                f"{name}_best_params": grid_search.best_params_,
            }
        )

        print(
            f"Mean CV score: {np.mean(grid_search.cv_results_['mean_test_score']):.4f}"
        )
        print(
            f"Std deviation: {np.std(grid_search.cv_results_['mean_test_score']):.4f}"
        )
        # Get per-fold columns dynamically
        fold_cols = [
            col
            for col in grid_search.cv_results_.keys()
            if col.startswith("split") and col.endswith("_test_score")
        ]

        print("\nPer-fold CV performance (held out):")

        if use_group_cv:
            rows = []
            for fold_name, test_group in zip(fold_cols, test_groups_per_fold):
                score = grid_search.cv_results_[fold_name][grid_search.best_index_]
                rows.append({"dataset": test_group, "test_score": score})

            cv_summary = pd.DataFrame(rows).sort_values("test_score", ascending=False)

        else:
            # For normal KFold, you keep the original behavior
            cv_summary = pd.DataFrame(
                {
                    "fold": list(range(len(fold_cols))),
                    "test_score": [
                        grid_search.cv_results_[col][grid_search.best_index_]
                        for col in fold_cols
                    ],
                }
            ).sort_values("test_score", ascending=False)

        print(cv_summary)
        wandb.log({f"cv_summary_{name}": cv_summary})

    results_df = pd.DataFrame(results).sort_values("Best_CV_Score", ascending=False)
    print("\nTuning Results:")
    print(results_df)
    wandb.log({"results_df": results_df})

    return best_models


def create_ensemble_classifier(models, X, y):
    """Create a voting ensemble of the best models."""
    # Select top 3 models for ensemble
    top_models = list(models.items())[:3]

    # Keep only models that support probability
    probabilistic = [(name, m) for name, m in top_models if hasattr(m, "predict_proba")]

    ensemble = VotingClassifier(
        estimators=probabilistic, voting="soft"
    )

    ensemble.fit(X, y)
    return ensemble


def create_ensemble_regressor(models, X, y, top_n=3):
    """Create a voting ensemble of the top regression models.

    Args:
        models (dict): Dictionary of trained regression models (name -> model object).
        X (pd.DataFrame or np.ndarray): Training features.
        y (pd.Series or np.ndarray): Target values.
        top_n (int): Number of top models to include in the ensemble (default=3).

    Returns:
        VotingRegressor: Trained ensemble model.
    """
    # Select top N models (assumes models are ordered by performance)
    top_models = list(models.items())[:top_n]

    # Create a voting regressor (averages model predictions)
    ensemble = VotingRegressor(estimators=top_models)

    # Train ensemble on full data
    ensemble.fit(X, y)

    return ensemble


def evaluate_with_detailed_metrics(y_true, y_pred, class_names=None):
    """Comprehensive evaluation metrics."""

    y_true = np.array(y_true).ravel()  # Flatten to 1D
    y_pred = np.array(y_pred).ravel()

    print(f"True labels: {y_true}, predicted labels: {y_pred}")

    accuracy = accuracy_score(y_true, y_pred)

    print(f"Overall Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    class_names = [str(c) for c in class_names]
    class_report = classification_report(
        y_true, y_pred, labels=list(range(8)), target_names=class_names
    )
    print(class_report)

    labels = sorted(set(y_true) | set(y_pred))  # get actual labels present
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    label_to_class = {int(float(name)): name for name in class_names}

    cm_df = pd.DataFrame(
        cm,
        index=[label_to_class.get(int(i), f"Unknown_{i}") for i in labels],
        columns=[label_to_class.get(int(i), f"Unknown_{i}") for i in labels],
    )
    print("\nConfusion Matrix:")
    print(cm_df)

    wandb.log(
        {
            "classification_report": class_report,
            "confusion_matrix": cm_df,
        }
    )

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"{label_to_class.get(int(i), f'Unknown_{i}')}: {acc:.3f}")

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


def prepare_regression_eval(data, group_cols, objective):
    import numpy as np
    import pandas as pd

    # --- Step 0: clean group_cols ---
    group_cols = list(group_cols)
    if "label" in group_cols:
        group_cols.remove("label")
    if "run" not in data.columns and "run" in group_cols:
        group_cols.remove("run")
    if "sampler" in group_cols:
        group_cols.remove("sampler")
    if (
        "hidden" in group_cols
        and "hidden" not in data.columns
        and "hidden_per_layer" in data.columns
    ):
        group_cols = ["hidden_per_layer" if c == "hidden" else c for c in group_cols]

    keep_cols = group_cols + ["sampler", "ranked_labels", "ranked_values"]

    # --- Step 1: true labels ---
    if objective == "test_acc":
        best_idx = data.groupby(group_cols)[objective].idxmax()
    else:
        best_idx = data.groupby(group_cols)[objective].idxmin()

    best_samplers = data.loc[best_idx, keep_cols].rename(columns={"sampler": "label"})

    # --- Step 2: predicted labels ---
    if objective == "test_acc":
        pred_idx = data.groupby(group_cols)["y_pred"].idxmax()
    else:
        pred_idx = data.groupby(group_cols)["y_pred"].idxmin()

    pred_samplers = data.loc[pred_idx, keep_cols].rename(
        columns={"sampler": "pred_label"}
    )

    # --- Step 3: merge true and predicted ---
    labels = pd.merge(
        best_samplers, pred_samplers, on=group_cols, how="left", suffixes=("", "_pred")
    )
    labels.drop(["ranked_labels_pred", "ranked_values_pred"], axis=1, inplace=True)

    ranked_labels = labels["ranked_labels"]
    ranked_values = labels["ranked_values"]

    # --- Step 4: prepare sampler columns ---
    ALL_SAMPLERS = sorted(data["sampler"].unique())

    data2 = data.drop_duplicates(subset=group_cols + ["sampler"])
    pivot = (
        data2.set_index(group_cols + ["sampler"])["y_pred"]
        .unstack("sampler")
        .reindex(columns=ALL_SAMPLERS)
        .reset_index()
    )
    pivot["sampler_values"] = pivot[ALL_SAMPLERS].apply(list, axis=1)
    pivot["sampler_list"] = [ALL_SAMPLERS] * len(pivot)

    all_values = np.concatenate(pivot["sampler_values"].values)  # flatten all lists
    if objective == "test_acc":
        # For test_acc (higher is better), fill missing with a very low value
        min_value = np.nanmin(all_values)
        fill_val = (
            min_value - abs(min_value) - 1
        )  # Ensure it's worse than the worst actual value
    elif objective in ("time_per_epoch", "used_memory"):
        # For objectives where lower is better, fill missing with a very high value
        max_value = np.nanmax(all_values)
        fill_val = (
            max_value + abs(max_value) + 1
        )  # Ensure it's worse than the worst actual value

    print(f"fill_val: {fill_val}")

    pivot[ALL_SAMPLERS] = pivot[ALL_SAMPLERS].fillna(fill_val)

    pivot["sampler_values"] = pivot[ALL_SAMPLERS].apply(list, axis=1)

    sampler_lists = pivot[group_cols + ["sampler_values", "sampler_list"]].copy()
    labels = labels.merge(sampler_lists, on=group_cols, how="left")

    def make_proba_from_sampler_values(row, all_samplers, objective):
        """
        Create probability distribution from sampler values based on ranks.
        - For 'time_per_epoch': smallest value = rank 1 (best)
        - For 'test_accuracy': largest value = rank 1 (best)
        """
        ascending = objective in ("time_per_epoch", "used_memory")

        # Get sampler values from the 'sampler_values' column
        sampler_values_array = row["sampler_values"]  # This should be an array

        sampler_values = {
            sampler: sampler_values_array[ALL_SAMPLERS.index(sampler)]
            for sampler in all_samplers
        }

        # Sort and assign ranks
        sorted_samplers = sorted(
            sampler_values.items(), key=lambda x: x[1], reverse=(not ascending)
        )

        ranks = {sampler: rank + 1 for rank, (sampler, _) in enumerate(sorted_samplers)}

        # Create probability from ranks (in the order of all_samplers)
        proba = np.array([1.0 / ranks[sampler] for sampler in all_samplers])
        proba = proba / proba.sum()

        return proba

    labels["sampler_proba"] = labels.apply(
        lambda row: make_proba_from_sampler_values(row, ALL_SAMPLERS, objective), axis=1
    )

    labels["sampler_proba"] = labels.apply(
        lambda row: make_proba_from_sampler_values(row, ALL_SAMPLERS, objective), axis=1
    )

    proba_mat = np.vstack(labels["sampler_proba"].values)

    for i, row in labels.iterrows():
        pred = row["pred_label"]
        am = ALL_SAMPLERS[np.argmax(proba_mat[i])]
        if pred != am:
            print(f"Mismatch at {i}:", pred, am)
            print(labels["sampler_values"][i])
            print(pivot["sampler_values"][i])

    # --- Step 6: return values ---
    return (
        labels["label"],  # true labels
        labels["pred_label"],  # predicted labels
        ranked_labels,  # ranked labels (original)
        ranked_values,  # ranked values (original)
        np.vstack(labels["sampler_proba"].values),
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


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        scaler = MinMaxScaler()
        return scaler.fit_transform(scores)
    else:
        # Return None instead of raising
        return None


def get_probabilities_2(model, X):
    """
    Universal probability extractor for any classifier—
    handles predict_proba, decision_function, and fallbacks.
    """
    # Models with calibrated probabilities.
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except:
            pass

    # Margin-based models converted with softmax.
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)

    # Convert binary margins to two probability columns.
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T

        return softmax(scores)

    # Fall back to hard predictions.
    preds = model.predict(X)
    classes = model.classes_
    proba = np.zeros((len(preds), len(classes)))
    for i, p in enumerate(preds):
        proba[i, np.where(classes == p)[0]] = 1.0
    return proba


def create_label_mapping(labels):
    """
    labels: list or array of original class labels (may have missing values)
    Returns:
        - original_to_dense mapping
        - dense_to_original mapping
        - labels_dense (labels converted to dense form)
    """
    unique_labels = sorted(set(labels))

    # Create mapping from original labels -> dense (0..n-1)
    original_to_dense = {orig: i for i, orig in enumerate(unique_labels)}

    # Create reverse mapping
    dense_to_original = {v: k for k, v in original_to_dense.items()}

    # Convert labels
    labels_dense = [original_to_dense[l] for l in labels]

    return original_to_dense, dense_to_original, labels_dense


def is_perfect_ascending(nums):
    return nums == list(range(1, len(nums) + 1))


def topk_accuracy(y_true, y_prob, k=5, labels=None):

    # If there are fewer than k classes, return 1
    if y_prob.shape[1] < k:
        return 1

    # Compute top-k class indices for each row
    topk = np.argpartition(-y_prob, kth=k - 1, axis=1)[:, :k]

    # Determine if true label is inside top-k predictions
    matches = np.array([y_true[i] in topk[i] for i in range(len(y_true))])

    top_k_indices = np.argsort(y_prob, axis=1)[:, -k:][:, ::-1]
    hits = np.array([true in preds for true, preds in zip(y_true, top_k_indices)])

    # Return mean accuracy
    return np.mean(hits)


def topk_accuracy_2(y_true, y_prob, k=5, labels=None):
    print(f"y_true: {y_true}, y_prob: {y_prob}")
    # If y_prob has fewer than k classes, return 1
    if y_prob.shape[1] < k:
        return 1
    topk = np.argpartition(-y_prob, kth=k - 1, axis=1)[:, :k]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])


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
    use_group_cv=False,
    remove_dup=False,
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

    difficulty_analysis = analyze_prediction_difficulty(data, objective)

    # Preprocess data and construct labels.
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

    # Use original approach
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
    if objective == "test_acc":
        fill_value = 0
    elif objective in ("time_per_epoch", "used_memory"):
        fill_value = 100000000000

    test_data["ranked_values"] = test_data["ranked_values"].apply(
        lambda x: pad_with_value(x, target_len=8, fill_value=fill_value)
    )

    ranked_labels = test_data["ranked_labels"]
    ranked_values = test_data["ranked_values"]

    print("Number of unique datasets in training:", train_data["dataset"].nunique())
    print("Number of samples per dataset:")
    print(train_data["dataset"].value_counts())

    print(f"\nData sizes:")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    wandb.log({"train_size": len(train_data), "test_size": len(test_data)})

    exclude_cols = [
        objective,
        target_col,
        "dataset",
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
    print(f"Groups: {groups}")

    if remove_dup:
        tmp_X_train = X_train.copy()
        tmp_X_train["dataset"] = train_data["dataset"]
        X_train, y_train, _, _, groups = remove_duplicates(
            tmp_X_train, y_train, target_col, runs
        )
        tmp_X_test = X_test.copy()
        tmp_X_test["ranked_values"] = ranked_values
        tmp_X_test["ranked_labels"] = ranked_labels
        X_test, y_test, ranked_values, ranked_labels, _ = remove_duplicates(
            tmp_X_test, y_test, target_col, runs
        )

    print(f"Groups: {groups}")

    print(f"\nData sizes:")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    wandb.log({"train_size": len(X_train), "test_size": len(X_test)})

    X_test_fanout_col = X_test["fanout"]
    X_test_layer_col = X_test["layer"]
    X_test_epochs_col = X_test["epochs"]

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

    # numeric mapping for dataset
    ds_map = {d: i for i, d in enumerate(train_data["dataset"].unique())}
    ds_ids = train_data["dataset"].map(ds_map).values
    corrs = {
        col: np.corrcoef(train_data[col].astype(float).fillna(0), ds_ids)[0, 1]
        for col in numeric_features
    }
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("Top features correlated with dataset id (abs corr):", sorted_corrs)

    print(f"y train unique: {y_train.unique()}")
    if use_regression_approach:
        print(f"X_train_scaled columns: {X_train_scaled.columns}")
        best_models = train_regression_models(
            X_train_scaled, y_train, use_group_cv=use_group_cv, groups=groups
        )
    else:
        best_models = train_classification_models(
            X_train_scaled, y_train, use_group_cv=use_group_cv, groups=groups
        )

    if use_regression_approach:
        ensemble = create_ensemble_regressor(best_models, X_train_scaled, y_train)
    else:
        ensemble = create_ensemble_classifier(best_models, X_train_scaled, y_train)
    best_models["Ensemble"] = ensemble

    results = {}
    results_top_1 = {}
    results_top_2 = {}
    results_top_3 = {}
    results_top_4 = {}
    results_top_5 = {}
    results_top_6 = {}
    results_top_7 = {}

    relative_utility_ratios = {}
    relative_utility_losses = {}
    relative_penalized_losses = {}
    relative_log_losses = {}
    rank_based_scores = {}

    train_features = list(X_train_scaled.columns)
    for name, model in best_models.items():

        class_names = list(np.unique(y_train))
        print(f"Evaluate {name}")

        print(f"X_test_scaled columns: {X_test_scaled.columns}")
        X_test_aligned = X_test_scaled[train_features]

        print(f"X_test_aligned columns: {X_test_aligned.columns}")

        # Now predict
        y_pred = model.predict(X_test_aligned)
        y_pred_curr = y_pred.copy()

        top_1_accuracy = None
        top_2_accuracy = None
        top_3_accuracy = None
        top_4_accuracy = None
        top_5_accuracy = None
        top_6_accuracy = None
        top_7_accuracy = None

        if use_regression_approach:
            # Apply the mapping
            X_test_scaled["dataset"] = test_dataset
            X_test_scaled["fanout"] = X_test_fanout_col
            X_test_scaled["layer"] = X_test_layer_col
            X_test_scaled["epochs"] = X_test_epochs_col
            X_test_scaled["sampler"] = X_test["sampler"]
            X_test_scaled[objective] = y_test
            X_test_scaled["y_pred"] = y_pred_curr
            X_test_scaled["ranked_labels"] = ranked_labels
            X_test_scaled["ranked_values"] = ranked_values

            y_true, y_pred_curr, ranked_labels_, ranked_values_, y_proba = (
                prepare_regression_eval(X_test_scaled, group_cols, objective)
            )

            all_labels = list(range(y_proba.shape[1]))

            acc_direct = accuracy_score(y_true.astype(int), y_pred_curr.astype(int))

            accuracy = evaluate_with_detailed_metrics(
                y_true.astype(int),
                y_pred_curr.astype(int),
                all_labels,
            )

            class_names = list(np.unique(y_true))
            delete_idx = [label for label in range(y_proba.shape[1]) if label not in class_names]
            y_proba = np.delete(y_proba, delete_idx, axis=1)

            original_to_dense, _, labels_dense = create_label_mapping(class_names)
            y_true_mapped = np.asarray(
                [original_to_dense[label] for label in np.asarray(y_true, dtype=int)]
            )
            top_1_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=1, labels=labels_dense
            )
            top_2_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=2, labels=labels_dense
            )
            top_3_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=3, labels=labels_dense
            )
            top_4_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=4, labels=labels_dense
            )
            top_5_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=5, labels=labels_dense
            )
            top_6_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=6, labels=labels_dense
            )
            top_7_accuracy = topk_accuracy(
                y_true_mapped.astype(int), y_proba, k=7, labels=labels_dense
            )
            print(f"Direct accuracy: {acc_direct}, Top-1 accuracy: {top_1_accuracy}")
            print(f"Top 3 accuracy score: {top_3_accuracy}")

            (
                relative_utility_ratio,
                relative_utility_loss_score,
                relative_penalized_loss_score,
                relative_log_loss_utility_score,
                rank_based_score,
            ) = evaluate_relative_score(
                y_true.astype(int),
                y_pred_curr.astype(int),
                ranked_labels_,
                ranked_values_,
                objective,
            )

        else:
            y_proba = get_probabilities(model, X_test_aligned)

            map = False
            if len(class_names) != 8 and not is_perfect_ascending(class_names):
                map = True

                original_to_dense, dense_to_original, labels_dense = (
                    create_label_mapping(class_names)
                )

                tmp = y_pred.copy()
                if isinstance(tmp, np.ndarray):
                    tmp = tmp.reshape(-1)  # make it 1D
                    tmp = tmp.astype(int)
                y_pred_mapped = [original_to_dense[p] for p in tmp]
                class_names = labels_dense

                accuracy = evaluate_with_detailed_metrics(
                    y_test, y_pred_mapped, class_names
                )
            else:
                accuracy = evaluate_with_detailed_metrics(y_test, y_pred, class_names)
            print(f"y_proba: {y_proba}")
            if isinstance(y_proba, np.ndarray) and y_proba.size > 0:
                y_proba = scipy.special.softmax(y_proba, axis=1).copy()

                original_labels = list(np.unique(y_test))
                original_to_dense, _, labels_dense = create_label_mapping(original_labels)
                y_test_mapped = pd.Series(
                    [original_to_dense[label] for label in y_test.astype(int)],
                    index=y_test.index,
                    name=y_test.name,
                )
                delete_idx = [
                    label
                    for label in range(y_proba.shape[1])
                    if label not in original_labels
                ]
                y_proba = np.delete(y_proba, delete_idx, axis=1)

                top_1_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=1, labels=labels_dense
                )
                top_2_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=2, labels=labels_dense
                )
                top_3_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=3, labels=labels_dense
                )
                top_4_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=4, labels=labels_dense
                )
                top_5_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=5, labels=labels_dense
                )
                top_6_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=6, labels=labels_dense
                )
                top_7_accuracy = topk_accuracy(
                    y_test_mapped.astype(int), y_proba, k=7, labels=labels_dense
                )

            (
                relative_utility_ratio,
                relative_utility_loss_score,
                relative_penalized_loss_score,
                relative_log_loss_utility_score,
                rank_based_score,
            ) = evaluate_relative_score(
                y_test, y_pred, ranked_labels, ranked_values, objective
            )
        results[f"{name}"] = accuracy
        results_top_1[f"top_1_accuracy_{name}"] = top_1_accuracy
        results_top_2[f"top_2_accuracy_{name}"] = top_2_accuracy
        results_top_3[f"top_3_accuracy_{name}"] = top_3_accuracy
        results_top_4[f"top_4_accuracy_{name}"] = top_4_accuracy
        results_top_5[f"top_5_accuracy_{name}"] = top_5_accuracy
        results_top_6[f"top_6_accuracy_{name}"] = top_6_accuracy
        results_top_7[f"top_7_accuracy_{name}"] = top_7_accuracy

        relative_utility_ratios[f"relative_utility_ratio_{name}"] = (
            relative_utility_ratio
        )
        relative_utility_losses[f"relative_utility_loss_{name}"] = (
            relative_utility_loss_score
        )
        relative_penalized_losses[f"relative_penalized_loss_{name}"] = (
            relative_penalized_loss_score
        )
        relative_log_losses[f"relative_log_loss_{name}"] = (
            relative_log_loss_utility_score
        )
        rank_based_scores[f"rank_based_score_{name}"] = rank_based_score
        print("-" * 40)

    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Test_Accuracy"])
    results_df = results_df.sort_values("Test_Accuracy", ascending=False)
    print(results_df)
    best_models = results_df.loc[
        results_df["Test_Accuracy"] == results_df["Test_Accuracy"].max(), "Model"
    ].tolist()

    best_model = results_df.iloc[0]["Model"]
    wandb.log({"best_model": best_model, "best_models": best_models})

    wandb.log(relative_utility_ratios)
    wandb.log(relative_utility_losses)
    wandb.log(relative_penalized_losses)
    wandb.log(relative_log_losses)
    wandb.log(rank_based_scores)
    wandb.log(results)
    wandb.log(results_top_1)
    wandb.log(results_top_2)
    wandb.log(results_top_3)
    wandb.log(results_top_4)
    wandb.log(results_top_5)
    wandb.log(results_top_6)
    wandb.log(results_top_7)
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
    df_scores = df_scores.fillna(-1)

    # Optional: sort rows by a specific score
    df_scores = df_scores.sort_values("Relative Utility Score", ascending=False)

    print("Scores summary:")
    print(df_scores)

    # Convert to DataFrame
    results_top_3_df = pd.DataFrame(
        list(results_top_3.items()), columns=["Model", "Top 3 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_3_df = results_top_3_df.sort_values(
        by="Top 3 Accuracy", ascending=False, na_position="last"
    )

    print(results_top_3_df)

    # Convert to DataFrame
    results_top_1_df = pd.DataFrame(
        list(results_top_1.items()), columns=["Model", "Top 1 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_1_df = results_top_1_df.sort_values(
        by="Top 1 Accuracy", ascending=False, na_position="last"
    )

    print(results_top_1_df)

    # Convert to DataFrame
    results_top_2_df = pd.DataFrame(
        list(results_top_2.items()), columns=["Model", "Top 2 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_2_df = results_top_2_df.sort_values(
        by="Top 2 Accuracy", ascending=False, na_position="last"
    )

    # Convert to DataFrame
    results_top_4_df = pd.DataFrame(
        list(results_top_4.items()), columns=["Model", "Top 4 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_4_df = results_top_4_df.sort_values(
        by="Top 4 Accuracy", ascending=False, na_position="last"
    )

    # Convert to DataFrame
    results_top_5_df = pd.DataFrame(
        list(results_top_5.items()), columns=["Model", "Top 5 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_5_df = results_top_5_df.sort_values(
        by="Top 5 Accuracy", ascending=False, na_position="last"
    )

    # Convert to DataFrame
    results_top_6_df = pd.DataFrame(
        list(results_top_6.items()), columns=["Model", "Top 6 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_6_df = results_top_6_df.sort_values(
        by="Top 6 Accuracy", ascending=False, na_position="last"
    )

    # Convert to DataFrame
    results_top_7_df = pd.DataFrame(
        list(results_top_7.items()), columns=["Model", "Top 7 Accuracy"]
    )
    # Sort descending, putting None/NaN at the bottom
    results_top_7_df = results_top_7_df.sort_values(
        by="Top 7 Accuracy", ascending=False, na_position="last"
    )

    wandb.log(
        {
            "final_results": results_df,
            "final_scores": df_scores,
        }
    )
    wandb.log(relative_utility_ratios)
    wandb.log(relative_utility_losses)
    wandb.log(relative_penalized_losses)
    wandb.log(relative_log_losses)
    wandb.log(rank_based_scores)

    return {
        "models": best_models,
        "results": results_df,
        "scores": df_scores,
        "difficulty_analysis": difficulty_analysis,
        "top_1_accuracy": results_top_1_df,
        "top_2_accuracy": results_top_2_df,
        "top_3_accuracy": results_top_3_df,
        "top_4_accuracy": results_top_4_df,
        "top_5_accuracy": results_top_5_df,
        "top_6_accuracy": results_top_6_df,
        "top_7_accuracy": results_top_7_df,
    }


def analyze_feature_importance_and_leakage(models, feature_names):
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


@hydra.main(version_base=None, config_path="./", config_name="config")
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
        "use_group_cv": cfg.use_group_cv,
        "remove_duplicates": cfg.remove_dup,
    }

    wandb.init(
        project="sampler_selector",
        name=f"sampler_selector_{cfg.test_dataset}",
        config=config,
        mode="offline",
        reinit="finish_previous",
    )

    results = load_results(
        file="full_results_bias.pkl",
    )

    preprocessed = prepare_data(results, objective)
    print(preprocessed)
    data = build_data_with_graph_characteristics(
        preprocessed, GRAPH_CHARACTERISTICS, objective, merge_runs
    )

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
        use_group_cv=cfg.use_group_cv,
        remove_dup=cfg.remove_dup,
    )

    # Check the best accuracy
    print(f"Best accuracy: {results['results'].iloc[0]['Test_Accuracy']:.3f}")
    print(
        f"Best utility score: {results['scores'].iloc[0]['Relative Utility Score']:.3f}"
    )
    print(
        f"Best Top 3 accuracy: {results['top_3_accuracy'].iloc[0]['Top 3 Accuracy']:.3f}"
    )

    wandb.log(
        {
            "best_accuracy": results["results"].iloc[0]["Test_Accuracy"],
            "best_score": results["scores"].iloc[0]["Relative Utility Score"],
            "best_utility_loss": results["scores"].iloc[0][
                "Relative Utility Loss Score"
            ],
            "best_penalized_loss": results["scores"].iloc[0][
                "Relative Penalized Loss Score"
            ],
            "best_log_loss": results["scores"].iloc[0]["Relative Log Loss Score"],
            "best_rank_based_score": results["scores"].iloc[0]["Rank Based Score"],
            "best_top_1_accuracy": results["top_1_accuracy"].iloc[0]["Top 1 Accuracy"],
            "best_top_2_accuracy": results["top_2_accuracy"].iloc[0]["Top 2 Accuracy"],
            "best_top_3_accuracy": results["top_3_accuracy"].iloc[0]["Top 3 Accuracy"],
            "best_top_4_accuracy": results["top_4_accuracy"].iloc[0]["Top 4 Accuracy"],
            "best_top_5_accuracy": results["top_5_accuracy"].iloc[0]["Top 5 Accuracy"],
            "best_top_6_accuracy": results["top_6_accuracy"].iloc[0]["Top 6 Accuracy"],
            "best_top_7_accuracy": results["top_7_accuracy"].iloc[0]["Top 7 Accuracy"],
        }
    )

    run_dir = os.path.dirname(wandb.run.dir) if wandb.run.dir else None
    wandb.finish()
    if run_dir and os.path.exists(run_dir):
        shutil.rmtree(run_dir)



if __name__ == "__main__":
    main()
