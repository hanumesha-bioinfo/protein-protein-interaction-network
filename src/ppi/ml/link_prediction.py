"""Research-oriented baseline for protein interaction link prediction.

Unobserved protein pairs are sampled as negatives. They are not proven
biological non-interactions; results are exploratory and dataset-specific.
"""

from itertools import combinations
from pathlib import Path
import json

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EDGE_FILE = ROOT / "string_edges.csv"
OUT = ROOT / "results" / "ml"
RANDOM_STATE = 42

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
    "degree_sum",
]


def load_graph(path: Path) -> nx.Graph:
    df = pd.read_csv(path)
    required = {"preferredName_A", "preferredName_B", "score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df = df[list(required)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna()

    df["preferredName_A"] = df["preferredName_A"].astype(str).str.strip()
    df["preferredName_B"] = df["preferredName_B"].astype(str).str.strip()

    df = df[
        (df["preferredName_A"] != "")
        & (df["preferredName_B"] != "")
        & (df["preferredName_A"] != df["preferredName_B"])
        & (df["score"] > 0)
        & (df["score"] <= 1)
    ]

    graph = nx.Graph()
    for row in df.itertuples(index=False):
        a, b = sorted((row.preferredName_A, row.preferredName_B))
        if graph.has_edge(a, b):
            graph[a][b]["score"] = max(graph[a][b]["score"], float(row.score))
        else:
            graph.add_edge(a, b, score=float(row.score))

    return graph


def pair_features(graph: nx.Graph, u: str, v: str) -> list[float]:
    """Calculate features with the queried edge masked, if present."""
    g = graph.copy()

    if g.has_edge(u, v):
        g.remove_edge(u, v)

    neighbors_u = set(g.neighbors(u))
    neighbors_v = set(g.neighbors(v))
    common = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v

    adamic_adar = sum(
        1.0 / np.log(g.degree(w))
        for w in common
        if g.degree(w) > 1
    )
    resource_allocation = sum(
        1.0 / g.degree(w)
        for w in common
        if g.degree(w) > 0
    )

    du, dv = g.degree(u), g.degree(v)

    return [
        float(len(common)),
        len(common) / len(union) if union else 0.0,
        float(adamic_adar),
        float(resource_allocation),
        float(du * dv),
        float(du + dv),
    ]


def build_feature_table(
    graph: nx.Graph,
    pairs: list[tuple[str, str]],
    labels: list[int],
) -> pd.DataFrame:
    rows = []

    for (u, v), label in zip(pairs, labels):
        values = pair_features(graph, u, v)
        rows.append({
            "Protein_A": u,
            "Protein_B": v,
            **dict(zip(FEATURES, values)),
            "label": label,
        })

    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graph = load_graph(EDGE_FILE)

    nodes = sorted(graph.nodes())
    observed_edges = {
        tuple(sorted((u, v))) for u, v in graph.edges()
    }

    if len(nodes) < 5 or len(observed_edges) < 8:
        raise ValueError(
            "This baseline needs at least 5 proteins and 8 observed edges. "
            "Use a larger interaction network for meaningful evaluation."
        )

    # Keep the full observed edge set only for defining held-out examples.
    positive_edges = sorted(observed_edges)
    all_possible = set(combinations(nodes, 2))
    non_edges = sorted(all_possible - observed_edges)

    n_negative = min(len(positive_edges), len(non_edges))
    if n_negative < 4:
        raise ValueError(
            "Too few unobserved pairs for a train/test experiment. "
            "Use a larger, less dense network."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    selected_indices = rng.choice(
        len(non_edges), size=n_negative, replace=False
    )
    negative_pairs = [non_edges[i] for i in selected_indices]

    pos_train, pos_test = train_test_split(
        positive_edges,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )
    neg_train, neg_test = train_test_split(
        negative_pairs,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    # Only training positive edges are visible in the graph used for features.
    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(
        (u, v, {"score": graph[u][v]["score"]})
        for u, v in pos_train
    )

    train_pairs = pos_train + neg_train
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)

    test_pairs = pos_test + neg_test
    test_labels = [1] * len(pos_test) + [0] * len(neg_test)

    train_df = build_feature_table(
        train_graph, train_pairs, train_labels
    )
    test_df = build_feature_table(
        train_graph, test_pairs, test_labels
    )

    # Save the exact experiment examples and their labels for reproducibility.
    train_df.to_csv(OUT / "training_examples.csv", index=False)
    test_df.to_csv(OUT / "test_examples.csv", index=False)

    X_train = train_df[FEATURES].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df[FEATURES].to_numpy()
    y_test = test_df["label"].to_numpy()

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    reports = []
    predictions = test_df[["Protein_A", "Protein_B", "label"]].copy()

    for name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)

        reports.append({
            "model": name,
            "test_examples": len(y_test),
            "precision": precision_score(
                y_test, predicted, zero_division=0
            ),
            "recall": recall_score(
                y_test, predicted, zero_division=0
            ),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "average_precision": average_precision_score(
                y_test, probability
            ),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
                y_test, predicted, labels=[0, 1]
            ).ravel().tolist(),
        })

        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_predicted_label"] = predicted

        joblib.dump(model, OUT / f"{name}.joblib")

        if name == "random_forest":
            importance = pd.DataFrame({
                "feature": FEATURES,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance.to_csv(
                OUT / "random_forest_feature_importance.csv",
                index=False,
            )

    pd.DataFrame(reports).to_csv(
        OUT / "model_metrics.csv", index=False
    )
    predictions.to_csv(
        OUT / "test_predictions.csv", index=False
    )

    metadata = {
        "task": "observed_edge_vs_sampled_unobserved_pair",
        "nodes": len(nodes),
        "observed_edges": len(positive_edges),
        "sampled_unobserved_pairs": n_negative,
        "training_positive_edges": len(pos_train),
        "test_positive_edges": len(pos_test),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "warning": (
            "Unobserved pairs are not confirmed biological negatives. "
            "Metrics are exploratory and may be unstable for small networks."
        ),
    }
    (OUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nExperiment metadata")
    print(json.dumps(metadata, indent=2))
    print("\nModel evaluation")
    print(pd.DataFrame(reports).to_string(index=False))
    print(f"\nResults saved in: {OUT}")


if __name__ == "__main__":
    main()
