import networkx as nx
import pandas as pd
import pytest

from src.ppi.network.analyze_network import analyze_network, load_network


def test_load_network_removes_duplicates_and_self_loops(tmp_path):
    path = tmp_path / "edges.csv"

    pd.DataFrame({
        "preferredName_A": ["TP53", "TP53", "EGFR", "AKT1"],
        "preferredName_B": ["EGFR", "EGFR", "EGFR", "TP53"],
        "score": [0.7, 0.9, 0.8, 0.6],
    }).to_csv(path, index=False)

    graph = load_network(path)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph["TP53"]["EGFR"]["score"] == pytest.approx(0.9)
    assert not graph.has_edge("EGFR", "EGFR")


def test_load_network_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"protein": ["TP53"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_network(path)


def test_analyze_network_returns_expected_metrics():
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
    ])

    metrics, summary, communities = analyze_network(graph)

    assert len(metrics) == 3
    assert set(metrics["Protein"]) == {"A", "B", "C"}
    assert (metrics["Degree"] == 2).all()

    density = summary.loc[
        summary["Measure"] == "Network density", "Value"
    ].iloc[0]
    assert density == pytest.approx(1.0)

    component_count = summary.loc[
        summary["Measure"] == "Number of connected components", "Value"
    ].iloc[0]
    assert component_count == 1

    assert set(communities["Protein"]) == {"A", "B", "C"}
