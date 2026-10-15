import networkx as nx
import pytest

from src.ppi.ml.link_prediction import FEATURES, pair_features


def test_pair_features_for_known_graph():
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A", "D"),
    ])

    values = pair_features(graph, "B", "D")
    features = dict(zip(FEATURES, values))

    # B and D share exactly one neighbour: A.
    assert features["common_neighbors"] == 1.0
    assert features["jaccard"] == pytest.approx(0.5)
    assert features["adamic_adar"] == pytest.approx(1 / __import__("math").log(3))
    assert features["resource_allocation"] == pytest.approx(1 / 3)
    assert features["preferential_attachment"] == 2.0
    assert features["degree_sum"] == 3.0


def test_existing_edge_is_masked_before_feature_calculation():
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ])

    # The queried B-C edge must be removed before features are calculated.
    values = pair_features(graph, "B", "C")
    features = dict(zip(FEATURES, values))

    assert features["common_neighbors"] == 1.0
    assert features["jaccard"] == pytest.approx(1.0)
    assert features["preferential_attachment"] == 1.0
    assert features["degree_sum"] == 2.0
