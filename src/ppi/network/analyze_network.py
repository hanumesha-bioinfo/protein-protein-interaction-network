"""Network analysis for a STRING protein-protein interaction network."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EDGE_FILE = ROOT / "string_edges.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"


def load_network(path: Path) -> nx.Graph:
    """Load and validate STRING interaction data."""
    df = pd.read_csv(path)

    required = {
        "preferredName_A",
        "preferredName_B",
        "score",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[list(required)].copy()
    df = df.dropna(subset=list(required))

    df["preferredName_A"] = df["preferredName_A"].astype(str).str.strip()
    df["preferredName_B"] = df["preferredName_B"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    df = df[
        (df["preferredName_A"] != "")
        & (df["preferredName_B"] != "")
        & (df["preferredName_A"] != df["preferredName_B"])
        & (df["score"] > 0)
        & (df["score"] <= 1)
    ]

    if df.empty:
        raise ValueError("No valid interaction edges remain after cleaning.")

    # Treat interactions as undirected. Keep the strongest duplicate edge.
    df["node_a"] = df[
        ["preferredName_A", "preferredName_B"]
    ].min(axis=1)
    df["node_b"] = df[
        ["preferredName_A", "preferredName_B"]
    ].max(axis=1)

    df = (
        df.groupby(["node_a", "node_b"], as_index=False)["score"]
        .max()
    )

    graph = nx.Graph()

    for row in df.itertuples(index=False):
        graph.add_edge(
            row.node_a,
            row.node_b,
            score=float(row.score),
            distance=1.0 / float(row.score),
        )

    return graph


def analyze_network(graph: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate node-level metrics and network-level summaries."""
    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(
        graph, weight="distance", normalized=True
    )
    closeness = nx.closeness_centrality(graph, distance="distance")
    pagerank = nx.pagerank(graph, weight="score")

    clustering = nx.clustering(graph, weight="score")

    rows = []

    for node in graph.nodes():
        rows.append({
            "Protein": node,
            "Degree": degree[node],
            "Degree_Centrality": degree_centrality[node],
            "Betweenness_Centrality": betweenness[node],
            "Closeness_Centrality": closeness[node],
            "PageRank": pagerank[node],
            "Weighted_Clustering": clustering[node],
        })

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(
        ["Degree", "PageRank"], ascending=False
    ).reset_index(drop=True)

    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)

    communities = []

    if graph.number_of_edges() > 0:
        detected = nx.community.greedy_modularity_communities(
            graph, weight="score"
        )

        ordered = sorted(
            detected,
            key=lambda community: (-len(community), sorted(community)[0]),
        )

        for community_id, members in enumerate(ordered, start=1):
            for protein in sorted(members):
                communities.append({
                    "Protein": protein,
                    "Community": community_id,
                    "Community_Size": len(members),
                })

    community_df = pd.DataFrame(
        communities,
        columns=["Protein", "Community", "Community_Size"],
    )

    summary = pd.DataFrame([
        {"Measure": "Number of nodes", "Value": graph.number_of_nodes()},
        {"Measure": "Number of unique edges", "Value": graph.number_of_edges()},
        {
            "Measure": "Network density",
            "Value": nx.density(graph),
        },
        {
            "Measure": "Number of connected components",
            "Value": len(components),
        },
        {
            "Measure": "Largest component size",
            "Value": len(components[0]) if components else 0,
        },
        {
            "Measure": "Average degree",
            "Value": (
                sum(degree.values()) / graph.number_of_nodes()
                if graph.number_of_nodes() else 0
            ),
        },
        {
            "Measure": "Average clustering coefficient",
            "Value": nx.average_clustering(graph, weight="score"),
        },
        {
            "Measure": "Number of detected communities",
            "Value": len(community_df["Community"].unique())
            if not community_df.empty else 0,
        },
    ])

    return metrics, summary, community_df


def save_visualization(graph: nx.Graph, metrics: pd.DataFrame) -> None:
    """Save a network plot with node size proportional to degree."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    positions = nx.spring_layout(
        graph, seed=42, weight="score"
    )

    metric_lookup = metrics.set_index("Protein")["Degree"].to_dict()
    node_sizes = [
        250 + 100 * metric_lookup[node]
        for node in graph.nodes()
    ]
    edge_widths = [
        0.5 + 2.0 * graph[u][v]["score"]
        for u, v in graph.edges()
    ]

    fig, ax = plt.subplots(figsize=(13, 10))

    nx.draw_networkx_edges(
        graph, positions, width=edge_widths,
        alpha=0.35, ax=ax
    )
    nx.draw_networkx_nodes(
        graph, positions, node_size=node_sizes,
        alpha=0.9, ax=ax
    )
    nx.draw_networkx_labels(
        graph, positions, font_size=8, ax=ax
    )

    ax.set_title("Human Protein–Protein Interaction Network")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "ppi_network_advanced.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading interactions from: {EDGE_FILE}")
    graph = load_network(EDGE_FILE)

    if graph.number_of_nodes() < 2:
        raise ValueError("At least two connected proteins are required.")

    metrics, summary, communities = analyze_network(graph)

    metrics.to_csv(
        TABLE_DIR / "protein_metrics.csv", index=False
    )
    summary.to_csv(
        TABLE_DIR / "network_summary_advanced.csv", index=False
    )
    communities.to_csv(
        TABLE_DIR / "protein_communities.csv", index=False
    )

    save_visualization(graph, metrics)

    print("\nNETWORK SUMMARY")
    print(summary.to_string(index=False))

    print("\nTOP 10 HUB PROTEINS")
    print(metrics.head(10).to_string(index=False))

    print("\nCreated:")
    print(" results/tables/protein_metrics.csv")
    print(" results/tables/network_summary_advanced.csv")
    print(" results/tables/protein_communities.csv")
    print(" results/figures/ppi_network_advanced.png")


if __name__ == "__main__":
    main()
