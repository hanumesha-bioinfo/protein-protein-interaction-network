"""Interactive dashboard for the human PPI network research project."""

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
EDGE_FILE = ROOT / "string_edges.csv"
TABLES = ROOT / "results" / "tables"
ML_DIR = ROOT / "results" / "ml"


st.set_page_config(
    page_title="Human PPI Explorer",
    page_icon="🧬",
    layout="wide",
)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_edges(path_string):
    df = pd.read_csv(path_string)
    required = {"preferredName_A", "preferredName_B", "score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Edge file needs columns: {sorted(required)}")

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

    # Merge duplicate undirected edges by keeping the maximum score.
    df["node_a"] = df[
        ["preferredName_A", "preferredName_B"]
    ].min(axis=1)
    df["node_b"] = df[
        ["preferredName_A", "preferredName_B"]
    ].max(axis=1)

    return (
        df.groupby(["node_a", "node_b"], as_index=False)["score"]
        .max()
        .rename(columns={"node_a": "Protein_A", "node_b": "Protein_B"})
    )


def make_network_figure(edges, selected_protein=None):
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.Protein_A, row.Protein_B, score=row.score)

    if graph.number_of_nodes() == 0:
        return None

    positions = nx.spring_layout(graph, seed=42, weight="score")

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    figure = go.Figure()

    figure.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "#9AA4B2"},
        hoverinfo="skip",
        name="Interactions",
    ))

    degree = dict(graph.degree())
    node_x, node_y, node_names, node_sizes, node_colors, hover_text = (
        [], [], [], [], [], []
    )

    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_names.append(node)
        node_sizes.append(12 + 4 * degree[node])
        node_colors.append(
            "#E45756" if node == selected_protein else "#2878B5"
        )
        hover_text.append(
            f"Protein: {node}<br>"
            f"Degree in filtered network: {degree[node]}"
        )

    figure.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_names,
        textposition="top center",
        textfont={"size": 10},
        hovertext=hover_text,
        hoverinfo="text",
        marker={
            "size": node_sizes,
            "color": node_colors,
            "line": {"width": 1, "color": "white"},
        },
        name="Proteins",
    ))

    figure.update_layout(
        title="Protein interaction network",
        height=650,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
    )
    return figure


st.title("🧬 Human Protein–Protein Interaction Explorer")
st.caption(
    "Explore STRING-derived interactions, network centrality, "
    "communities and exploratory machine-learning results."
)

if not EDGE_FILE.exists():
    st.error(f"Interaction data not found: {EDGE_FILE}")
    st.stop()

try:
    all_edges = load_edges(str(EDGE_FILE))
except Exception as error:
    st.error(f"Could not load interaction data: {error}")
    st.stop()

if all_edges.empty:
    st.error("No valid interaction edges were found.")
    st.stop()

metrics = load_csv(str(TABLES / "protein_metrics.csv"))
summary = load_csv(str(TABLES / "network_summary_advanced.csv"))
communities = load_csv(str(TABLES / "protein_communities.csv"))
model_metrics = load_csv(str(ML_DIR / "model_metrics.csv"))
predictions = load_csv(str(ML_DIR / "test_predictions.csv"))
importance = load_csv(
    str(ML_DIR / "random_forest_feature_importance.csv")
)

st.sidebar.header("Network filters")
min_score = st.sidebar.slider(
    "Minimum STRING confidence score",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Filters the displayed edges; does not retrain your ML models.",
)

filtered_edges = all_edges[all_edges["score"] >= min_score].copy()
filtered_graph = nx.Graph()
for row in filtered_edges.itertuples(index=False):
    filtered_graph.add_edge(row.Protein_A, row.Protein_B, score=row.score)

if filtered_graph.number_of_nodes() == 0:
    st.warning("No edges meet this confidence threshold.")
    st.stop()

nodes = sorted(filtered_graph.nodes())
selected = st.sidebar.selectbox(
    "Highlight a protein",
    options=["None"] + nodes,
)
selected_protein = None if selected == "None" else selected

degree_filtered = dict(filtered_graph.degree())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Proteins displayed", filtered_graph.number_of_nodes())
c2.metric("Interactions displayed", filtered_graph.number_of_edges())
c3.metric("Network density", f"{nx.density(filtered_graph):.3f}")
c4.metric(
    "Connected components",
    nx.number_connected_components(filtered_graph),
)

tab_network, tab_proteins, tab_ml, tab_data = st.tabs([
    "🌐 Network explorer",
    "🧪 Protein analysis",
    "🤖 ML results",
    "📁 Data & downloads",
])

with tab_network:
    st.subheader("Explore interactions")
    st.write(
        "Node size reflects degree in the filtered network. "
        "The highlighted protein is shown in red."
    )
    fig = make_network_figure(filtered_edges, selected_protein)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download filtered interactions",
        data=filtered_edges.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ppi_edges.csv",
        mime="text/csv",
    )

with tab_proteins:
    st.subheader("Protein centrality rankings")

    if not metrics.empty and "Protein" in metrics.columns:
        view = metrics.copy()
        if selected_protein:
            view = view[view["Protein"] == selected_protein]
        st.dataframe(view, use_container_width=True)

        st.download_button(
            "Download protein metrics",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="protein_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the network analysis module to generate protein metrics.")

    st.subheader("Degree ranking in the filtered network")
    degree_df = pd.DataFrame(
        [{"Protein": p, "Degree": d} for p, d in degree_filtered.items()]
    ).sort_values("Degree", ascending=False)
    st.bar_chart(degree_df.head(15).set_index("Protein"))

    st.subheader("Community assignments")
    if not communities.empty:
        st.dataframe(communities, use_container_width=True)
    else:
        st.info("Run network analysis to generate community assignments.")

    if not summary.empty:
        st.subheader("Saved network analysis summary")
        st.dataframe(summary, use_container_width=True)

with tab_ml:
    st.subheader("Model evaluation")
    st.caption(
        "These metrics are from the saved experiment and do not change "
        "when you adjust the network confidence filter."
    )

    if not model_metrics.empty:
        st.dataframe(model_metrics, use_container_width=True)

        chart_columns = [
            column for column in ["model", "roc_auc", "average_precision", "f1"]
            if column in model_metrics.columns
        ]
        if len(chart_columns) > 1:
            st.bar_chart(
                model_metrics[chart_columns].set_index("model")
            )
    else:
        st.info(
            "No ML evaluation found. Run "
            "`python -m src.ppi.ml.link_prediction` first."
        )

    st.subheader("Random Forest feature importance")
    if not importance.empty:
        st.dataframe(importance, use_container_width=True)
        if {"feature", "importance"}.issubset(importance.columns):
            st.bar_chart(importance.set_index("feature"))
    else:
        st.info("Feature-importance results are not available yet.")

    st.subheader("Held-out pair predictions")
    st.warning(
        "These are experimental scores for sampled pairs from a small network. "
        "Unobserved pairs are not confirmed biological negatives, and these "
        "scores should not be interpreted as validated biological probabilities."
    )
    if not predictions.empty:
        st.dataframe(predictions, use_container_width=True)
        st.download_button(
            "Download test predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="test_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the ML module to generate test predictions.")

with tab_data:
    st.subheader("Dataset information")
    st.write(f"Source edge file: `{EDGE_FILE.name}`")
    st.write(f"Rows in validated, deduplicated edge table: {len(all_edges)}")
    st.write(f"Unique proteins in the edge table: {len(set(all_edges.Protein_A) | set(all_edges.Protein_B))}")

    st.subheader("Available reports")
    reports = [
        TABLES / "protein_metrics.csv",
        TABLES / "network_summary_advanced.csv",
        TABLES / "protein_communities.csv",
        ML_DIR / "model_metrics.csv",
        ML_DIR / "test_predictions.csv",
        ML_DIR / "experiment_metadata.json",
    ]

    for report in reports:
        if report.exists():
            st.write(f"✅ `{report.relative_to(ROOT)}`")
        else:
            st.write(f"— Not generated: `{report.relative_to(ROOT)}`")

    st.caption(
        "STRING interaction scores represent confidence in functional "
        "associations, not necessarily direct physical binding."
    )

st.divider()
st.caption(
    "Research prototype • Results depend on the selected proteins, "
    "STRING evidence, confidence threshold and evaluation design."
)
