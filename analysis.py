import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO


# ============================================================
# PROTEIN-PROTEIN INTERACTION NETWORK ANALYSIS
# STRING DATABASE + PYTHON + NETWORKX
# ============================================================


# ============================================================
# 1. SELECT 20 PROTEINS
# ============================================================

proteins = [
    "TP53",
    "EGFR",
    "AKT1",
    "MTOR",
    "MYC",
    "KRAS",
    "BRAF",
    "MAPK1",
    "MAPK3",
    "STAT3",
    "JAK2",
    "SRC",
    "PTEN",
    "CDK1",
    "CDK2",
    "CCND1",
    "RB1",
    "BRCA1",
    "BRCA2",
    "VEGFA"
]

# Homo sapiens
species = 9606

# STRING confidence threshold
# 400 = medium confidence
required_score = 400

caller_identity = "my_ppi_project"


# ============================================================
# 2. GET STRING IDs
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1: GETTING STRING IDs")
print("=" * 70)

mapping_params = {
    "identifiers": "\r".join(proteins),
    "species": species,
    "limit": 1,
    "echo_query": 1,
    "caller_identity": caller_identity
}

mapping_url = "https://string-db.org/api/tsv/get_string_ids"

response = requests.post(
    mapping_url,
    data=mapping_params
)

response.raise_for_status()

string_ids = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)


# ============================================================
# 3. DISPLAY STRING ID MAPPING
# ============================================================

print("\nSTRING ID MAPPING")
print("-" * 70)

mapping_columns = [
    "queryItem",
    "stringId",
    "preferredName",
    "annotation"
]

print(
    string_ids[mapping_columns].to_string(index=False)
)


# ============================================================
# 4. CHECK PROTEIN MAPPING
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2: CHECKING PROTEIN MAPPINGS")
print("=" * 70)

mapped_proteins = []

for protein in proteins:

    match = string_ids[
        string_ids["queryItem"].astype(str).str.upper()
        == protein.upper()
    ]

    if len(match) == 0:

        print(f"WARNING: {protein} was not mapped")

    else:

        row = match.iloc[0]

        print(
            f"{protein:8s} -> "
            f"{row['preferredName']:10s} "
            f"-> {row['stringId']}"
        )

        mapped_proteins.append(
            row["stringId"]
        )


# ============================================================
# 5. SAVE STRING MAPPING
# ============================================================

string_ids.to_csv(
    "string_id_mapping.csv",
    index=False
)

print("\nSaved:")
print("string_id_mapping.csv")


# ============================================================
# 6. CHECK WHETHER ALL 20 PROTEINS WERE MAPPED
# ============================================================

print("\n")
print("=" * 70)
print("MAPPING SUMMARY")
print("=" * 70)

print("Proteins requested :", len(proteins))
print("Proteins mapped    :", len(mapped_proteins))
print("Proteins missing   :", len(proteins) - len(mapped_proteins))


if len(mapped_proteins) != len(proteins):

    print("\nWARNING:")
    print("Not all proteins were mapped.")
    print("Check string_id_mapping.csv before continuing.")

    missing_proteins = []

    for protein in proteins:

        match = string_ids[
            string_ids["queryItem"].astype(str).str.upper()
            == protein.upper()
        ]

        if len(match) == 0:
            missing_proteins.append(protein)

    print("\nMissing proteins:")
    for protein in missing_proteins:
        print("-", protein)

else:

    print("\nAll 20 proteins were mapped.")


# ============================================================
# 7. GET STRING PPI NETWORK
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3: GETTING STRING PPI NETWORK")
print("=" * 70)


network_params = {
    "identifiers": "\r".join(mapped_proteins),
    "species": species,
    "required_score": required_score,
    "caller_identity": caller_identity
}

network_url = "https://string-db.org/api/tsv/network"

network_response = requests.post(
    network_url,
    data=network_params
)

network_response.raise_for_status()


# Convert STRING response into pandas DataFrame
network = pd.read_csv(
    StringIO(network_response.text),
    sep="\t"
)


# ============================================================
# 8. DISPLAY NETWORK INFORMATION
# ============================================================

print("\nSTRING network columns:")
print(network.columns.tolist())

print("\nNumber of STRING interactions retrieved:")
print(len(network))

print("\nFirst 5 interactions:")
print(network.head())


# ============================================================
# 9. CHECK REQUIRED NETWORK COLUMNS
# ============================================================

required_columns = [
    "stringId_A",
    "stringId_B",
    "preferredName_A",
    "preferredName_B",
    "score"
]

missing_columns = [
    column
    for column in required_columns
    if column not in network.columns
]

if missing_columns:

    print("\nERROR: Required columns are missing:")
    print(missing_columns)

    print("\nAvailable columns:")
    print(network.columns.tolist())

    raise SystemExit(
        "Network column structure is different from expected."
    )


# ============================================================
# 10. CREATE CLEAN EDGE TABLE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4: PREPROCESSING PPI EDGES")
print("=" * 70)


edges = network[
    [
        "stringId_A",
        "stringId_B",
        "preferredName_A",
        "preferredName_B",
        "score"
    ]
].copy()


# Remove missing values
edges = edges.dropna(
    subset=[
        "preferredName_A",
        "preferredName_B"
    ]
)


# Remove duplicate interactions
edges = edges.drop_duplicates()


# Remove self-interactions
edges = edges[
    edges["preferredName_A"]
    != edges["preferredName_B"]
]


print("Number of cleaned edges:", len(edges))


# ============================================================
# 11. SAVE STRING EDGES
# ============================================================

edges.to_csv(
    "string_edges.csv",
    index=False
)

print("\nSaved:")
print("string_edges.csv")


# ============================================================
# 12. CREATE NETWORKX GRAPH
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5: CREATING NETWORKX GRAPH")
print("=" * 70)


G = nx.Graph()


for _, row in edges.iterrows():

    protein_a = row["preferredName_A"]
    protein_b = row["preferredName_B"]
    score = row["score"]

    G.add_edge(
        protein_a,
        protein_b,
        score=score
    )


print("\nNetwork summary")
print("-" * 70)

print("Number of nodes :", G.number_of_nodes())
print("Number of edges :", G.number_of_edges())


# ============================================================
# 13. DEGREE CENTRALITY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6: CALCULATING DEGREE")
print("=" * 70)


degree = dict(
    G.degree()
)


# ============================================================
# 14. CLUSTERING COEFFICIENT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7: CALCULATING CLUSTERING COEFFICIENT")
print("=" * 70)


clustering = nx.clustering(
    G
)


# ============================================================
# 15. BETWEENNESS CENTRALITY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8: CALCULATING BETWEENNESS CENTRALITY")
print("=" * 70)


betweenness = nx.betweenness_centrality(
    G
)


# ============================================================
# 16. CREATE CENTRALITY RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    {
        "Protein": list(G.nodes()),

        "Degree": [
            degree[protein]
            for protein in G.nodes()
        ],

        "Clustering_Coefficient": [
            clustering[protein]
            for protein in G.nodes()
        ],

        "Betweenness_Centrality": [
            betweenness[protein]
            for protein in G.nodes()
        ]
    }
)


# ============================================================
# 17. SORT RESULTS
# ============================================================

results = results.sort_values(
    by="Betweenness_Centrality",
    ascending=False
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("NETWORK CENTRALITY RESULTS")
print("=" * 70)

print(
    results.to_string(index=False)
)


# ============================================================
# 19. SAVE CENTRALITY RESULTS
# ============================================================

results.to_csv(
    "protein_network_centrality.csv",
    index=False
)

print("\nSaved:")
print("protein_network_centrality.csv")


# ============================================================
# 20. IDENTIFY TOP PROTEINS
# ============================================================

print("\n")
print("=" * 70)
print("TOP PROTEINS")
print("=" * 70)


print("\nTop proteins by Degree:")
print(
    results.sort_values(
        by="Degree",
        ascending=False
    )[
        [
            "Protein",
            "Degree"
        ]
    ].head(10).to_string(index=False)
)


print("\nTop proteins by Betweenness Centrality:")
print(
    results.sort_values(
        by="Betweenness_Centrality",
        ascending=False
    )[
        [
            "Protein",
            "Betweenness_Centrality"
        ]
    ].head(10).to_string(index=False)
)


print("\nTop proteins by Clustering Coefficient:")
print(
    results.sort_values(
        by="Clustering_Coefficient",
        ascending=False
    )[
        [
            "Protein",
            "Clustering_Coefficient"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 21. SAVE COMPLETE NETWORK SUMMARY
# ============================================================

network_summary = pd.DataFrame(
    {
        "Measure": [
            "Number of input proteins",
            "Number of mapped proteins",
            "Number of network nodes",
            "Number of network edges",
            "STRING required score"
        ],

        "Value": [
            len(proteins),
            len(mapped_proteins),
            G.number_of_nodes(),
            G.number_of_edges(),
            required_score
        ]
    }
)


network_summary.to_csv(
    "network_summary.csv",
    index=False
)


# ============================================================
# 22. DRAW PPI NETWORK
# ============================================================

print("\n")
print("=" * 70)
print("STEP 9: CREATING NETWORK VISUALIZATION")
print("=" * 70)


plt.figure(
    figsize=(12, 10)
)


# Network layout
pos = nx.spring_layout(
    G,
    seed=42
)


# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=900
)


# Draw edges
nx.draw_networkx_edges(
    G,
    pos,
    width=1.5,
    alpha=0.7
)


# Draw labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=9
)


plt.title(
    "Protein-Protein Interaction Network"
)

plt.axis("off")

plt.tight_layout()


# Save figure
plt.savefig(
    "ppi_network.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("1. string_id_mapping.csv")
print("   -> STRING IDs, protein names and annotations")

print("\n2. string_edges.csv")
print("   -> Protein-protein interaction edges")

print("\n3. protein_network_centrality.csv")
print("   -> Degree, clustering coefficient and betweenness")

print("\n4. network_summary.csv")
print("   -> Overall network statistics")

print("\n5. ppi_network.png")
print("   -> PPI network visualization")

print("\nDone!")
