import pandas as pd

# Load Open Targets data
df = pd.read_csv("target-disease.tsv", sep="\t")

# Keep one row per gene
df = df.drop_duplicates(subset="symbol")

# Remove missing gene symbols
df = df.dropna(subset=["symbol"])

# Clean gene names
df["symbol"] = df["symbol"].astype(str).str.strip()

# Select high-confidence leukemia-associated genes
positive = df[df["globalScore"] >= 0.5].copy()

# Add supervised-learning label
positive["label"] = 1

# Keep useful columns
positive = positive[
    [
        "symbol",
        "globalScore",
        "clinicalPrecedence",
        "gwasCredibleSets",
        "geneBurden",
        "eva",
        "gene2Phenotype",
        "clingen",
        "cancerGeneCensus",
        "intogen",
        "evaSomatic",
        "cancerBiomarkers",
        "crisprScreen",
        "crispr",
        "reactome",
        "europepmc",
        "expressionAtlas",
        "label"
    ]
]

# Save
positive.to_csv("leukemia_positive_genes.csv", index=False)

print("Positive genes:", len(positive))
print("\nFirst 20 genes:")
print(positive[["symbol", "globalScore", "label"]].head(20).to_string(index=False))
