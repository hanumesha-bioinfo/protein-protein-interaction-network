# Reproducibility Record

## Project
Human Protein-Protein Interaction Network Analysis and Machine Learning.

## Data
- Organism: Homo sapiens.
- Taxonomy ID: 9606.
- Initial selected protein set: 20 proteins.
- Primary interaction input: string_edges.csv.
- Data source and retrieval date: verify against the original download records.

## Network Analysis
- Software: Python, pandas, NetworkX.
- Metrics: degree, betweenness centrality, clustering coefficient, network density, connected components, and community assignments.
- Interaction confidence threshold: verify in analysis.py.
- Community detection algorithm: verify in src/ppi/network/analyze_network.py.

## Machine Learning
- Models: Logistic Regression and Random Forest.
- Task: exploratory protein interaction link prediction.
- Random seed: 42.
- Features: common neighbours, Jaccard, Adamic-Adar, resource allocation, preferential attachment, and degree sum.
- Metrics: precision, recall, F1, ROC-AUC, and average precision.

## Reproduction
1. Create and activate a Python virtual environment.
2. Install the documented project dependencies.
3. Run the network analysis module.
4. Run the link prediction module.
5. Launch the Streamlit dashboard.

## Limitations
The initial network is a small, selected protein set, not the complete human interactome. STRING associations do not always indicate direct physical binding. Sampled unobserved pairs are not confirmed biological negatives. Model results are exploratory and require robust evaluation and independent validation.
