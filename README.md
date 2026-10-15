# Human Protein–Protein Interaction Network Analysis and Machine Learning

An exploratory bioinformatics project combining protein–protein interaction (PPI) network analysis, graph-based machine learning, and interactive visualisation.

## Project Overview

This project uses human protein interaction data from STRING to explore network structure, identify important proteins, detect communities, and investigate potential interactions using machine learning.

## Objectives

- Analyse human protein–protein interaction networks.
- Identify highly connected proteins and network hubs.
- Calculate degree, betweenness centrality, and clustering coefficients.
- Detect communities within the protein network.
- Compare Logistic Regression and Random Forest for exploratory link prediction.
- Visualise network statistics and ML results in an interactive dashboard.

## Technologies

- Python
- pandas and NumPy
- NetworkX
- scikit-learn
- Matplotlib
- Plotly
- Streamlit
- STRING database
- joblib

## Project Structure

```text
protein-protein-interaction/
├── analysis.py
├── create_leukemia_labels.py
├── string_edges.csv
├── string_id_mapping.csv
├── protein_network_centrality.csv
├── network_summary.csv
├── ppi_network.png
├── src/
│   └── ppi/
│       ├── network/
│       │   └── analyze_network.py
│       └── ml/
│           └── link_prediction.py
├── dashboard/
│   └── app.py
├── results/
│   ├── figures/
│   ├── tables/
│   └── ml/
└── tests/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Hanumesh-gif/protein-protein-interaction-network.git
cd protein-protein-interaction-network
```

Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install pandas numpy networkx scikit-learn matplotlib plotly streamlit joblib requests
```

## Running the Project

### Network analysis

```bash
python -m src.ppi.network.analyze_network
```

This generates network metrics, summary statistics, community assignments, and a network figure under `results/`.

### Machine-learning link prediction

```bash
python -m src.ppi.ml.link_prediction
```

This generates model metrics, test predictions, feature importance, and trained model files under `results/ml/`.

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open the local address displayed in your terminal, usually http://localhost:8501.

Run the analysis and ML modules first if their result files have not yet been generated.

## Initial Network Results

The initial exploratory network contains:

- 20 selected protein nodes
- 158 observed interactions
- Network density of approximately 0.832
- Average degree of 15.8
- Average clustering coefficient of approximately 0.755
- Two detected communities

These statistics describe the selected protein set, not the entire human interactome.

## Machine Learning

The link-prediction experiment compares Logistic Regression and Random Forest using graph-derived features:

- Common neighbours
- Jaccard coefficient
- Adamic–Adar index
- Resource allocation
- Preferential attachment
- Sum of endpoint degrees

Evaluation metrics and predictions are saved in `results/ml/`.

## Research Limitations

- The initial network is small and contains selected proteins.
- STRING associations do not always represent direct physical binding.
- Unobserved protein pairs are not necessarily biological negatives.
- Initial ML metrics are exploratory and require independent validation.
- Model predictions should not be treated as proof of biological causation or clinical validity.

## Future Work

- Expand the network with additional proteins and interaction evidence.
- Compare networks across disease-related gene sets.
- Improve negative sampling and independent model validation.
- Add automated tests and reproducible configuration files.
- Extend the dashboard with additional network comparisons.

## Author

Hanumesha

GitHub: https://github.com/Hanumesh-gif

## Disclaimer

This project is intended for educational and research purposes. It is not a clinical diagnostic or treatment tool.
