# Scripts Directory

This directory contains executable Python scripts for the crime analytics pipeline.

## Execution Order

Run scripts in sequence:

```bash
# 1. Data acquisition and cleaning
python scripts/01_data_acquisition.py

# 2. Exploratory data analysis
python scripts/02_eda.py

# 3. Feature engineering (coming soon)
python scripts/03_feature_engineering.py

# 4. Clustering analysis (coming soon)
python scripts/04_clustering_analysis.py

# 5. Dimensionality reduction (coming soon)
python scripts/05_dimensionality_reduction.py
```

## Scripts

- **01_data_acquisition.py** - Loads raw data, samples 500k records, cleans and validates
- **02_eda.py** - Comprehensive exploratory data analysis with visualizations
- **03_feature_engineering.py** - Creates temporal, geographic, and derived features
- **04_clustering_analysis.py** - K-Means, DBSCAN, Hierarchical clustering with MLflow
- **05_dimensionality_reduction.py** - PCA, t-SNE/UMAP visualizations

## Output Locations

- Processed data: `data/processed/`
- Visualizations: `outputs/eda/`, `outputs/clustering/`, `outputs/dimensionality/`
- Models: `models/`
- MLflow runs: `mlruns/`
