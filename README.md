# Chicago Crime Analytics Platform

## Urban Safety Intelligence Platform

A comprehensive crime analytics platform leveraging unsupervised machine learning to analyze 500,000 Chicago crime records, identify crime hotspots, and optimize police resource allocation.

## Dataset Overview

### Chicago Crime Dataset (2001-2025)
- **Full Dataset**: 7.8 Million crime records
- **Sample Used**: 500,000 recent crime records
- **Input Features**: 22 comprehensive variables
- **Crime Categories**: 33 distinct crime types
- **Geographic Coverage**: City of Chicago districts and wards
- **Data Source**: [Chicago Data Portal - Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)
- **Format**: CSV (Comma-separated values)
- **Update Frequency**: Daily updates from Chicago Police Department

### Key Features (22 Variables)
**Crime Identification**: ID, Case Number, IUCR, FBI Code  
**Crime Classification**: Primary Type (33 categories), Description, Location Description  
**Temporal**: Date, Year, Hour, Day of Week, Month, Season  
**Geographic**: Latitude, Longitude, Block, Beat, District, Ward, Community Area  
**Status**: Arrest (True/False), Domestic (True/False)  
**Engineered**: Crime Severity Score, Is Weekend, Time Period

## Project Structure

```
ChicagoCrimeAnalytics/
├── data/
│   ├── raw/                    # Original Chicago crime dataset
│   └── processed/              # Cleaned and engineered datasets
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering_analysis.ipynb
│   └── 05_dimensionality_reduction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   └── visualization.py
├── models/                     # Saved clustering models
├── mlruns/                     # MLflow tracking data
├── streamlit_app/
│   ├── app.py                  # Main Streamlit application
│   └── pages/                  # Multi-page application
└── requirements.txt
```

## Features

- **Crime Hotspot Identification**: K-Means, DBSCAN, Hierarchical Clustering
- **Temporal Pattern Analysis**: Time-based crime clustering
- **Dimensionality Reduction**: PCA, t-SNE/UMAP visualizations
- **MLflow Integration**: Experiment tracking and model comparison
- **Interactive Dashboard**: Streamlit-based web application
- **Cloud Deployment**: Production-ready on Streamlit Cloud

## Business Impact

- Optimize patrol routes and reduce response time by 60%
- Identify high-risk areas for increased police presence
- Data-driven urban planning for safer neighborhoods
- Evidence-based resource allocation

## Expected Results

### Clustering Analysis Results
- **Geographic Hotspots**: 5-10 distinct crime zones identified
- **K-Means**: Circular hotspot zones with patrol focus centers
- **DBSCAN**: Naturally formed high-crime areas with noise filtering
- **Hierarchical**: Nested zone relationships and hierarchies
- **Metrics**: Silhouette score > 0.5, Davies-Bouldin index optimization
- **Deliverable**: Color-coded crime heatmap (red=high risk, yellow=medium, green=low)

### Temporal Pattern Results
- **Time Clusters**: 3-5 distinct time-based crime patterns
- **Peak Times**: Identify high-risk hours (e.g., 10 PM - 2 AM)
- **Seasonal Trends**: Monthly and seasonal crime variations
- **Weekly Patterns**: Weekday vs weekend crime comparison
- **Deliverable**: Hourly heatmap showing crime concentration

### Dimensionality Reduction Results
- **PCA**: Reduce 22 features to 2-3 components (70%+ variance retained)
- **Feature Importance**: Identify top 5 features driving patterns
- **t-SNE/UMAP**: 2D visualizations showing distinct crime clusters
- **Deliverable**: Interactive scatter plots with crime type coloring

## Technology Stack

- **Python 3.9+**: Core programming language
- **Scikit-learn**: Clustering & dimensionality reduction algorithms
- **MLflow**: Experiment tracking and model registry
- **Streamlit**: Interactive web application framework
- **Pandas, NumPy**: Data processing and numerical operations
- **Plotly, Seaborn**: Interactive and static visualizations
- **Folium**: Geographic crime heatmaps
- **GeoPandas**: Geospatial data analysis

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd ChicagoCrimeAnalytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition
- Visit [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)
- Click "Export" → Select "CSV" format
- Download recent 500,000 records (use filters if needed)
- Place as `data/raw/chicago_crimes_500k.csv`

### 3. Run Analysis Pipeline
```bash
# Run notebooks in sequence
jupyter notebook notebooks/01_data_acquisition.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_clustering_analysis.ipynb
jupyter notebook notebooks/05_dimensionality_reduction.ipynb

# Start MLflow tracking server
mlflow ui

# Launch Streamlit application
streamlit run streamlit_app/app.py
```

### 4. Cloud Deployment (Streamlit Cloud)
- Push repository to GitHub
- Connect to Streamlit Cloud
- Deploy from `streamlit_app/app.py`
- Configure secrets if needed

## Project Evaluation Criteria

### Technical Performance (70%)
- **Data Preprocessing & Sampling** (10%): Data quality, cleaning methodology
- **Clustering Analysis** (30%): 3+ algorithms, performance comparison, metrics
- **Dimensionality Reduction** (20%): PCA and t-SNE/UMAP visualizations
- **MLflow Integration** (10%): Experiment tracking for all models

### Application Development (30%)
- **Streamlit Application** (20%): Interactive visualizations, multi-page UI
- **Cloud Deployment** (10%): Stability, performance, accessibility on Streamlit Cloud

### Bonus Deliverables (+10%)
- **Docker Containerization**: Dockerfile, docker-compose.yml, deployment docs

## Technical Tags

`Python` `Data Preprocessing` `Feature Engineering` `Unsupervised Learning` `K-Means Clustering` `DBSCAN` `Hierarchical Clustering` `PCA` `t-SNE` `UMAP` `MLflow` `Streamlit Cloud` `Public Safety` `Crime Analytics` `Geographic Data Analysis` `Temporal Analysis` `Data Visualization` `Big Data Processing`

## Crime Categories (33 Types)

THEFT, BATTERY, CRIMINAL DAMAGE, NARCOTICS, ASSAULT, BURGLARY, MOTOR VEHICLE THEFT, ROBBERY, DECEPTIVE PRACTICE, CRIMINAL TRESPASS, WEAPONS VIOLATION, PUBLIC PEACE VIOLATION, OFFENSE INVOLVING CHILDREN, CRIM SEXUAL ASSAULT, SEX OFFENSE, GAMBLING, LIQUOR LAW VIOLATION, ARSON, INTERFERENCE WITH PUBLIC OFFICER, HOMICIDE, KIDNAPPING, INTIMIDATION, STALKING, OBSCENITY, and others

## Author

Crime Intelligence Analyst  
Chicago Police Department Analytics Team

---

**Note**: This is an educational project for crime pattern analysis and should be used responsibly for public safety research and urban planning purposes.
