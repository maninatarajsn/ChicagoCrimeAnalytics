"""
Clustering Analysis Script
Performs K-Means, DBSCAN, and Hierarchical clustering with MLflow tracking
"""

import sys
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from clustering import CrimeClustering
    from visualization import CrimeVisualizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure project structure is correct and modules are in: {project_root / 'src'}")
    sys.exit(1)

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_clustering_data():
    """Load feature matrix for clustering"""
    data_path = project_root / 'data' / 'processed' / 'clustering_features.csv'
    logger.info(f"Loading clustering features from: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df):,} records with {len(df.columns)} features")
    logger.info(f"✓ Features: {', '.join(df.columns.tolist())}")
    
    return df

def setup_mlflow():
    """Setup MLflow experiment tracking"""
    mlflow_dir = project_root / 'mlruns'
    mlflow_dir.mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("Chicago_Crime_Clustering")
    
    logger.info(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"✓ Experiment: Chicago_Crime_Clustering")

def preprocess_features(df, sample_size=50000):
    """Preprocess and sample features for clustering"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE PREPROCESSING")
    logger.info("="*60)
    
    # Sample data for faster clustering (optional)
    if sample_size and len(df) > sample_size:
        logger.info(f"\nSampling {sample_size:,} records for clustering...")
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.info(f"✓ Sampled dataset: {len(df_sample):,} records")
    else:
        df_sample = df.copy()
    
    # Standardize features
    logger.info("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample)
    
    logger.info(f"✓ Feature scaling complete")
    logger.info(f"✓ Scaled data shape: {X_scaled.shape}")
    logger.info(f"✓ Mean: {X_scaled.mean():.6f}, Std: {X_scaled.std():.6f}")
    
    return X_scaled, df_sample, scaler

def evaluate_clustering(X, labels, algorithm_name):
    """Calculate clustering metrics"""
    # Filter out noise points for DBSCAN (-1 labels)
    mask = labels != -1
    if mask.sum() < 2:
        logger.warning(f"Not enough valid clusters for {algorithm_name}")
        return {}
    
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    n_clusters = len(np.unique(labels_valid))
    n_noise = (labels == -1).sum()
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise if algorithm_name == 'DBSCAN' else 0,
        'n_samples': len(labels)
    }
    
    # Calculate metrics only if we have valid clusters
    if n_clusters > 1 and len(labels_valid) > n_clusters:
        try:
            metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_valid, labels_valid)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_valid, labels_valid)
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
    
    return metrics

def run_kmeans_clustering(X, df_sample, feature_names, k_range=(3, 10)):
    """Run K-Means clustering with optimal k selection"""
    logger.info("\n" + "="*60)
    logger.info("K-MEANS CLUSTERING")
    logger.info("="*60)
    
    clustering = CrimeClustering(df_sample)
    visualizer = CrimeVisualizer(df_sample)
    
    # Preprocess features first
    clustering.preprocess_features(feature_names)
    
    # Find optimal k
    logger.info(f"\nFinding optimal k in range {k_range}...")
    opt_results = clustering.find_optimal_k(k_range=range(*k_range))
    optimal_k = opt_results['optimal_k']
    opt_metrics = opt_results['metrics']
    logger.info(f"✓ Optimal k: {optimal_k}")
    
    # Create elbow curve visualization
    output_dir = project_root / 'outputs' / 'clustering'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = visualizer.plot_elbow_curve(opt_metrics)
    fig.write_html(str(output_dir / 'kmeans_elbow_curve.html'))
    logger.info(f"✓ Saved elbow curve: kmeans_elbow_curve.html")
    
    # Run K-Means with optimal k
    with mlflow.start_run(run_name=f"KMeans_k{optimal_k}"):
        logger.info(f"\nRunning K-Means with k={optimal_k}...")
        results = clustering.kmeans_clustering(n_clusters=optimal_k)
        
        labels = results['labels']
        model = results['model']
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", optimal_k)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("silhouette_score", results['silhouette_score'])
        mlflow.log_metric("davies_bouldin_score", results['davies_bouldin_score'])
        mlflow.log_metric("inertia", results['inertia'])
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = project_root / 'models' / 'kmeans_model.pkl'
        clustering.save_model(model, model_path)
        
        logger.info(f"\n✓ K-Means Results:")
        logger.info(f"   Clusters: {results['n_clusters']}")
        logger.info(f"   Silhouette Score: {results['silhouette_score']:.4f}")
        logger.info(f"   Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
        logger.info(f"   Inertia: {results['inertia']:.2f}")
    
    return labels, results

def run_dbscan_clustering(X, df_sample, feature_names, eps=0.5, min_samples=50):
    """Run DBSCAN clustering"""
    logger.info("\n" + "="*60)
    logger.info("DBSCAN CLUSTERING")
    logger.info("="*60)
    
    clustering = CrimeClustering(df_sample)
    clustering.preprocess_features(feature_names)
    
    with mlflow.start_run(run_name=f"DBSCAN_eps{eps}_min{min_samples}"):
        logger.info(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")
        results = clustering.dbscan_clustering(eps=eps, min_samples=min_samples)
        
        labels = results['labels']
        model = results['model']
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_param("eps", eps)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("n_clusters", results['n_clusters'])
        mlflow.log_metric("n_noise", results['n_noise'])
        mlflow.log_metric("noise_ratio", results['noise_ratio'])
        
        if 'silhouette_score' in results:
            mlflow.log_metric("silhouette_score", results['silhouette_score'])
        if 'davies_bouldin_score' in results:
            mlflow.log_metric("davies_bouldin_score", results['davies_bouldin_score'])
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = project_root / 'models' / 'dbscan_model.pkl'
        clustering.save_model(model, model_path)
        
        logger.info(f"\n✓ DBSCAN Results:")
        logger.info(f"   Clusters: {results['n_clusters']}")
        logger.info(f"   Noise points: {results['n_noise']:,} ({results['noise_ratio']*100:.1f}%)")
        if 'silhouette_score' in results:
            logger.info(f"   Silhouette Score: {results['silhouette_score']:.4f}")
    
    return labels, results

def run_hierarchical_clustering(X, df_sample, feature_names, n_clusters=5):
    """Run Hierarchical clustering"""
    logger.info("\n" + "="*60)
    logger.info("HIERARCHICAL CLUSTERING")
    logger.info("="*60)
    
    clustering = CrimeClustering(df_sample)
    clustering.preprocess_features(feature_names)
    
    with mlflow.start_run(run_name=f"Hierarchical_k{n_clusters}"):
        logger.info(f"\nRunning Hierarchical clustering (k={n_clusters})...")
        results = clustering.hierarchical_clustering(n_clusters=n_clusters)
        
        labels = results['labels']
        model = results['model']
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("algorithm", "Hierarchical")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("linkage", "ward")
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("silhouette_score", results['silhouette_score'])
        mlflow.log_metric("davies_bouldin_score", results['davies_bouldin_score'])
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = project_root / 'models' / 'hierarchical_model.pkl'
        clustering.save_model(model, model_path)
        
        logger.info(f"\n✓ Hierarchical Results:")
        logger.info(f"   Clusters: {results['n_clusters']}")
        logger.info(f"   Silhouette Score: {results['silhouette_score']:.4f}")
        logger.info(f"   Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
    
    return labels, results

def create_cluster_visualizations(df_sample, kmeans_labels, dbscan_labels, hierarchical_labels):
    """Create visualizations for all clustering results"""
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    visualizer = CrimeVisualizer(df_sample)
    output_dir = project_root / 'outputs' / 'clustering'
    
    # Add cluster labels to dataframe
    df_viz = df_sample.copy()
    df_viz['KMeans_Cluster'] = kmeans_labels
    df_viz['DBSCAN_Cluster'] = dbscan_labels
    df_viz['Hierarchical_Cluster'] = hierarchical_labels
    
    # 1. K-Means cluster distribution
    logger.info("\nCreating K-Means visualizations...")
    fig = visualizer.plot_cluster_distribution(kmeans_labels)
    fig.update_layout(title="K-Means Cluster Distribution")
    fig.write_html(str(output_dir / 'kmeans_distribution.html'))
    
    # 2. DBSCAN cluster distribution
    logger.info("Creating DBSCAN visualizations...")
    fig = visualizer.plot_cluster_distribution(dbscan_labels)
    fig.update_layout(title="DBSCAN Cluster Distribution")
    fig.write_html(str(output_dir / 'dbscan_distribution.html'))
    
    # 3. Hierarchical cluster distribution
    logger.info("Creating Hierarchical visualizations...")
    fig = visualizer.plot_cluster_distribution(hierarchical_labels)
    fig.update_layout(title="Hierarchical Cluster Distribution")
    fig.write_html(str(output_dir / 'hierarchical_distribution.html'))
    
    # 4. Geographic cluster maps
    logger.info("Creating geographic heatmaps...")
    
    # Only create heatmap if we have lat/lon in original data
    full_df = pd.read_csv(project_root / 'data' / 'processed' / 'crimes_with_features.csv')
    full_df_sample = full_df.sample(n=min(10000, len(df_sample)), random_state=42)
    full_df_sample['KMeans_Cluster'] = kmeans_labels[:len(full_df_sample)]
    
    visualizer_full = CrimeVisualizer(full_df_sample)
    fig = visualizer_full.plot_crime_heatmap()
    fig.update_layout(title="Crime Hotspots - K-Means Clusters")
    fig.write_html(str(output_dir / 'kmeans_geographic_heatmap.html'))
    
    logger.info(f"\n✓ All visualizations saved to: {output_dir}")

def save_results_summary(kmeans_metrics, dbscan_metrics, hierarchical_metrics):
    """Save comprehensive results summary"""
    output_path = project_root / 'outputs' / 'clustering' / 'clustering_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CHICAGO CRIME CLUSTERING ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. K-MEANS CLUSTERING\n")
        f.write("-" * 40 + "\n")
        for key, value in kmeans_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n2. DBSCAN CLUSTERING\n")
        f.write("-" * 40 + "\n")
        for key, value in dbscan_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n3. HIERARCHICAL CLUSTERING\n")
        f.write("-" * 40 + "\n")
        for key, value in hierarchical_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*60 + "\n\n")
        
        # Compare silhouette scores
        scores = {
            'K-Means': kmeans_metrics.get('silhouette_score', kmeans_metrics.get('silhouette_score', 0)),
            'DBSCAN': dbscan_metrics.get('silhouette_score', dbscan_metrics.get('silhouette_score', 0)),
            'Hierarchical': hierarchical_metrics.get('silhouette_score', hierarchical_metrics.get('silhouette_score', 0))
        }
        
        best_algo = max(scores, key=scores.get)
        f.write(f"Best algorithm by Silhouette Score: {best_algo} ({scores[best_algo]:.4f})\n\n")
        
        f.write("Next Steps:\n")
        f.write("- Run dimensionality reduction (PCA, t-SNE, UMAP)\n")
        f.write("- Deploy Streamlit dashboard for interactive exploration\n")
        f.write("- Analyze cluster characteristics for actionable insights\n")
    
    logger.info(f"✓ Summary saved: {output_path}")

def main():
    """Main clustering analysis pipeline"""
    try:
        # Setup
        setup_mlflow()
        
        # Load data
        df = load_clustering_data()
        feature_names = df.columns.tolist()
        
        # Preprocess features
        X_scaled, df_sample, scaler = preprocess_features(df, sample_size=50000)
        
        # Run clustering algorithms
        kmeans_labels, kmeans_metrics = run_kmeans_clustering(
            X_scaled, df_sample, feature_names, k_range=(3, 10)
        )
        

        dbscan_labels, dbscan_metrics = run_dbscan_clustering(
            X_scaled, df_sample, feature_names, eps=1.5, min_samples=25
        )
        
        hierarchical_labels, hierarchical_metrics = run_hierarchical_clustering(
            X_scaled, df_sample, feature_names, n_clusters=5
        )
        
        # Create visualizations
        create_cluster_visualizations(
            df_sample, kmeans_labels, dbscan_labels, hierarchical_labels
        )
        
        # Save results summary
        save_results_summary(kmeans_metrics, dbscan_metrics, hierarchical_metrics)
        
        logger.info("\n" + "="*60)
        logger.info("CLUSTERING ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Models saved to: {project_root / 'models'}")
        logger.info(f"✓ Visualizations saved to: {project_root / 'outputs' / 'clustering'}")
        logger.info(f"✓ MLflow experiments: {mlflow.get_tracking_uri()}")
        logger.info(f"✓ Next step: Run 05_dimensionality_reduction.py")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in clustering pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
