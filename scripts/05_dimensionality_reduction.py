"""
Dimensionality Reduction Script
Applies PCA, t-SNE, and UMAP for visualization and variance analysis
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
    from visualization import CrimeVisualizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure project structure is correct and modules are in: {project_root / 'src'}")
    sys.exit(1)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load clustering features and results"""
    logger.info("Loading data...")
    
    # Load features
    features_path = project_root / 'data' / 'processed' / 'clustering_features.csv'
    df_features = pd.read_csv(features_path)
    
    # Load full dataset for labels
    full_path = project_root / 'data' / 'processed' / 'crimes_with_features.csv'
    df_full = pd.read_csv(full_path)
    
    logger.info(f"✓ Loaded {len(df_features):,} records with {len(df_features.columns)} features")
    
    return df_features, df_full

def setup_mlflow():
    """Setup MLflow experiment tracking"""
    mlflow_dir = project_root / 'mlruns'
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("Chicago_Crime_DimReduction")
    
    logger.info(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"✓ Experiment: Chicago_Crime_DimReduction")

def preprocess_features(df, sample_size=50000):
    """Preprocess and sample features"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE PREPROCESSING")
    logger.info("="*60)
    
    # Sample data
    if sample_size and len(df) > sample_size:
        logger.info(f"\nSampling {sample_size:,} records...")
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
    
    return X_scaled, df_sample, scaler

def apply_pca(X, n_components=None, target_variance=0.70):
    """Apply PCA for dimensionality reduction"""
    logger.info("\n" + "="*60)
    logger.info("PCA - PRINCIPAL COMPONENT ANALYSIS")
    logger.info("="*60)
    
    with mlflow.start_run(run_name="PCA_Analysis"):
        # First, fit PCA with all components to analyze variance
        logger.info("\nAnalyzing variance explained...")
        pca_full = PCA()
        pca_full.fit(X)
        
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_target = np.argmax(cumsum_variance >= target_variance) + 1
        
        logger.info(f"✓ Original features: {X.shape[1]}")
        logger.info(f"✓ Components for {target_variance*100:.0f}% variance: {n_components_target}")
        logger.info(f"✓ Total variance explained: {cumsum_variance[n_components_target-1]*100:.2f}%")
        
        # Apply PCA with target components
        if n_components is None:
            n_components = max(3, n_components_target)
        
        logger.info(f"\nApplying PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Log to MLflow
        mlflow.log_param("algorithm", "PCA")
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("target_variance", target_variance)
        mlflow.log_metric("variance_explained", cumsum_variance[n_components-1])
        mlflow.log_metric("n_features_original", X.shape[1])
        mlflow.log_metric("n_features_reduced", n_components)
        mlflow.sklearn.log_model(pca, "model")
        
        # Save model
        model_path = project_root / 'models' / 'pca_model.pkl'
        import joblib
        joblib.dump(pca, model_path)
        
        logger.info(f"\n✓ PCA Results:")
        logger.info(f"   Reduced dimensions: {X.shape[1]} → {n_components}")
        logger.info(f"   Variance explained: {cumsum_variance[n_components-1]*100:.2f}%")
        logger.info(f"   Dimensionality reduction: {(1 - n_components/X.shape[1])*100:.1f}%")
        
        # Variance per component
        logger.info(f"\n   Top 5 components variance:")
        for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
            logger.info(f"      PC{i}: {var*100:.2f}%")
    
    return X_pca, pca, cumsum_variance

def apply_tsne(X, n_components=2, perplexity=30, n_iter=1000):
    """Apply t-SNE for visualization"""
    logger.info("\n" + "="*60)
    logger.info("t-SNE - T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING")
    logger.info("="*60)
    
    with mlflow.start_run(run_name=f"tSNE_perplexity{perplexity}"):
        logger.info(f"\nApplying t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            verbose=0
        )
        
        X_tsne = tsne.fit_transform(X)
        
        # Log to MLflow
        mlflow.log_param("algorithm", "t-SNE")
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("perplexity", perplexity)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_metric("kl_divergence", tsne.kl_divergence_)
        
        logger.info(f"\n✓ t-SNE Results:")
        logger.info(f"   Reduced dimensions: {X.shape[1]} → {n_components}")
        logger.info(f"   KL divergence: {tsne.kl_divergence_:.4f}")
        logger.info(f"   Iterations: {tsne.n_iter_}")
    
    return X_tsne, tsne

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """Apply UMAP for visualization"""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available, skipping...")
        return None, None
    
    logger.info("\n" + "="*60)
    logger.info("UMAP - UNIFORM MANIFOLD APPROXIMATION AND PROJECTION")
    logger.info("="*60)
    
    with mlflow.start_run(run_name=f"UMAP_neighbors{n_neighbors}"):
        logger.info(f"\nApplying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        
        umap = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False
        )
        
        X_umap = umap.fit_transform(X)
        
        # Log to MLflow
        mlflow.log_param("algorithm", "UMAP")
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("min_dist", min_dist)
        
        # Save model
        model_path = project_root / 'models' / 'umap_model.pkl'
        import joblib
        joblib.dump(umap, model_path)
        
        logger.info(f"\n✓ UMAP Results:")
        logger.info(f"   Reduced dimensions: {X.shape[1]} → {n_components}")
        logger.info(f"   Model saved: {model_path}")
    
    return X_umap, umap

def create_pca_visualizations(pca, cumsum_variance, output_dir):
    """Create PCA-specific visualizations"""
    logger.info("\nCreating PCA visualizations...")
    
    # 1. Scree plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
        y=pca.explained_variance_ratio_ * 100,
        mode='lines+markers',
        name='Individual Variance'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumsum_variance) + 1)),
        y=cumsum_variance * 100,
        mode='lines+markers',
        name='Cumulative Variance'
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="70% Target")
    fig.update_layout(
        title='PCA Variance Explained - Scree Plot',
        xaxis_title='Principal Component',
        yaxis_title='Variance Explained (%)',
        showlegend=True
    )
    fig.write_html(str(output_dir / 'pca_scree_plot.html'))
    logger.info("   ✓ Saved: pca_scree_plot.html")
    
    # 2. Component loadings (top 3 PCs)
    if pca.n_components_ >= 3:
        feature_names = ['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month', 
                        'IsWeekend', 'IsNight', 'CrimeSeverity', 'Arrest', 'Domestic']
        
        loadings = pd.DataFrame(
            pca.components_[:3].T,
            columns=['PC1', 'PC2', 'PC3'],
            index=feature_names
        )
        
        fig = go.Figure()
        for col in loadings.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=loadings.index,
                y=loadings[col],
                text=loadings[col].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='PCA Component Loadings (Top 3 Components)',
            xaxis_title='Features',
            yaxis_title='Loading',
            barmode='group',
            showlegend=True
        )
        fig.write_html(str(output_dir / 'pca_loadings.html'))
        logger.info("   ✓ Saved: pca_loadings.html")

def create_2d_scatter_plots(X_pca, X_tsne, X_umap, df_sample, output_dir):
    """Create 2D scatter plots for all methods"""
    logger.info("\nCreating 2D scatter visualizations...")
    
    # Load crime type for coloring
    full_df = pd.read_csv(project_root / 'data' / 'processed' / 'crimes_with_features.csv')
    df_viz = full_df.sample(n=len(df_sample), random_state=42)
    
    # 1. PCA 2D
    if X_pca.shape[1] >= 2:
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df_viz['Primary Type'].values,
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Crime Type'},
            title='PCA - 2D Projection',
            opacity=0.6
        )
        fig.write_html(str(output_dir / 'pca_2d_scatter.html'))
        logger.info("   ✓ Saved: pca_2d_scatter.html")
    
    # 2. t-SNE 2D
    if X_tsne is not None:
        fig = px.scatter(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            color=df_viz['Primary Type'].values,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Crime Type'},
            title='t-SNE - 2D Projection',
            opacity=0.6
        )
        fig.write_html(str(output_dir / 'tsne_2d_scatter.html'))
        logger.info("   ✓ Saved: tsne_2d_scatter.html")
    
    # 3. UMAP 2D
    if X_umap is not None:
        fig = px.scatter(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            color=df_viz['Primary Type'].values,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': 'Crime Type'},
            title='UMAP - 2D Projection',
            opacity=0.6
        )
        fig.write_html(str(output_dir / 'umap_2d_scatter.html'))
        logger.info("   ✓ Saved: umap_2d_scatter.html")

def create_3d_scatter_plots(X_pca, df_sample, output_dir):
    """Create 3D scatter plot for PCA"""
    if X_pca.shape[1] < 3:
        return
    
    logger.info("\nCreating 3D scatter visualization...")
    
    # Load crime type for coloring
    full_df = pd.read_csv(project_root / 'data' / 'processed' / 'crimes_with_features.csv')
    df_viz = full_df.sample(n=len(df_sample), random_state=42)
    
    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=df_viz['Primary Type'].values,
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Crime Type'},
        title='PCA - 3D Projection',
        opacity=0.5
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(str(output_dir / 'pca_3d_scatter.html'))
    logger.info("   ✓ Saved: pca_3d_scatter.html")

def save_results_summary(pca, tsne, umap, output_dir):
    """Save comprehensive results summary"""
    summary_path = output_dir / 'dimensionality_reduction_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DIMENSIONALITY REDUCTION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. PCA (Principal Component Analysis)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Components: {pca.n_components_}\n")
        f.write(f"Variance explained (total): {sum(pca.explained_variance_ratio_)*100:.2f}%\n")
        f.write(f"Top 3 components: {pca.explained_variance_ratio_[:3]*100}\n\n")
        
        f.write("2. t-SNE (T-Distributed Stochastic Neighbor Embedding)\n")
        f.write("-" * 40 + "\n")
        if tsne:
            f.write(f"Components: {tsne.n_components}\n")
            f.write(f"KL divergence: {tsne.kl_divergence_:.4f}\n")
            f.write(f"Iterations: {tsne.n_iter_}\n\n")
        else:
            f.write("Not applied\n\n")
        
        f.write("3. UMAP (Uniform Manifold Approximation and Projection)\n")
        f.write("-" * 40 + "\n")
        if umap:
            f.write(f"Components: {umap.n_components}\n")
            f.write(f"N neighbors: {umap.n_neighbors}\n")
            f.write(f"Min distance: {umap.min_dist}\n\n")
        else:
            f.write("Not available (install: pip install umap-learn)\n\n")
        
        f.write("="*60 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*60 + "\n\n")
        f.write(f"- PCA achieved {sum(pca.explained_variance_ratio_)*100:.1f}% variance with {pca.n_components_} components\n")
        f.write(f"- Dimensionality reduced from 10 to {pca.n_components_} features\n")
        f.write(f"- All visualizations support crime pattern analysis\n")
        f.write(f"- Ready for Streamlit dashboard deployment\n")
    
    logger.info(f"✓ Summary saved: {summary_path}")

def main():
    """Main dimensionality reduction pipeline"""
    try:
        # Setup
        setup_mlflow()
        
        # Load data
        df_features, df_full = load_data()
        
        # Preprocess
        X_scaled, df_sample, scaler = preprocess_features(df_features, sample_size=50000)
        
        # Apply PCA
        X_pca, pca, cumsum_variance = apply_pca(X_scaled, n_components=5, target_variance=0.70)
        
        # Apply t-SNE
        X_tsne, tsne = apply_tsne(X_scaled, n_components=2, perplexity=30, n_iter=1000)
        
        # Apply UMAP
        X_umap, umap = apply_umap(X_scaled, n_components=2, n_neighbors=15, min_dist=0.1)
        
        # Create visualizations
        output_dir = project_root / 'outputs' / 'dimensionality_reduction'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        create_pca_visualizations(pca, cumsum_variance, output_dir)
        create_2d_scatter_plots(X_pca, X_tsne, X_umap, df_sample, output_dir)
        create_3d_scatter_plots(X_pca, df_sample, output_dir)
        
        # Save summary
        save_results_summary(pca, tsne, umap, output_dir)
        
        logger.info("\n" + "="*60)
        logger.info("DIMENSIONALITY REDUCTION COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ PCA: {X_scaled.shape[1]} → {pca.n_components_} dimensions ({sum(pca.explained_variance_ratio_)*100:.1f}% variance)")
        logger.info(f"✓ t-SNE: 2D projection complete")
        if X_umap is not None:
            logger.info(f"✓ UMAP: 2D projection complete")
        logger.info(f"✓ Visualizations saved to: {output_dir}")
        logger.info(f"✓ MLflow experiments: {mlflow.get_tracking_uri()}")
        logger.info(f"✓ All analysis complete - Ready for Streamlit deployment!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in dimensionality reduction pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
