"""
Clustering Module for Chicago Crime Analytics
Implements K-Means, DBSCAN, and Hierarchical Clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeClustering:
    """
    Crime clustering analysis with multiple algorithms
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize clustering analyzer
        
        Args:
            df: Feature matrix for clustering
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.scaled_features = None
        self.models = {}
        self.results = {}
        
    def preprocess_features(self, features: list) -> np.ndarray:
        """
        Scale features for clustering
        
        Args:
            features: List of feature column names
            
        Returns:
            Scaled feature array
        """
        logger.info("Preprocessing features for clustering...")
        
        X = self.df[features].values
        self.scaled_features = self.scaler.fit_transform(X)
        
        logger.info(f"Features scaled: {self.scaled_features.shape}")
        return self.scaled_features
    
    def kmeans_clustering(self, n_clusters: int = 5, **kwargs) -> dict:
        """
        Perform K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            **kwargs: Additional KMeans parameters
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Running K-Means clustering with {n_clusters} clusters...")
        
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = model.fit_predict(self.scaled_features)
        
        # Calculate metrics
        silhouette = silhouette_score(self.scaled_features, labels)
        davies_bouldin = davies_bouldin_score(self.scaled_features, labels)
        
        results = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'inertia': model.inertia_
        }
        
        self.models['kmeans'] = model
        self.results['kmeans'] = results
        
        logger.info(f"K-Means - Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}")
        return results
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 50, **kwargs) -> dict:
        """
        Perform DBSCAN clustering
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            **kwargs: Additional DBSCAN parameters
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Running DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = model.fit_predict(self.scaled_features)
        
        # Calculate metrics (excluding noise points)
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        n_noise = list(labels).count(-1)
        
        results = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels)
        }
        
        if n_clusters > 1:
            silhouette = silhouette_score(self.scaled_features[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(self.scaled_features[mask], labels[mask])
            results['silhouette_score'] = silhouette
            results['davies_bouldin_score'] = davies_bouldin
            logger.info(f"DBSCAN - Clusters: {n_clusters}, Silhouette: {silhouette:.3f}")
        else:
            logger.warning("DBSCAN found insufficient clusters for metric calculation")
        
        self.models['dbscan'] = model
        self.results['dbscan'] = results
        
        return results
    
    def hierarchical_clustering(self, n_clusters: int = 5, linkage: str = 'ward', **kwargs) -> dict:
        """
        Perform Hierarchical clustering
        
        Args:
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average')
            **kwargs: Additional AgglomerativeClustering parameters
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Running Hierarchical clustering ({linkage} linkage)...")
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage, 
            **kwargs
        )
        labels = model.fit_predict(self.scaled_features)
        
        # Calculate metrics
        silhouette = silhouette_score(self.scaled_features, labels)
        davies_bouldin = davies_bouldin_score(self.scaled_features, labels)
        
        results = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'linkage': linkage
        }
        
        self.models['hierarchical'] = model
        self.results['hierarchical'] = results
        
        logger.info(f"Hierarchical - Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}")
        return results
    
    def find_optimal_k(self, k_range: range = range(2, 11)) -> dict:
        """
        Find optimal number of clusters using elbow method
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Dictionary with metrics for each k
        """
        logger.info("Finding optimal K using elbow method...")
        
        metrics = {
            'k_values': [],
            'inertias': [],
            'silhouettes': [],
            'davies_bouldins': []
        }
        
        for k in k_range:
            results = self.kmeans_clustering(n_clusters=k)
            metrics['k_values'].append(k)
            metrics['inertias'].append(results['inertia'])
            metrics['silhouettes'].append(results['silhouette_score'])
            metrics['davies_bouldins'].append(results['davies_bouldin_score'])
        
        # Find optimal k (highest silhouette score)
        optimal_idx = np.argmax(metrics['silhouettes'])
        optimal_k = metrics['k_values'][optimal_idx]
        
        logger.info(f"Optimal K: {optimal_k} (Silhouette: {metrics['silhouettes'][optimal_idx]:.3f})")
        
        return {
            'optimal_k': optimal_k,
            'metrics': metrics
        }
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save clustering model to file
        
        Args:
            model_name: Name of the model ('kmeans', 'dbscan', 'hierarchical')
            filepath: Path to save the model
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model '{model_name}' saved to {filepath}")
        else:
            logger.error(f"Model '{model_name}' not found")
