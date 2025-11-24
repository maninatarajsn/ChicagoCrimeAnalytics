"""
Visualization Module for Chicago Crime Analytics
Creates interactive plots and dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeVisualizer:
    """
    Visualization tools for crime analytics
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize visualizer
        
        Args:
            df: DataFrame with crime data and cluster labels
        """
        self.df = df.copy()
        
    def plot_crime_heatmap(self, cluster_labels: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create interactive geographic heatmap of crimes
        
        Args:
            cluster_labels: Optional cluster labels for coloring
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating crime heatmap...")
        
        if cluster_labels is not None:
            self.df['Cluster'] = cluster_labels
            fig = px.scatter_mapbox(
                self.df,
                lat='Latitude',
                lon='Longitude',
                color='Cluster',
                hover_data=['Primary Type', 'Date'],
                zoom=10,
                height=600,
                title='Crime Hotspot Clusters'
            )
        else:
            fig = px.density_mapbox(
                self.df,
                lat='Latitude',
                lon='Longitude',
                zoom=10,
                height=600,
                title='Crime Density Heatmap'
            )
        
        fig.update_layout(mapbox_style='open-street-map')
        return fig
    
    def plot_temporal_patterns(self, groupby: str = 'Hour') -> go.Figure:
        """
        Plot temporal crime patterns
        
        Args:
            groupby: Time dimension to group by ('Hour', 'DayOfWeek', 'Month')
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Plotting temporal patterns by {groupby}...")
        
        crime_counts = self.df.groupby(groupby).size().reset_index(name='Count')
        
        fig = px.bar(
            crime_counts,
            x=groupby,
            y='Count',
            title=f'Crime Distribution by {groupby}',
            labels={'Count': 'Number of Crimes'}
        )
        
        return fig
    
    def plot_cluster_distribution(self, cluster_labels: np.ndarray) -> go.Figure:
        """
        Plot cluster size distribution
        
        Args:
            cluster_labels: Cluster assignment labels
            
        Returns:
            Plotly figure object
        """
        logger.info("Plotting cluster distribution...")
        
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster ID', 'y': 'Number of Crimes'},
            title='Crime Distribution Across Clusters'
        )
        
        return fig
    
    def plot_elbow_curve(self, metrics: dict) -> go.Figure:
        """
        Plot elbow curve for optimal K selection
        
        Args:
            metrics: Dictionary with k_values and inertias
            
        Returns:
            Plotly figure object
        """
        logger.info("Plotting elbow curve...")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics['k_values'],
            y=metrics['inertias'],
            mode='lines+markers',
            name='Inertia'
        ))
        
        fig.update_layout(
            title='Elbow Method for Optimal K',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia',
            showlegend=True
        )
        
        return fig
    
    def plot_silhouette_analysis(self, metrics: dict) -> go.Figure:
        """
        Plot silhouette scores across different K values
        
        Args:
            metrics: Dictionary with k_values and silhouettes
            
        Returns:
            Plotly figure object
        """
        logger.info("Plotting silhouette analysis...")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics['k_values'],
            y=metrics['silhouettes'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Silhouette Score Analysis',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            showlegend=True
        )
        
        return fig
