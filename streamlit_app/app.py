"""
Chicago Crime Analytics - Streamlit Dashboard
Multi-page interactive application for crime analysis visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Page configuration
st.set_page_config(
    page_title="Chicago Crime Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Chicago Crime Analytics - Urban Safety Intelligence Platform"
    }
)

# Custom CSS
st.markdown("""
    <style>
    /* Force light theme colors */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    /* Enhanced metric visibility */
    .stMetric {
        background-color: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .stMetric label {
        color: #262730 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    /* Force white background for HTML components */
    iframe {
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 5px !important;
    }
    .element-container iframe {
        background-color: white !important;
    }
    /* Ensure text is visible */
    p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data functions
@st.cache_data
def load_cleaned_data():
    """Load cleaned crime data"""
    data_path = project_root / 'data' / 'processed' / 'crimes_cleaned.csv'
    df = pd.read_csv(data_path, parse_dates=['Date'])
    return df

@st.cache_data
def load_features_data():
    """Load data with engineered features"""
    data_path = project_root / 'data' / 'processed' / 'crimes_with_features.csv'
    df = pd.read_csv(data_path, parse_dates=['Date'])
    return df

@st.cache_data
def get_summary_stats(df):
    """Calculate summary statistics"""
    return {
        'total_crimes': len(df),
        'crime_types': df['Primary Type'].nunique(),
        'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
        'districts': df['District'].nunique(),
        'arrest_rate': (df['Arrest'].sum() / len(df) * 100),
        'domestic_rate': (df['Domestic'].sum() / len(df) * 100)
    }

@st.cache_data
def load_clustering_results():
    """Load clustering results from summary file"""
    summary_path = project_root / 'outputs' / 'clustering' / 'clustering_summary.txt'
    results = {
        'kmeans': {'n_clusters': 8, 'silhouette': 0.1721, 'davies_bouldin': 1.9208},
        'dbscan': {'n_clusters': 0, 'noise': 50000, 'noise_ratio': 1.0, 'silhouette': None, 'davies_bouldin': None},
        'hierarchical': {'n_clusters': 5, 'silhouette': 0.1542, 'davies_bouldin': 1.7284}
    }
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            content = f.read()
            
            # Parse K-Means results
            if 'K-MEANS CLUSTERING' in content:
                kmeans_section = content.split('K-MEANS CLUSTERING')[1].split('2. DBSCAN')[0]
                if 'n_clusters:' in kmeans_section:
                    results['kmeans']['n_clusters'] = int(kmeans_section.split('n_clusters:')[1].split('\n')[0].strip())
                if 'silhouette_score:' in kmeans_section:
                    results['kmeans']['silhouette'] = float(kmeans_section.split('silhouette_score:')[1].split('\n')[0].strip())
                if 'davies_bouldin_score:' in kmeans_section:
                    results['kmeans']['davies_bouldin'] = float(kmeans_section.split('davies_bouldin_score:')[1].split('\n')[0].strip())
            
            # Parse DBSCAN results
            if 'DBSCAN CLUSTERING' in content:
                dbscan_section = content.split('DBSCAN CLUSTERING')[1].split('3. HIERARCHICAL')[0]
                if 'n_clusters:' in dbscan_section:
                    results['dbscan']['n_clusters'] = int(dbscan_section.split('n_clusters:')[1].split('\n')[0].strip())
                if 'n_noise:' in dbscan_section:
                    results['dbscan']['noise'] = int(dbscan_section.split('n_noise:')[1].split('\n')[0].strip())
                if 'noise_ratio:' in dbscan_section:
                    results['dbscan']['noise_ratio'] = float(dbscan_section.split('noise_ratio:')[1].split('\n')[0].strip())
                if 'silhouette_score:' in dbscan_section:
                    results['dbscan']['silhouette'] = float(dbscan_section.split('silhouette_score:')[1].split('\n')[0].strip())
                if 'davies_bouldin_score:' in dbscan_section:
                    results['dbscan']['davies_bouldin'] = float(dbscan_section.split('davies_bouldin_score:')[1].split('\n')[0].strip())
            
            # Parse Hierarchical results
            if 'HIERARCHICAL CLUSTERING' in content:
                hier_section = content.split('HIERARCHICAL CLUSTERING')[1].split('============================================================')[0]
                if 'n_clusters:' in hier_section:
                    results['hierarchical']['n_clusters'] = int(hier_section.split('n_clusters:')[1].split('\n')[0].strip())
                if 'silhouette_score:' in hier_section:
                    results['hierarchical']['silhouette'] = float(hier_section.split('silhouette_score:')[1].split('\n')[0].strip())
                if 'davies_bouldin_score:' in hier_section:
                    results['hierarchical']['davies_bouldin'] = float(hier_section.split('davies_bouldin_score:')[1].split('\n')[0].strip())
    
    return results

def load_html_with_background(file_path, height=700):
    """Load HTML file and wrap it with white background styling"""
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Wrap with styled div to ensure white background
    wrapped_html = f"""
    <div style="background-color: white; padding: 10px; border-radius: 5px;">
        {html_content}
    </div>
    """
    st.components.v1.html(wrapped_html, height=height, scrolling=True)

# Sidebar navigation
st.sidebar.title("🚔 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 EDA Dashboard", "🎯 Clustering Analysis", "📉 Dimensionality Reduction", "📈 Key Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Chicago Crime Analytics Platform**

Analyzing 496K+ crime records using:
- K-Means, DBSCAN, Hierarchical Clustering
- PCA, t-SNE Dimensionality Reduction
- MLflow Experiment Tracking
""")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<div class="main-header">🚔 Chicago Crime Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Urban Safety Intelligence Platform</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data for metrics
    df = load_cleaned_data()
    stats = get_summary_stats(df)
    
    # Key Metrics
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Crimes", f"{stats['total_crimes']:,}")
    with col2:
        st.metric("Crime Types", stats['crime_types'])
    with col3:
        st.metric("Police Districts", stats['districts'])
    with col4:
        st.metric("Arrest Rate", f"{stats['arrest_rate']:.1f}%")
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Goals")
        st.markdown("""
        - **Identify Crime Hotspots**: Geographic clustering of high-crime areas
        - **Temporal Pattern Analysis**: When crimes occur most frequently
        - **Crime Type Classification**: Categorize and analyze crime patterns
        - **Resource Optimization**: Support police deployment decisions
        - **Public Safety**: Enhance community awareness and prevention
        """)
        
        st.subheader("🔧 Technologies Used")
        st.markdown("""
        - **Python**: Data processing and ML
        - **Scikit-learn**: Clustering algorithms
        - **Plotly**: Interactive visualizations
        - **Streamlit**: Web dashboard
        - **MLflow**: Experiment tracking
        - **Pandas/NumPy**: Data manipulation
        """)
    
    with col2:
        st.subheader("📈 Analysis Pipeline")
        st.markdown("""
        1. **Data Acquisition** ✅
           - Loaded 2.7M records, sampled 500K
           - 99.3% data retention after cleaning
        
        2. **Exploratory Data Analysis** ✅
           - Crime type distribution
           - Temporal and geographic patterns
           - Arrest and domestic incident rates
        
        3. **Feature Engineering** ✅
           - 10 engineered features
           - Temporal, geographic, severity encoding
        
        4. **Clustering Analysis** ✅
           - K-Means (8 clusters, Silhouette: 0.172)
           - Hierarchical (5 clusters, DB: 1.728)
           - DBSCAN (parameter tuning needed)
        
        5. **Dimensionality Reduction** ✅
           - PCA: 72.5% variance with 5 components
           - t-SNE: 2D visualization
        """)
    
    st.markdown("---")
    
    # Date Range Info
    st.subheader("📅 Data Coverage")
    st.info(f"**Date Range:** {stats['date_range']}")
    
    # Quick Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Domestic Incidents", f"{stats['domestic_rate']:.1f}%")
    with col2:
        top_crime = df['Primary Type'].value_counts().index[0]
        st.metric("Most Common Crime", top_crime)
    with col3:
        peak_hour = df['Date'].dt.hour.value_counts().index[0]
        st.metric("Peak Crime Hour", f"{peak_hour}:00")

# ============================================================================
# EDA DASHBOARD
# ============================================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("---")
    
    df = load_cleaned_data()
    
    # Crime Type Distribution
    st.subheader("🔍 Crime Type Distribution")
    top_n = st.slider("Show Top N Crime Types", 5, 25, 15)
    
    crime_counts = df['Primary Type'].value_counts().head(top_n)
    fig = px.bar(
        x=crime_counts.values,
        y=crime_counts.index,
        orientation='h',
        labels={'x': 'Number of Crimes', 'y': 'Crime Type'},
        title=f'Top {top_n} Crime Types',
        color=crime_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Temporal Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Crimes by Hour of Day")
        hourly = df['Date'].dt.hour.value_counts().sort_index()
        fig = px.line(
            x=hourly.index,
            y=hourly.values,
            labels={'x': 'Hour of Day', 'y': 'Number of Crimes'},
            title='Crime Distribution by Hour',
            markers=True
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Midnight Peak")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Crimes by Day of Week")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = df['Date'].dt.dayofweek.value_counts().sort_index()
        fig = px.bar(
            x=[days[i] for i in daily.index],
            y=daily.values,
            labels={'x': 'Day of Week', 'y': 'Number of Crimes'},
            title='Crime Distribution by Day',
            color=daily.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ EDA Dashboard - Use sidebar to explore other pages")

# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================
elif page == "🎯 Clustering Analysis":
    st.title("🎯 Clustering Analysis Results")
    st.markdown("---")
    
    # Load clustering visualizations
    clustering_dir = project_root / 'outputs' / 'clustering'
    
    # Algorithm selector
    algorithm = st.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical"]
    )
    
    if algorithm == "K-Means":
        st.subheader("K-Means Clustering (k=8)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters", "8")
        with col2:
            st.metric("Silhouette Score", "0.1721")
        with col3:
            st.metric("Davies-Bouldin", "1.9208")
        with col4:
            st.metric("Samples", "50,000")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:

            st.subheader("📈 Elbow Curve")

            elbow_path = clustering_dir / 'kmeans_elbow_curve.html'
            if elbow_path.exists():
                load_html_with_background(elbow_path, height=700)
            else:
                st.warning("⚠️ Elbow curve not found. Run: `python scripts/04_clustering_analysis.py`")
        
        with col2:

            st.subheader("📊 Cluster Distribution")

            dist_path = clustering_dir / 'kmeans_distribution.html'
            if dist_path.exists():
                load_html_with_background(dist_path, height=700)
            else:
                st.warning("⚠️ Distribution chart not found.")
        

        st.markdown("---")
        st.subheader("🗺️ Geographic Cluster Heatmap")

        heatmap_path = clustering_dir / 'kmeans_geographic_heatmap.html'
        if heatmap_path.exists():
            load_html_with_background(heatmap_path, height=800)
        else:
            st.info("💡 Geographic heatmap shows crime hotspots by cluster")
    
    elif algorithm == "DBSCAN":
        # ::: Code Generated by Copilot [550e8400-e29b-41d4-a716-446655440031]. This comment will be removed automatically after the file is saved :::
        st.subheader("DBSCAN Clustering")
        
        # Load actual results
        cluster_results = load_clustering_results()
        dbscan_res = cluster_results['dbscan']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters Found", f"{dbscan_res['n_clusters']}")
        with col2:
            st.metric("Noise Points", f"{dbscan_res['noise']:,}")
        with col3:
            st.metric("Noise Ratio", f"{dbscan_res['noise_ratio']*100:.1f}%")
        with col4:
            if dbscan_res['silhouette']:
                st.metric("Silhouette Score", f"{dbscan_res['silhouette']:.4f}")
            else:
                st.metric("Silhouette Score", "N/A")
        
        if dbscan_res['n_clusters'] > 0:
            st.success(f"✅ **DBSCAN found {dbscan_res['n_clusters']} clusters** with only {dbscan_res['noise_ratio']*100:.1f}% noise!")
            if dbscan_res['davies_bouldin']:
                st.info(f"📊 Davies-Bouldin Score: {dbscan_res['davies_bouldin']:.4f} (Best performing algorithm!)")
        else:
            st.warning("""
            **⚠️ Parameter Tuning Needed**
            
            DBSCAN with current parameters classified all points as noise.
            Consider adjusting:
            - Increase `eps` (neighborhood radius)
            - Decrease `min_samples` (minimum cluster size)
            """)
        
        st.markdown("---")
        
        dist_path = clustering_dir / 'dbscan_distribution.html'
        if dist_path.exists():
            load_html_with_background(dist_path, height=700)
    
    else:  # Hierarchical
        st.subheader("Hierarchical Clustering (k=5)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters", "5")
        with col2:
            st.metric("Silhouette Score", "0.1542")
        with col3:
            st.metric("Davies-Bouldin", "1.7284", delta="-0.19 vs K-Means", delta_color="inverse")
        with col4:
            st.metric("Linkage", "Ward")
        
        st.success("✅ **Best Davies-Bouldin Score** - Hierarchical has the lowest score (1.7284)")
        
        st.markdown("---")
        

        dist_path = clustering_dir / 'hierarchical_distribution.html'
        if dist_path.exists():
            load_html_with_background(dist_path, height=700)
        else:
            st.warning("⚠️ Distribution chart not found.")
    
    st.markdown("---")
    

    # ::: Code Generated by Copilot [550e8400-e29b-41d4-a716-446655440032]. This comment will be removed automatically after the file is saved :::
    # Comparison Table
    st.subheader("📋 Algorithm Comparison")
    
    cluster_results = load_clustering_results()
    kmeans_res = cluster_results['kmeans']
    dbscan_res = cluster_results['dbscan']
    hier_res = cluster_results['hierarchical']
    
    comparison_df = pd.DataFrame({
        'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical'],
        'Clusters': [kmeans_res['n_clusters'], dbscan_res['n_clusters'], hier_res['n_clusters']],
        'Silhouette Score': [
            kmeans_res['silhouette'], 
            dbscan_res['silhouette'] if dbscan_res['silhouette'] else np.nan, 
            hier_res['silhouette']
        ],
        'Davies-Bouldin': [
            kmeans_res['davies_bouldin'], 
            dbscan_res['davies_bouldin'] if dbscan_res['davies_bouldin'] else np.nan, 
            hier_res['davies_bouldin']
        ],
        'Status': [
            '✅ Complete', 
            '✅ Complete' if dbscan_res['n_clusters'] > 0 else '⚠️ Needs Tuning', 
            '✅ Complete'
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================
elif page == "📉 Dimensionality Reduction":
    st.title("📉 Dimensionality Reduction Analysis")
    st.markdown("---")
    
    dimred_dir = project_root / 'outputs' / 'dimensionality_reduction'
    
    # Method selector
    method = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "t-SNE"]
    )
    
    if method == "PCA":
        st.subheader("PCA - Principal Component Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Features", "10")
        with col2:
            st.metric("Reduced Features", "5")
        with col3:
            st.metric("Variance Explained", "72.5%", delta="✅ Target: 70%")
        with col4:
            st.metric("Reduction", "50%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Scree Plot")
            scree_path = dimred_dir / 'pca_scree_plot.html'

            if scree_path.exists():
                load_html_with_background(scree_path, height=700)
            else:
                st.warning("⚠️ Run: `python scripts/05_dimensionality_reduction.py`")
        
        with col2:
            st.subheader("🔍 Component Loadings")
            loadings_path = dimred_dir / 'pca_loadings.html'

            if loadings_path.exists():
                load_html_with_background(loadings_path, height=700)
            else:
                st.info("Component loadings show feature importance")
        
        st.markdown("---")
        
        # Variance breakdown
        st.subheader("📈 Variance Explained by Component")
        variance_df = pd.DataFrame({
            'Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
            'Variance (%)': [20.04, 17.09, 12.54, 11.65, 11.20]
        })
        fig = px.bar(variance_df, x='Component', y='Variance (%)', 
                     title='Individual Component Variance',
                     color='Variance (%)', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 2D and 3D visualizations
        tab1, tab2 = st.tabs(["2D Projection", "3D Projection"])
        
        with tab1:

            pca_2d_path = dimred_dir / 'pca_2d_scatter.html'

            if pca_2d_path.exists():
                load_html_with_background(pca_2d_path, height=800)
            else:
                st.warning("⚠️ 2D scatter plot not found.")
        
        with tab2:

            pca_3d_path = dimred_dir / 'pca_3d_scatter.html'

            if pca_3d_path.exists():
                load_html_with_background(pca_3d_path, height=800)
            else:
                st.warning("⚠️ 3D scatter plot not found.")
    
    else:  # t-SNE
        st.subheader("t-SNE - T-Distributed Stochastic Neighbor Embedding")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reduced Dimensions", "2")
        with col2:
            st.metric("KL Divergence", "1.9828")
        with col3:
            st.metric("Iterations", "999")
        
        st.info("💡 t-SNE is ideal for non-linear visualization of high-dimensional crime patterns")
        
        st.markdown("---")
        

        tsne_path = dimred_dir / 'tsne_2d_scatter.html'

        if tsne_path.exists():
            load_html_with_background(tsne_path, height=800)
        else:
            st.warning("⚠️ t-SNE visualization not found.")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
elif page == "📈 Key Insights":
    st.title("📈 Key Insights & Recommendations")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Findings")
        st.markdown("""
        ### Crime Patterns
        - **Peak Time:** Midnight (0:00) - 33,645 crimes
        - **Peak Day:** Friday - 72,786 crimes
        - **Peak Month:** July - 46,449 crimes
        - **Most Common:** Theft (23.3%), Battery (17.9%)
        
        ### Geographic Insights
        - **Hotspot Districts:** 1, 4, 3 (26K-29K crimes each)
        - **224 Unique Grid Cells** identified
        - **Clustering:** 8 distinct crime zones (K-Means)
        
        ### Arrest Patterns
        - **Overall Arrest Rate:** 14.5%
        - **Highest Arrest Rates:** Gambling (97%), Narcotics (96%)
        - **Domestic Incidents:** 18.6% of all crimes
        
        ### Dimensionality
        - **PCA Success:** 72.5% variance with 5 components
        - **Top Features:** Geographic (Lat/Lon), Temporal (Hour, Day)
        - **50% reduction** in feature space
        """)
    
    with col2:
        st.subheader("💡 Recommendations")
        st.markdown("""
        ### Police Resource Allocation
        1. **Increase Patrols:**
           - Midnight to 3 AM shift
           - Friday-Saturday nights
           - Districts 1, 3, 4
        
        2. **Seasonal Planning:**
           - Enhanced summer deployment (June-August)
           - Winter reduction in non-essential patrols
        
        ### Prevention Strategies
        1. **Hotspot Policing:**
           - Focus on 8 identified K-Means clusters
           - Predictive deployment based on patterns
        
        2. **Community Programs:**
           - Theft prevention workshops
           - Domestic violence intervention
           - Youth engagement in high-crime areas
        
        ### Technology Integration
        1. **Real-time Monitoring:**
           - Dashboard for live crime tracking
           - Alert system for hotspot activation
        
        2. **Data-Driven Decisions:**
           - Evidence-based policy making
           - Performance metrics tracking
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Next Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Short Term**
        - Deploy real-time dashboard ✅
        - Train officers on insights
        - Implement hotspot patrols
        """)
    
    with col2:
        st.success("""
        **Medium Term**
        - Add predictive models
        - Integrate weather data
        - Expand to more districts
        """)
    
    with col3:
        st.warning("""
        **Long Term**
        - City-wide analytics platform
        - Mobile app for officers
        - Public safety portal
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p><strong>Chicago Crime Analytics Platform</strong></p>
    <p>Built with Streamlit | Powered by Scikit-learn & MLflow</p>
    <p>© 2025 Urban Safety Intelligence</p>
</div>
""", unsafe_allow_html=True)
