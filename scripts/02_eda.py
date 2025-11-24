"""
Chicago Crime Analytics - Exploratory Data Analysis Script
Comprehensive EDA with visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_cleaned_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned crime data"""
    logger.info(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"✓ Loaded {len(df):,} records")
    return df

def analyze_crime_types(df: pd.DataFrame, output_dir: Path):
    """Analyze crime type distribution"""
    logger.info("\nAnalyzing crime type distribution...")
    
    # Crime type counts
    crime_counts = df['Primary Type'].value_counts()
    
    logger.info(f"\nTop 15 Crime Types:")
    for i, (crime, count) in enumerate(crime_counts.head(15).items(), 1):
        logger.info(f"  {i}. {crime}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Plot crime distribution
    fig = px.bar(
        x=crime_counts.head(15).index,
        y=crime_counts.head(15).values,
        title='Top 15 Crime Types in Chicago',
        labels={'x': 'Crime Type', 'y': 'Count'},
        color=crime_counts.head(15).values,
        color_continuous_scale='Reds'
    )
    fig.write_html(output_dir / 'crime_types_distribution.html')
    logger.info(f"✓ Saved: crime_types_distribution.html")

def analyze_temporal_patterns(df: pd.DataFrame, output_dir: Path):
    """Analyze temporal crime patterns"""
    logger.info("\nAnalyzing temporal patterns...")
    
    # Extract temporal features
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Hourly distribution
    hourly_crimes = df['Hour'].value_counts().sort_index()
    fig1 = px.line(
        x=hourly_crimes.index,
        y=hourly_crimes.values,
        title='Crime Distribution by Hour of Day',
        labels={'x': 'Hour', 'y': 'Number of Crimes'},
        markers=True
    )
    fig1.write_html(output_dir / 'hourly_crime_pattern.html')
    logger.info(f"✓ Peak crime hour: {hourly_crimes.idxmax()}:00 ({hourly_crimes.max():,} crimes)")
    
    # Day of week distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_crimes = df['DayOfWeek'].value_counts().reindex(day_order)
    fig2 = px.bar(
        x=daily_crimes.index,
        y=daily_crimes.values,
        title='Crime Distribution by Day of Week',
        labels={'x': 'Day', 'y': 'Number of Crimes'},
        color=daily_crimes.values,
        color_continuous_scale='Blues'
    )
    fig2.write_html(output_dir / 'daily_crime_pattern.html')
    logger.info(f"✓ Peak crime day: {daily_crimes.idxmax()} ({daily_crimes.max():,} crimes)")
    
    # Monthly distribution
    monthly_crimes = df['Month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig3 = px.bar(
        x=[month_names[i-1] for i in monthly_crimes.index],
        y=monthly_crimes.values,
        title='Crime Distribution by Month',
        labels={'x': 'Month', 'y': 'Number of Crimes'},
        color=monthly_crimes.values,
        color_continuous_scale='Greens'
    )
    fig3.write_html(output_dir / 'monthly_crime_pattern.html')
    logger.info(f"✓ Peak crime month: {month_names[monthly_crimes.idxmax()-1]} ({monthly_crimes.max():,} crimes)")

def analyze_geographic_patterns(df: pd.DataFrame, output_dir: Path):
    """Analyze geographic crime distribution"""
    logger.info("\nAnalyzing geographic patterns...")
    
    # District distribution
    if 'District' in df.columns:
        district_crimes = df['District'].value_counts().sort_index()
        logger.info(f"\nCrime by District (Top 5):")
        for district, count in district_crimes.head().items():
            logger.info(f"  District {district}: {count:,} crimes")
    
    # Create crime heatmap
    fig = px.density_mapbox(
        df.sample(min(50000, len(df))),  # Sample for performance
        lat='Latitude',
        lon='Longitude',
        radius=10,
        zoom=10,
        height=600,
        title='Chicago Crime Density Heatmap',
        mapbox_style='open-street-map'
    )
    fig.write_html(output_dir / 'crime_heatmap.html')
    logger.info(f"✓ Saved: crime_heatmap.html")

def analyze_arrest_rates(df: pd.DataFrame):
    """Analyze arrest rates and domestic incidents"""
    logger.info("\nAnalyzing arrest and domestic incident rates...")
    
    if 'Arrest' in df.columns:
        arrest_rate = df['Arrest'].sum() / len(df) * 100
        logger.info(f"  Overall arrest rate: {arrest_rate:.2f}%")
        
        # Arrest rate by crime type
        arrest_by_crime = df.groupby('Primary Type')['Arrest'].mean().sort_values(ascending=False).head(10)
        logger.info(f"\n  Top 5 Crime Types by Arrest Rate:")
        for crime, rate in arrest_by_crime.head().items():
            logger.info(f"    {crime}: {rate*100:.2f}%")
    
    if 'Domestic' in df.columns:
        domestic_rate = df['Domestic'].sum() / len(df) * 100
        logger.info(f"\n  Domestic incident rate: {domestic_rate:.2f}%")

def generate_summary_statistics(df: pd.DataFrame):
    """Generate comprehensive summary statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"\nDataset Overview:")
    logger.info(f"  Total crimes: {len(df):,}")
    logger.info(f"  Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    logger.info(f"  Unique crime types: {df['Primary Type'].nunique()}")
    logger.info(f"  Geographic coverage: {df['District'].nunique() if 'District' in df.columns else 'N/A'} districts")
    
    logger.info("=" * 60)

def main():
    """Main EDA pipeline"""
    # Define paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'crimes_cleaned.csv'
    output_dir = project_root / 'outputs' / 'eda'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        df = load_cleaned_data(data_path)
        
        # Run analyses
        analyze_crime_types(df, output_dir)
        analyze_temporal_patterns(df, output_dir)
        analyze_geographic_patterns(df, output_dir)
        analyze_arrest_rates(df)
        generate_summary_statistics(df)
        
        logger.info(f"\n✓ EDA complete! Visualizations saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in EDA pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
