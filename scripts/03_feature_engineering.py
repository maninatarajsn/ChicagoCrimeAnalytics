"""
Feature Engineering Script
Creates temporal, geographic, and crime severity features for clustering analysis
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from feature_engineering import CrimeFeatureEngineer
    from data_preprocessing import CrimeDataPreprocessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure project structure is correct and modules are in: {project_root / 'src'}")
    sys.exit(1)

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_cleaned_data():
    """Load the cleaned crime data"""
    data_path = project_root / 'data' / 'processed' / 'crimes_cleaned.csv'
    logger.info(f"Loading cleaned data from: {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['Date'])
    logger.info(f"✓ Loaded {len(df):,} records")
    
    return df

def create_features(df):
    """Create all engineered features"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)
    
    feature_engineer = CrimeFeatureEngineer(df)
    
    # 1. Temporal Features
    logger.info("\n1. Creating temporal features...")
    df = feature_engineer.create_temporal_features()
    logger.info(f"   ✓ Created: Hour, DayOfWeek, Month, Season, IsWeekend, IsNight")
    logger.info(f"   ✓ Night crimes: {df['IsNight'].sum():,} ({df['IsNight'].mean()*100:.1f}%)")
    logger.info(f"   ✓ Weekend crimes: {df['IsWeekend'].sum():,} ({df['IsWeekend'].mean()*100:.1f}%)")
    
    # 2. Geographic Features
    logger.info("\n2. Creating geographic features...")
    df = feature_engineer.create_geographic_features()
    logger.info(f"   ✓ Created: Lat_Bin, Lon_Bin, GridCell")
    logger.info(f"   ✓ Grid resolution: 20x20 cells")
    logger.info(f"   ✓ Unique grid cells: {df['GridCell'].nunique():,}")
    
    # 3. Crime Severity Score
    logger.info("\n3. Creating crime severity scores...")
    df = feature_engineer.create_crime_severity_score()
    logger.info(f"   ✓ Severity distribution:")
    severity_counts = df['CrimeSeverity'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        logger.info(f"      Level {severity}: {count:,} crimes ({count/len(df)*100:.1f}%)")
    
    # 4. Encode Categorical Features
    logger.info("\n4. Encoding categorical features...")
    df = feature_engineer.encode_categorical_features()
    logger.info(f"   ✓ Crime types grouped to top 10 + OTHER")
    logger.info(f"   ✓ Arrest flag encoded: {df['Arrest'].sum():,} arrests")
    logger.info(f"   ✓ Domestic flag encoded: {df['Domestic'].sum():,} domestic incidents")
    
    return df

def validate_features(df):
    """Validate engineered features"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE VALIDATION")
    logger.info("="*60)
    
    # Check for missing values in new features
    temporal_features = ['Hour', 'DayOfWeek', 'Month', 'Season', 'IsWeekend', 'IsNight']
    geographic_features = ['Lat_Bin', 'Lon_Bin', 'GridCell']
    severity_features = ['CrimeSeverity']
    encoded_features = ['CrimeType_Top10', 'Arrest', 'Domestic']
    
    all_features = temporal_features + geographic_features + severity_features + encoded_features
    
    logger.info("\nMissing values in engineered features:")
    missing = df[all_features].isnull().sum()
    if missing.sum() == 0:
        logger.info("   ✓ No missing values in any engineered features!")
    else:
        for feature, count in missing[missing > 0].items():
            logger.info(f"   ⚠ {feature}: {count} ({count/len(df)*100:.2f}%)")
    
    # Feature value ranges
    logger.info("\nFeature value ranges:")
    logger.info(f"   Hour: {df['Hour'].min()} to {df['Hour'].max()}")
    logger.info(f"   DayOfWeek: {df['DayOfWeek'].min()} to {df['DayOfWeek'].max()}")
    logger.info(f"   Month: {df['Month'].min()} to {df['Month'].max()}")
    logger.info(f"   CrimeSeverity: {df['CrimeSeverity'].min()} to {df['CrimeSeverity'].max()}")
    logger.info(f"   GridCell: {df['GridCell'].nunique()} unique cells")
    
    return True

def prepare_clustering_features(df):
    """Prepare final feature set for clustering"""
    logger.info("\n" + "="*60)
    logger.info("CLUSTERING FEATURE PREPARATION")
    logger.info("="*60)
    
    # Select features for clustering
    clustering_features = [
        'Latitude', 'Longitude',  # Geographic
        'Hour', 'DayOfWeek', 'Month',  # Temporal
        'IsWeekend', 'IsNight',  # Temporal flags
        'CrimeSeverity',  # Severity
        'Arrest',  # Arrest flag
        'Domestic'  # Domestic flag
    ]
    
    # Check all features exist
    missing_cols = [col for col in clustering_features if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"\n✓ Selected {len(clustering_features)} features for clustering:")
    for i, feature in enumerate(clustering_features, 1):
        logger.info(f"   {i}. {feature}")
    
    # Create feature matrix
    X = df[clustering_features].copy()
    
    logger.info(f"\n✓ Feature matrix shape: {X.shape}")
    logger.info(f"✓ Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return X, clustering_features

def save_engineered_data(df, feature_names):
    """Save data with engineered features"""
    # Save full dataset with all features
    output_path = project_root / 'data' / 'processed' / 'crimes_with_features.csv'
    logger.info(f"\nSaving engineered dataset to: {output_path}")
    
    df.to_csv(output_path, index=False)
    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Full dataset saved! ({file_size:.2f} MB)")
    
    # Save feature matrix for clustering
    clustering_output = project_root / 'data' / 'processed' / 'clustering_features.csv'
    logger.info(f"\nSaving clustering features to: {clustering_output}")
    
    X = df[feature_names].copy()
    X.to_csv(clustering_output, index=False)
    file_size = clustering_output.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Clustering features saved! ({file_size:.2f} MB)")
    
    # Save feature metadata
    metadata_path = project_root / 'data' / 'processed' / 'feature_metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write("Feature Engineering Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Total features: {len(feature_names)}\n\n")
        f.write("Clustering Features:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"{i}. {feature}\n")
        f.write(f"\nDataset shape: {df.shape}\n")
        f.write(f"Feature matrix shape: {X.shape}\n")
    
    logger.info(f"✓ Metadata saved: {metadata_path}")

def main():
    """Main feature engineering pipeline"""
    try:
        # Load cleaned data
        df = load_cleaned_data()
        
        # Create features
        df = create_features(df)
        
        # Validate features
        validate_features(df)
        
        # Prepare clustering features
        X, feature_names = prepare_clustering_features(df)
        
        # Save engineered data
        save_engineered_data(df, feature_names)
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Total features created: {len(feature_names)}")
        logger.info(f"✓ Dataset ready for clustering analysis")
        logger.info(f"✓ Next step: Run 04_clustering_analysis.py")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
