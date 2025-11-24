"""
Chicago Crime Analytics - Data Acquisition & Sampling Script
Loads, samples, and preprocesses 500,000 crime records from the Chicago Crime Dataset
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from data_preprocessing import CrimeDataPreprocessor
except ImportError:
    logger.error("Could not import data_preprocessing module. Make sure src/data_preprocessing.py exists.")
    sys.exit(1)

def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the full crime dataset"""
    logger.info(f"Loading dataset from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    return df

def explore_dataset(df: pd.DataFrame):
    """Explore dataset structure and quality"""
    logger.info("=" * 60)
    logger.info("DATASET EXPLORATION")
    logger.info("=" * 60)
    
    # Basic info
    logger.info(f"\nTotal records: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info("\nMissing Values:")
        for col, count in missing[missing > 0].items():
            logger.info(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Date range
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    logger.info(f"\nDate Range:")
    logger.info(f"  Earliest: {df['Date'].min()}")
    logger.info(f"  Latest: {df['Date'].max()}")
    logger.info(f"  Span: {(df['Date'].max() - df['Date'].min()).days} days")
    
    # Crime types
    logger.info(f"\nUnique Crime Types: {df['Primary Type'].nunique()}")
    logger.info("\nTop 10 Crime Types:")
    crime_counts = df['Primary Type'].value_counts().head(10)
    for crime, count in crime_counts.items():
        logger.info(f"  {crime}: {count:,} ({count/len(df)*100:.2f}%)")
    
    logger.info("=" * 60)

def sample_recent_records(df: pd.DataFrame, sample_size: int = 500000) -> pd.DataFrame:
    """Sample most recent crime records"""
    logger.info(f"\nSampling {sample_size:,} most recent records...")
    
    # Sort by date and take most recent
    df_sampled = df.sort_values('Date', ascending=False).head(sample_size).copy()
    
    logger.info(f"✓ Sample created successfully!")
    logger.info(f"  Date range: {df_sampled['Date'].min()} to {df_sampled['Date'].max()}")
    logger.info(f"  Records: {len(df_sampled):,}")
    
    return df_sampled

def clean_and_validate(df: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    """Clean and validate the sampled data"""
    logger.info("\nStarting data cleaning...")
    
    initial_count = len(df)
    
    # Initialize preprocessor
    preprocessor = CrimeDataPreprocessor(data_path)
    preprocessor.df = df.copy()
    
    # Clean data
    df_cleaned = preprocessor.clean_data()
    
    logger.info(f"\nCleaning Summary:")
    logger.info(f"  Original records: {initial_count:,}")
    logger.info(f"  Cleaned records: {len(df_cleaned):,}")
    logger.info(f"  Records removed: {initial_count - len(df_cleaned):,}")
    logger.info(f"  Retention rate: {len(df_cleaned)/initial_count*100:.2f}%")
    
    # Validate critical fields
    critical_fields = ['Latitude', 'Longitude', 'Primary Type', 'Date']
    logger.info("\nValidation - Missing values in critical fields:")
    for field in critical_fields:
        missing = df_cleaned[field].isnull().sum()
        logger.info(f"  {field}: {missing} ({missing/len(df_cleaned)*100:.2f}%)")
    
    # Validate coordinates
    logger.info("\nGeographic Coordinate Validation:")
    logger.info(f"  Latitude range: {df_cleaned['Latitude'].min():.4f} to {df_cleaned['Latitude'].max():.4f}")
    logger.info(f"  Longitude range: {df_cleaned['Longitude'].min():.4f} to {df_cleaned['Longitude'].max():.4f}")
    logger.info(f"  Expected Chicago bounds: Lat [41.6, 42.1], Lon [-87.9, -87.5]")
    
    return df_cleaned

def save_processed_data(df: pd.DataFrame, output_path: Path):
    """Save cleaned data to processed folder"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving cleaned data to: {output_path}")
    df.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Data saved successfully! ({file_size_mb:.2f} MB)")

def main():
    """Main execution pipeline"""
    # Define paths
    root = Path(__file__).parent.parent
    data_path = root / 'data' / 'raw' / 'Crimes_-_2001_to_Present_20251122.csv'
    output_path = project_root / 'data' / 'processed' / 'crimes_cleaned.csv'
    
    try:
        # Step 1: Load dataset
        df_full = load_dataset(data_path)
        
        # Step 2: Explore dataset
        explore_dataset(df_full)
        
        # Step 3: Sample 500k recent records
        df_sampled = sample_recent_records(df_full, sample_size=500000)
        
        # Step 4: Clean and validate
        df_cleaned = clean_and_validate(df_sampled, data_path)
        
        # Step 5: Save processed data
        save_processed_data(df_cleaned, output_path)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("DATA ACQUISITION & CLEANING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Records processed: {len(df_cleaned):,}")
        logger.info(f"✓ Unique crime types: {df_cleaned['Primary Type'].nunique()}")
        logger.info(f"✓ Geographic coverage: {df_cleaned['District'].nunique()} police districts")
        logger.info(f"✓ Output: {output_path}")
        logger.info(f"✓ Ready for EDA and feature engineering!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
