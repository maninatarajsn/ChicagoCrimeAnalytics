"""
Data Preprocessing Module for Chicago Crime Analytics
Handles data loading, cleaning, and initial transformations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeDataPreprocessor:
    """
    Preprocessor for Chicago Crime Dataset
    """
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with data path
        
        Args:
            data_path: Path to the raw crime dataset
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load crime data from CSV file
        
        Args:
            sample_size: Number of records to sample (default: all)
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if sample_size:
            # Load with sampling for large datasets
            self.df = pd.read_csv(self.data_path, nrows=sample_size)
            logger.info(f"Loaded {sample_size} sampled records")
        else:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
            
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(self.df)} duplicate records")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Clean geographic coordinates
        self._clean_coordinates()
        
        # Standardize text columns
        self._standardize_text()
        
        logger.info("Data cleaning completed")
        return self.df
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Remove rows with missing critical fields
        critical_fields = ['Latitude', 'Longitude', 'Primary Type', 'Date']
        self.df = self.df.dropna(subset=critical_fields)
        
        # Fill other missing values appropriately
        if 'Location Description' in self.df.columns:
            self.df['Location Description'].fillna('UNKNOWN', inplace=True)
            
        logger.info("Missing values handled")
    
    def _clean_coordinates(self):
        """Clean and validate geographic coordinates"""
        # Remove invalid coordinates (outside Chicago bounds)
        chicago_lat_bounds = (41.6, 42.1)
        chicago_lon_bounds = (-87.9, -87.5)
        
        self.df = self.df[
            (self.df['Latitude'].between(*chicago_lat_bounds)) &
            (self.df['Longitude'].between(*chicago_lon_bounds))
        ]
        
        logger.info("Geographic coordinates cleaned")
    
    def _standardize_text(self):
        """Standardize text columns"""
        text_columns = ['Primary Type', 'Location Description', 'Description']
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.upper().str.strip()
                
        logger.info("Text columns standardized")
    
    def save_processed_data(self, output_path: str):
        """
        Save processed data to CSV
        
        Args:
            output_path: Path to save processed data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
