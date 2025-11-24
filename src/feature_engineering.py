"""
Feature Engineering Module for Chicago Crime Analytics
Creates temporal, geographic, and derived features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeFeatureEngineer:
    """
    Feature engineering for crime data analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            df: Input DataFrame with crime data
        """
        self.df = df.copy()
        
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Extract temporal features from datetime field
        
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Creating temporal features...")
        
        # Convert Date column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Extract temporal components
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Hour'] = self.df['Date'].dt.hour
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        
        # Create binary flags
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        self.df['IsNight'] = self.df['Hour'].between(20, 5).astype(int)  # 8PM to 5AM
        
        # Create season feature
        self.df['Season'] = self.df['Month'].apply(self._get_season)
        
        # Create time period categories
        self.df['TimePeriod'] = self.df['Hour'].apply(self._get_time_period)
        
        logger.info("Temporal features created")
        return self.df
    
    def create_geographic_features(self) -> pd.DataFrame:
        """
        Create geographic features from coordinates
        
        Returns:
            DataFrame with geographic features added
        """
        logger.info("Creating geographic features...")
        
        # Coordinate binning for spatial analysis
        self.df['Lat_Bin'] = pd.cut(self.df['Latitude'], bins=20, labels=False)
        self.df['Lon_Bin'] = pd.cut(self.df['Longitude'], bins=20, labels=False)
        
        # Create grid cell identifier
        self.df['GridCell'] = (
            self.df['Lat_Bin'].astype(str) + '_' + 
            self.df['Lon_Bin'].astype(str)
        )
        
        logger.info("Geographic features created")
        return self.df
    
    def create_crime_severity_score(self) -> pd.DataFrame:
        """
        Create crime severity score based on crime type
        
        Returns:
            DataFrame with severity scores added
        """
        logger.info("Creating crime severity scores...")
        
        # Crime severity mapping (1=low, 5=high)
        severity_map = {
            'HOMICIDE': 5,
            'CRIMINAL SEXUAL ASSAULT': 5,
            'ROBBERY': 4,
            'ASSAULT': 4,
            'BATTERY': 4,
            'BURGLARY': 3,
            'THEFT': 3,
            'MOTOR VEHICLE THEFT': 3,
            'CRIMINAL DAMAGE': 2,
            'NARCOTICS': 2,
            'WEAPONS VIOLATION': 4,
            'PUBLIC PEACE VIOLATION': 1,
            'DECEPTIVE PRACTICE': 2,
            'OTHER OFFENSE': 1
        }
        
        self.df['CrimeSeverity'] = self.df['Primary Type'].map(severity_map).fillna(2)
        
        logger.info("Crime severity scores created")
        return self.df
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features for ML models
        
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        # One-hot encode crime type (top 10 most frequent)
        top_crime_types = self.df['Primary Type'].value_counts().head(10).index
        self.df['CrimeType_Top10'] = self.df['Primary Type'].where(
            self.df['Primary Type'].isin(top_crime_types), 
            'OTHER'
        )
        
        # Encode arrest flag
        if 'Arrest' in self.df.columns:
            self.df['Arrest'] = self.df['Arrest'].astype(int)
            
        # Encode domestic flag
        if 'Domestic' in self.df.columns:
            self.df['Domestic'] = self.df['Domestic'].astype(int)
        
        logger.info("Categorical encoding completed")
        return self.df
    
    def get_feature_matrix(self, features: list = None) -> pd.DataFrame:
        """
        Get feature matrix for ML models
        
        Args:
            features: List of feature names to include
            
        Returns:
            Feature matrix DataFrame
        """
        if features is None:
            # Default feature set for clustering
            features = [
                'Latitude', 'Longitude', 'Hour', 'DayOfWeek', 
                'Month', 'IsWeekend', 'IsNight', 'CrimeSeverity',
                'Arrest', 'Domestic'
            ]
        
        return self.df[features].copy()
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    @staticmethod
    def _get_time_period(hour: int) -> str:
        """Get time period from hour"""
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
