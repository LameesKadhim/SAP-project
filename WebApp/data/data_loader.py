"""
Data loading and preprocessing module for the Student Admission Prediction application.
"""

import pandas as pd
import logging
from typing import Tuple, Optional
from config.settings import DATASET_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing operations."""
    
    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load the dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset or None if loading fails
        """
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Successfully loaded dataset with shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {self.dataset_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None
    
    def get_sample_data(self, n_rows: int = 5) -> Optional[pd.DataFrame]:
        """
        Get a sample of the dataset for display purposes.
        
        Args:
            n_rows (int): Number of rows to return
            
        Returns:
            pd.DataFrame: Sample dataset or None if data not loaded
        """
        if self.df is None:
            logger.warning("Dataset not loaded. Call load_data() first.")
            return None
        
        try:
            return self.df.head(n_rows)
        except Exception as e:
            logger.error(f"Error getting sample data: {str(e)}")
            return None
    
    def prepare_features_target(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features (X) and target (y) for model training/prediction.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target or (None, None) if error
        """
        if self.df is None:
            logger.warning("Dataset not loaded. Call load_data() first.")
            return None, None
        
        try:
            logger.info("Preparing features and target variables")
            
            # Drop unnecessary columns
            self.X = self.df.drop(['Chance of Admit', 'Serial No.'], axis=1)
            self.y = self.df['Chance of Admit']
            
            logger.info(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")
            return self.X, self.y
            
        except KeyError as e:
            logger.error(f"Required column not found: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error preparing features and target: {str(e)}")
            return None, None
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            dict: Dataset information
        """
        if self.df is None:
            return {}
        
        try:
            return {
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'null_counts': self.df.isnull().sum().to_dict(),
                'dtypes': self.df.dtypes.to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            return {}


def create_data_loader() -> DataLoader:
    """
    Factory function to create and initialize a DataLoader instance.
    
    Returns:
        DataLoader: Initialized data loader
    """
    data_loader = DataLoader()
    data_loader.load_data()
    return data_loader
