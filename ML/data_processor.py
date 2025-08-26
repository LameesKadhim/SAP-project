"""
Data processing module for the Backend.
"""

import pandas as pd
import logging
from typing import Tuple, Optional
from config import DATASET_PATH, DROP_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and preprocessing for model training."""
    
    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        self.df = None
        
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
    
    def preprocess_data(self) -> Optional[pd.DataFrame]:
        """
        Preprocess the loaded dataset.
        
        Returns:
            pd.DataFrame: Preprocessed dataset or None if error
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            logger.info("Preprocessing dataset...")
            
            # Drop unnecessary columns
            self.df = self.df.drop(DROP_COLUMNS, axis=1)
            
            # Check for null values
            null_counts = self.df.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"Found null values: {null_counts.to_dict()}")
            
            logger.info("Data preprocessing completed successfully")
            return self.df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def prepare_features_target(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features (X) and target (y) for model training.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target or (None, None) if error
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None, None
        
        try:
            logger.info("Preparing features and target variables")
            
            # Prepare features
            X = self.df[FEATURE_COLUMNS]
            y = self.df[TARGET_COLUMN]
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
            
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


def create_data_processor(dataset_path: str = DATASET_PATH) -> DataProcessor:
    """
    Factory function to create a DataProcessor instance.
    
    Args:
        dataset_path (str): Path to the dataset file
        
    Returns:
        DataProcessor: Initialized data processor
    """
    return DataProcessor(dataset_path)
