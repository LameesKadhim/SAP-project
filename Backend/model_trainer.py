"""
Model training module for the Backend.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from config import (
    MODEL_SAVE_PATH, TEST_SIZE, RANDOM_STATE, SHUFFLE, 
    MODEL_TYPE, MODEL_PARAMS, FEATURE_COLUMNS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model_save_path: str = MODEL_SAVE_PATH):
        self.model_save_path = model_save_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Split data into training and test sets.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Splitting data into training and test sets")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RANDOM_STATE
            )
            
            logger.info(f"Training set shape: {self.X_train.shape}")
            logger.info(f"Test set shape: {self.X_test.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the RandomForest model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Training RandomForest model...")
            
            self.model = RandomForestRegressor(**MODEL_PARAMS)
            self.model.fit(self.X_train, self.y_train)
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def evaluate_model(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate the trained model.
        
        Returns:
            dict: Evaluation metrics or None if error
        """
        if self.model is None:
            logger.error("No model trained. Call train_model() first.")
            return None
        
        try:
            logger.info("Evaluating model...")
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate score
            score = self.model.score(self.X_test, self.y_test)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            evaluation_results = {
                'score': score,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actual': self.y_test.values
            }
            
            logger.info(f"Model evaluation completed. Score: {score:.4f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return None
    
    def save_model(self) -> bool:
        """
        Save the trained model to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return False
        
        try:
            logger.info(f"Saving model to {self.model_save_path}")
            joblib.dump(self.model, self.model_save_path)
            logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def test_model_loading(self) -> bool:
        """
        Test loading the saved model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Testing model loading...")
            loaded_model = joblib.load(self.model_save_path)
            
            if self.X_test is not None and self.y_test is not None:
                test_score = loaded_model.score(self.X_test, self.y_test)
                logger.info(f"Loaded model test score: {test_score:.4f}")
            
            logger.info("Model loading test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error testing model loading: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {}
        
        try:
            return {
                'model_type': type(self.model).__name__,
                'n_estimators': getattr(self.model, 'n_estimators', 'N/A'),
                'feature_importances': dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}


def create_model_trainer(model_save_path: str = MODEL_SAVE_PATH) -> ModelTrainer:
    """
    Factory function to create a ModelTrainer instance.
    
    Args:
        model_save_path (str): Path to save the model
        
    Returns:
        ModelTrainer: Initialized model trainer
    """
    return ModelTrainer(model_save_path)
