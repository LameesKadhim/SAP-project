"""
Model management module for the Student Admission Prediction application.
"""

import joblib
import numpy as np
import logging
from typing import Optional, Tuple
from config.settings import MODEL_PATH, FEATURE_NAMES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model loading and prediction operations."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the trained model from file.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        """
        Make a prediction using the loaded model.
        
        Args:
            features (np.ndarray): Input features for prediction
            
        Returns:
            float: Prediction result or None if error
        """
        if not self.is_loaded or self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Ensure features are in the correct shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            logger.info(f"Prediction made: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance from the model.
        
        Returns:
            dict: Feature importance dictionary or None if error
        """
        if not self.is_loaded or self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(FEATURE_NAMES, self.model.feature_importances_))
                logger.info("Feature importance retrieved successfully")
                return importance_dict
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return None
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def predict_admission_probability(self, gre_score: float, toefl_score: float, 
                                    university_rating: float, sop: float, 
                                    lor: float, cgpa: float, research: float) -> Optional[float]:
        """
        Predict admission probability with formatted input.
        
        Args:
            gre_score (float): GRE score
            toefl_score (float): TOEFL score
            university_rating (float): University rating
            sop (float): Statement of Purpose score
            lor (float): Letter of Recommendation score
            cgpa (float): CGPA
            research (float): Research experience (0 or 1)
            
        Returns:
            float: Admission probability (0-1) or None if error
        """
        try:
            # Create feature array in the correct order
            features = np.array([
                gre_score, toefl_score, university_rating, 
                sop, lor, cgpa, research
            ])
            
            prediction = self.predict(features)
            if prediction is not None:
                # Convert to percentage
                probability = round(prediction * 100, 2)
                logger.info(f"Admission probability: {probability}%")
                return probability
            return None
        except Exception as e:
            logger.error(f"Error predicting admission probability: {str(e)}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if not self.is_loaded or self.model is None:
            return {}
        
        try:
            info = {
                'model_type': type(self.model).__name__,
                'is_loaded': self.is_loaded,
                'has_feature_importance': hasattr(self.model, 'feature_importances_')
            }
            
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
                
            return info
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}


def create_model_manager() -> ModelManager:
    """
    Factory function to create and initialize a ModelManager instance.
    
    Returns:
        ModelManager: Initialized model manager
    """
    model_manager = ModelManager()
    model_manager.load_model()
    return model_manager
