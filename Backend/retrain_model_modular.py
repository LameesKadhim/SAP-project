#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular version of the model retraining script.
Uses the new modular backend components for better organization and maintainability.
"""

import warnings
import sklearn
from data_processor import create_data_processor
from model_trainer import create_model_trainer

warnings.filterwarnings("ignore")


def main():
    """Main function to retrain the model using modular components."""
    print("=== Student Admission Prediction - Model Retraining ===")
    print("Using modular backend components...")
    
    try:
        # Initialize components
        data_processor = create_data_processor()
        model_trainer = create_model_trainer()
        
        # Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        df = data_processor.load_data()
        if df is None:
            print("❌ Failed to load data")
            return
        
        df = data_processor.preprocess_data()
        if df is None:
            print("❌ Failed to preprocess data")
            return
        
        # Prepare features and target
        print("\n2. Preparing features and target...")
        X, y = data_processor.prepare_features_target()
        if X is None or y is None:
            print("❌ Failed to prepare features and target")
            return
        
        # Split data
        print("\n3. Splitting data...")
        if not model_trainer.split_data(X, y):
            print("❌ Failed to split data")
            return
        
        # Train model
        print("\n4. Training model...")
        if not model_trainer.train_model():
            print("❌ Failed to train model")
            return
        
        # Evaluate model
        print("\n5. Evaluating model...")
        evaluation_results = model_trainer.evaluate_model()
        if evaluation_results is None:
            print("❌ Failed to evaluate model")
            return
        
        # Print results
        print(f"\n✅ Model Score: {evaluation_results['score']:.4f}")
        print("\nFeature Importance:")
        print(evaluation_results['feature_importance'])
        
        # Save model
        print("\n6. Saving model...")
        if not model_trainer.save_model():
            print("❌ Failed to save model")
            return
        
        # Test model loading
        print("\n7. Testing model loading...")
        if not model_trainer.test_model_loading():
            print("❌ Failed to test model loading")
            return
        
        # Print final information
        print(f"\n✅ Model successfully retrained and saved!")
        print(f"📊 Current scikit-learn version: {sklearn.__version__}")
        print(f"📁 Model saved to: {model_trainer.model_save_path}")
        
        # Print model info
        model_info = model_trainer.get_model_info()
        print(f"🤖 Model type: {model_info.get('model_type', 'N/A')}")
        print(f"🌳 Number of estimators: {model_info.get('n_estimators', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return


if __name__ == "__main__":
    main()
