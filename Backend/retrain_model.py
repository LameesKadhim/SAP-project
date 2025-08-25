#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the RandomForest model with current scikit-learn version
to fix compatibility issues with the saved model.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import sklearn

warnings.filterwarnings("ignore")

def main():
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv('../Dataset/admission_predict_V1.2.csv')
    
    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns.tolist())
    
    # Drop the Serial No. column
    df = df.drop(['Serial No.'], axis=1)
    
    # Check for null values
    print("Null values in dataset:")
    print(df.isnull().sum())
    
    # Prepare features and target
    X = df.drop(['Chance of Admit'], axis=1)
    y = df['Chance of Admit']
    
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, random_state=42
    )
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Train RandomForest model
    print("Training RandomForest model...")
    model_RandF = RandomForestRegressor(random_state=42)
    model_RandF.fit(X_train, y_train)
    
    # Make predictions
    predicted = model_RandF.predict(X_test)
    
    # Calculate score
    score = model_RandF.score(X_test, y_test)
    print(f"Model score: {score:.4f}")
    
    # Save the model
    print("Saving model...")
    filename = 'model_RandF.sav'
    joblib.dump(model_RandF, filename)
    
    # Test loading the model
    print("Testing model loading...")
    loaded_model = joblib.load(filename)
    test_score = loaded_model.score(X_test, y_test)
    print(f"Loaded model score: {test_score:.4f}")
    
    print(f"Current scikit-learn version: {sklearn.__version__}")
    print("Model successfully retrained and saved!")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model_RandF.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    main()
