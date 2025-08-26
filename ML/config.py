"""
Configuration settings for the ML module.
"""

# Dataset configuration
DATASET_PATH = '../Dataset/admission_predict_V1.2.csv'
MODEL_SAVE_PATH = 'model_RandF.sav'

# Model training parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42
SHUFFLE = True

# Model parameters
MODEL_TYPE = 'RandomForestRegressor'
MODEL_PARAMS = {
    'random_state': 42
}

# Feature configuration
FEATURE_COLUMNS = [
    'GRE Score', 'TOEFL Score', 'University Rating', 
    'SOP', 'LOR', 'CGPA', 'Research'
]
TARGET_COLUMN = 'Chance of Admit'
DROP_COLUMNS = ['Serial No.']

# GRE Score conversion parameters (if needed)
GRE_CONVERSION = {
    'old_min': 200,
    'old_max': 800,
    'new_min': 130,
    'new_max': 170
}
