"""
Configuration settings for the Student Admission Prediction application.
"""

# External stylesheets
EXTERNAL_STYLESHEETS = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
]

# File paths
DATASET_PATH = '../Dataset/admission_predict_V1.2.csv'
MODEL_PATH = '../Backend/model_RandF.sav'

# Chart colors
PIE_CHART_COLORS = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']

# Chart dimensions
CHART_HEIGHT = 500
CHART_WIDTH = 700

# Prediction input ranges
GRE_SCORE_RANGE = {'min': 130, 'max': 170, 'default': 140}
TOEFL_SCORE_RANGE = {'min': 61, 'max': 120, 'default': 90}
CGPA_RANGE = {'min': 5, 'max': 10, 'default': 7}

# Dropdown options
UNIVERSITY_RATING_OPTIONS = [
    {'label': str(i), 'value': str(i)} for i in range(1, 6)
]

LOR_OPTIONS = [
    {'label': str(i), 'value': str(i)} for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
]

SOP_OPTIONS = [
    {'label': str(i), 'value': str(i)} for i in range(1, 6)
]

RESEARCH_OPTIONS = [
    {'label': 'YES', 'value': '1'},
    {'label': 'NO', 'value': '0'}
]

# Feature names for model input
FEATURE_NAMES = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

# App configuration
APP_TITLE = "Student Admission Prediction (SAP)"
DEBUG_MODE = True
HOST = '127.0.0.1'
PORT = 8050
