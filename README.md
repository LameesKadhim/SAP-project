# Student Admission Prediction

<p align="center">
<img src="Frontend/assets/logo.jpg" width="200" height="170" alt="SAP Project Logo">
</p>

# Project Idea
This is a student project about predicting the chance of admission. For this project we are using Graduate Admission Dataset from Kaggle. We will use machine learning to analyze the data, find a model to predict the University Ranking and then visualize the result.

  
# Requirements and Preparation

## Required Libraries
The project uses the following key libraries:
- [Dash](https://plotly.com/dash/) - Web application framework
- [Numpy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning
- [Joblib](https://joblib.readthedocs.io/en/latest/) - Model persistence
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Dash-DAQ](https://dash.plotly.com/dash-daq) - Advanced Dash components

### Dependencies Management
All dependencies are managed through `requirements.txt` with specific versions for compatibility:
- **Core Dependencies**: dash, pandas, numpy, scikit-learn, joblib, plotly
- **UI Components**: dash-daq, dash-table
- **Development**: gunicorn (for deployment)

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone <repository-url>
cd StdAdmitPred
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Retrain Model (if needed)
If you encounter model compatibility issues, retrain the model:
```bash
cd Backend
python retrain_model.py
```

### 5. Start the Application
```bash
cd Frontend

# Run the original application
python app.py

# OR run the new modular application
python app_refactored.py
```

### 6. Access the Application
Open your web browser and go to:
```
http://127.0.0.1:8050/
```

## Recent Updates and Fixes

### ✅ Major Refactoring - Modular Design Implementation
- **Complete Code Restructuring**: Refactored monolithic `app.py` into modular architecture
- **Separation of Concerns**: Each component now has its own dedicated module
- **Enhanced Maintainability**: Easy to modify, test, and extend individual components
- **Professional Code Structure**: Following industry best practices and design patterns

### ✅ New Modular Architecture
```
Frontend/
├── config/
│   ├── __init__.py
│   └── settings.py          # All configuration constants
├── data/
│   ├── __init__.py
│   └── data_loader.py       # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   └── model_manager.py     # Model loading and predictions
├── visualizations/
│   ├── __init__.py
│   └── charts.py           # All chart creation functions
├── components/
│   ├── __init__.py
│   ├── header.py           # Header component
│   ├── home_tab.py         # Home tab component
│   ├── dataset_tab.py      # Dataset tab component
│   ├── dashboard_tab.py    # Dashboard tab component
│   ├── ml_tab.py          # Machine Learning tab component
│   └── prediction_tab.py   # Prediction tab component
├── callbacks/
│   ├── __init__.py
│   └── prediction_callback.py # Prediction callback logic
├── app.py                  # Original monolithic application
└── app_refactored.py       # New modular application
```

### ✅ Compatibility Fixes
- **Updated Dash imports**: Modernized from deprecated `dash_html_components` to `from dash import html, dcc, dash_table`
- **Fixed Dash server**: Updated from deprecated `app.run_server()` to `app.run()`
- **Model compatibility**: Retrained RandomForest model for scikit-learn 1.7.1 compatibility
- **Added retraining script**: `Backend/retrain_model.py` for easy model updates
- **Fixed dash_daq import**: Corrected import statement for Dash-DAQ components

### ✅ Project Structure
- **Backend**: Machine learning model and training scripts
- **Frontend**: Dash web application with interactive interface (now modular)
- **Dataset**: Graduate admission data for training and visualization
- **Requirements**: Comprehensive dependency management

### ✅ Features
- **Interactive Predictions**: Real-time admission probability calculation
- **Data Visualizations**: Comprehensive charts and graphs
- **Model Performance**: Feature importance and evaluation metrics
- **Responsive Design**: Modern, user-friendly interface
- **Modular Architecture**: Clean, maintainable, and scalable codebase
- **Error Handling**: Comprehensive error handling and logging throughout
- **Type Safety**: Full type hints and documentation for better code clarity

## Implementation Approach   

### Dataset: 
https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv

### Algorithms:
- **Regression Models**:
  - DecisionTree
  - Linear Regression
  - RandomForest (Selected)
  - KNeighbours
  - SVM
  - AdaBoostClassifier
  - GradientBoostingClassifier
  - Ridge
  - BayesianRidge
  - ElasticNet
  - HuberRegressor

### Tools:
- **DASH/Plotly** - Web framework and visualizations
- **scikit-learn** - Machine learning algorithms
- **Pandas/Numpy** - Data processing

### Project Architecture:
- **Machine Learning Model**: RandomForestRegressor (78.53% accuracy)
- **Backend**: Python with scikit-learn
- **Frontend**: Dash/Plotly web application (modular design)
- **Model Persistence**: Joblib for model saving/loading
- **Code Organization**: Modular architecture with separation of concerns
- **Error Handling**: Comprehensive logging and error management

## Visualization Features

### Home
![Home](https://user-images.githubusercontent.com/57901189/107371093-258cbc00-6ae4-11eb-8c8b-c059b9f9cc26.png)
----------------------------------------------
### Dataset
![Dataset](https://user-images.githubusercontent.com/57901189/107371106-2887ac80-6ae4-11eb-9198-cca7ff58b900.png)
----------------------------------------------
### Dashboard
![Dashboard](https://user-images.githubusercontent.com/57901189/107371115-2c1b3380-6ae4-11eb-9044-572573527dec.png)
----------------------------------------------
### Machine Learning
![ML](https://user-images.githubusercontent.com/57901189/107371134-30475100-6ae4-11eb-8494-f6084c03b9a5.png)
----------------------------------------------
### Prediction
![Prediction](https://user-images.githubusercontent.com/57901189/107371144-34736e80-6ae4-11eb-8afb-3644751a2d65.png)

## Model Performance

### Feature Importance (Latest Model):
1. **CGPA**: 75.97% (Most important factor)
2. **GRE Score**: 10.27%
3. **TOEFL Score**: 4.48%
4. **Statement of Purpose**: 3.19%
5. **Letter of Recommendation**: 2.92%
6. **University Rating**: 1.67%
7. **Research Experience**: 1.51%

### Model Accuracy:
- **RandomForest Regressor**: 78.53%
- **Test Set Performance**: Consistent and reliable predictions

## Troubleshooting

### Common Issues and Solutions:

1. **Model Loading Error**: 
   - Run `python Backend/retrain_model.py` to create a compatible model

2. **Import Errors**:
   - Ensure you're using the virtual environment
   - Install dependencies with `pip install -r requirements.txt`

3. **Dash Compatibility Issues**:
   - The app has been updated for modern Dash versions
   - Uses `app.run()` instead of deprecated `app.run_server()`

4. **Port Already in Use**:
   - Change the port in `app.py` or kill the existing process

5. **Modular Application Issues**:
   - Both `app.py` (original) and `app_refactored.py` (modular) are available
   - Use `app_refactored.py` for the new modular version
   - All functionality is identical between versions

6. **Dash-DAQ Import Issues**:
   - Fixed import statement: `import dash_daq as daq`
   - Ensure `dash-daq` is installed: `pip install dash-daq`

## Deployment

### Local Development:
```bash
cd Frontend

# Run original application
python app.py

# OR run modular application
python app_refactored.py
```

### Production Deployment (Heroku):
1. Create `Procfile` with: `web: gunicorn app:server`
2. Ensure `requirements.txt` is up to date
3. Deploy using Heroku CLI or GitHub integration

## Project Links

- **GitHub Repository**: https://github.com/LameesKadhim/SAP-project
- **Live Demo**: https://predict-student-admission.herokuapp.com/
- **Video Trailer**: https://youtu.be/rXDHiqIxYuQ

## Contributors
- [Saif Almaliki](https://github.com/SaifAlmaliki)
- [Lamees Kadhim](https://github.com/LameesKadhim)
- [Tamanna](https://github.com/tamanna18)
- [Kunal](https://github.com/kunalait)
- [Sepideh Hosseini Dehkordi](https://github.com/Sepideh-hd)

## Code Quality and Standards

### ✅ Modular Design Benefits
- **Maintainability**: Each component is isolated and easy to modify
- **Testability**: Individual modules can be tested independently
- **Scalability**: Easy to add new features or components
- **Reusability**: Components can be reused across different parts
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Full type hints and docstrings

### ✅ Development Best Practices
- **Separation of Concerns**: Each module has a single responsibility
- **Factory Pattern**: Clean initialization patterns for components
- **Configuration Management**: Centralized settings management
- **Error Logging**: Comprehensive logging for debugging
- **Type Safety**: Full type annotations for better code clarity

## License
This project is part of the Learning Analysis course (WS20/21) by the Datology Group.

---

**Note**: This project has been updated for modern Python and Dash compatibility. The model has been retrained for scikit-learn 1.7.1+ compatibility. The codebase has been refactored into a modular architecture following industry best practices.

