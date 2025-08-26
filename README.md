# Student Admission Prediction

<p align="center">
<img src="WebApp/assets/logo.jpg" width="200" height="170" alt="SAP Project Logo">
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
cd ML
python retrain_model_modular.py
```

### 5. Start the Application
```bash
cd WebApp
python app.py
```

### 6. Access the Application
Open your web browser and go to:
```
http://127.0.0.1:8050/
```

## Project Structure

```
StdAdmitPred/
├── ML/                    # Machine Learning & Data Processing
│   ├── config.py          # ML configuration settings
│   ├── data_processor.py  # Data loading and preprocessing
│   ├── model_trainer.py   # Model training and evaluation
│   ├── retrain_model_modular.py  # Model retraining script
│   ├── model_RandF.sav    # Trained model file
│   └── Prediction.ipynb   # Jupyter notebook for model development
├── WebApp/                # Dash Web Application
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py    # Application configuration
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_manager.py # Model loading and predictions
│   ├── visualizations/
│   │   ├── __init__.py
│   │   └── charts.py      # Chart creation functions
│   ├── components/
│   │   ├── __init__.py
│   │   ├── header.py      # Header component
│   │   ├── home_tab.py    # Home tab component
│   │   ├── dataset_tab.py # Dataset tab component
│   │   ├── dashboard_tab.py # Dashboard tab component
│   │   ├── ml_tab.py      # Machine Learning tab component
│   │   └── prediction_tab.py # Prediction tab component
│   ├── callbacks/
│   │   ├── __init__.py
│   │   └── prediction_callback.py # Prediction callback logic
│   ├── assets/            # Static assets (images, CSS)
│   ├── app.py             # Main application file
│   └── Visualization.ipynb # Visualization development notebook
├── Dataset/               # Data Files
│   ├── admission_predict_V1.2.csv
│   └── Admission_Predict_Ver1.1.csv
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Features

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
- **ML Module**: Python with scikit-learn for model training and data processing
- **WebApp Module**: Dash/Plotly web application with modular design
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
   - Run `python ML/retrain_model_modular.py` to create a compatible model

2. **Import Errors**:
   - Ensure you're using the virtual environment
   - Install dependencies with `pip install -r requirements.txt`

3. **Port Already in Use**:
   - Change the port in `app.py` or kill the existing process

4. **Dash-DAQ Import Issues**:
   - Ensure `dash-daq` is installed: `pip install dash-daq`

## Deployment

### Local Development:
```bash
cd WebApp
python app.py
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

**Note**: This project uses modern Python and Dash compatibility with a modular architecture following industry best practices.

