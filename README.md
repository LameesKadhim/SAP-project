# Student Admission Prediction (SAP-Project)

<p align="center">
<img src="https://github.com/LameesKadhim/SAP-project/blob/main/Frontend/assets/logo.jpg" width="200" height="170">
</p>

# Project Idea
This is a student project about predicting the chance of admission. For this project we are using Graduate Admission Dataset from Kaggle. We will use machine learning to analyze the data, find a model to predict the University Ranking and then visualize the result.

  
# Requirments and Preparation

  * Required Libraries
    + [Dash](https://plotly.com/dash/)
    + [Numpy](https://numpy.org/)
    + [Pandas](https://pandas.pydata.org/)
    + [scikit-learn](https://scikit-learn.org/stable/)
    + [Joblib](https://joblib.readthedocs.io/en/latest/)
  
  * Start Server
    * Create a virtual environment for the project: <b> python -m venv venv </b>
    * Activate the virtual environment for windows: <b> venv\Scripts\activate </b>
      - Activate the virtual environment for Linux/macOS: <b> source venv/bin/activate </b>
    * Install required libraries in the project: </b> pip install dash pandas numpy joblib </b>
    * Start server: <b> python app.py </b>

 
 
# Implementation approach   

## Dataset: 
  https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv

## Algorithms:
  * Regression
      * DecisionTree
      * Linear Regression
      * RandomForest
      * KNeighbours
      * SVM
      * AdaBoostClassifier
      * GradientBoostingClassifier
      * Ridge
      * BayesianRidge
      * ElasticNet
      * HuberRegressor
      
## Tools:
* DASH/Plotly

## Visualization:

###  Home
![Home](https://user-images.githubusercontent.com/57901189/107371093-258cbc00-6ae4-11eb-8c8b-c059b9f9cc26.png)
----------------------------------------------
###  Dataset
![Dataset](https://user-images.githubusercontent.com/57901189/107371106-2887ac80-6ae4-11eb-9198-cca7ff58b900.png)
----------------------------------------------
### Dashboard
![Dashboard'](https://user-images.githubusercontent.com/57901189/107371115-2c1b3380-6ae4-11eb-9044-572573527dec.png)
----------------------------------------------
### Machine Learning
![ML](https://user-images.githubusercontent.com/57901189/107371134-30475100-6ae4-11eb-8494-f6084c03b9a5.png)
----------------------------------------------
### Prediction
![Prediction](https://user-images.githubusercontent.com/57901189/107371144-34736e80-6ae4-11eb-8afb-3644751a2d65.png)



# SAP-Project on GitHub:
  https://github.com/LameesKadhim/SAP-project

# SAP-Project on Heroku:
  https://predict-student-admission.herokuapp.com/

# SAP-Project Video Trailer:

# Deployment steps on Heroku

* <b> Step 1. Create a new folder for your project: </b>

  $ mkdir sap-project
  
  $ cd sap-project
  
  
* <b> Step 2. Initialize the folder with git and a virtualenv </b>

  $ git init      // initializes an empty git repo
  
  $ virtualenv venv // creates a virtualenv called "venv"
  
  $ venv\Scripts\activate // Activate the virtual environment for windows
  
    -uses the virtualenv for linux and Macos:    $ source venv/bin/activate 
    
  You will need to reinstall your app's dependencies with this virtualenv:

  <code> $ pip install dash <code>
  
  <code> $ pip install plotly </code>   
  
  You will also need a new dependency, gunicorn, for deploying the app:
  
  <code> $ pip install gunicorn </code>
  
* <b> Step 3. Initialize the folder with the (app.py), a .gitignore file, requirements.txt, and a Procfile for deployment </b>
     content of .gitignore file:
     
     <code> venv </code>
     
     <code> *.pyc </code>
     
     <code> .DS_Store </code>
     
     <code> .env </code>
     
  <b> content of Procfile is</b> --> web: gunicorn app:server
   
  <b> creation of requirements.txt file: </b>
   
   requirements.txt describes your Python dependencies. You can fill this file in automatically with:
   
  <code> $ pip freeze > requirements.txt </code>
   
* <b> Step 4 Initialize Heroku, add files to Git, and deploy </b>

  <code> $ heroku create my-dash-app # change my-dash-app to a unique name </code>
  
  <code> $ git add . # add all files to git </code>
  
  <code> $ git commit -m 'Initial app boilerplate' </code>
  
  <code> $ git push heroku master # deploy code to heroku </code>
   

# Contributors
  * <a href="https://github.com/SaifAlmaliki" target="_blank">Saif Almaliki</a>
  
  * <a href="https://github.com/LameesKadhim" target="_blank">Lamees Kadhim</a>
  
  * <a href="https://github.com/tamanna18" target="_blank">Tamanna</a>
  
  * <a href="https://github.com/kunalait" target="_blank">Kunal</a>
  
  * <a href="https://github.com/Sepideh-hd" target="_blank">Sepideh Hosseini Dehkordi</a>
  

