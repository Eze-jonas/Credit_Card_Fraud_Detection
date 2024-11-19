Credit Card Fraud Detection Web Application (Flask)
This is a Flask web application that serves predictions for credit card fraud detection using machine learning models. The app allows users to choose between different models (Decision Tree, Logistic Regression, and Artificial Neural Network) and get predictions based on input features.

Technologies Used:
Flask - Web framework for building the application.
TensorFlow - For loading and predicting with the ANN model.
Scikit-learn - For the Decision Tree and Logistic Regression models.
Joblib - For loading the models.
Numpy - For handling numerical data.
Features:
Users can select a machine learning model (Decision Tree, Logistic Regression, or ANN).
Users input 23 features related to credit card transactions.
The app returns predictions of whether a transaction is fraudulent or not.
Setup & Installation:
Prerequisites:
Python 3.10.13
Virtual environment (preferably Conda)
The models (best_decision_tree_model.joblib, best_logistic_regression_model.joblib, best_ann_model.keras) should be available in your project directory.