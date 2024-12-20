# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:13:46 2024

@author: USER
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Import the upsampled data
# specified file path
file_path = r"C:\Users\USER\Desktop\credit_card_fraud_detection\upsampled_data.csv"
# Load data into a dataframe; df
df = pd.read_csv(file_path)
print(df)

# Modeling
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]    # Target

# Step 1: Split the dataset into 70% training, 15% validation, and 15% test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Step 2: Set up k-fold cross-validation and grid search for hyperparameter tuning
k = 4  # Number of folds
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}  # Regularization strength parameter

# Define StratifiedKFold for preserving class distribution across folds
cv = StratifiedKFold(n_splits=k)

# Logistic Regression with GridSearchCV
log_reg = LogisticRegression(solver='liblinear', random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after Grid Search
best_log_reg = grid_search.best_estimator_

# Step 3: Evaluate on Training, Cross-Validation, Validation, and Test Sets
def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluates the model and prints Accuracy, Precision, Recall, and F1 score."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='binary')  # Use 'weighted' for multiclass
    rec = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    print(f"{dataset_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    return acc, prec, rec, f1, y_pred

# Evaluate on the Training Set
print("Logistic Regression Results:")
print("Training Set Evaluation:")
train_acc, train_prec, train_rec, train_f1, y_train_pred = evaluate_model(best_log_reg, X_train, y_train, "Training Set")

# Cross-Validation predictions and metrics using cross_val_predict
y_cv_pred = cross_val_predict(best_log_reg, X_train, y_train, cv=cv)
print("Cross-Validation Evaluation:")
cv_acc = accuracy_score(y_train, y_cv_pred)
cv_prec = precision_score(y_train, y_cv_pred, average='binary')
cv_rec = recall_score(y_train, y_cv_pred, average='binary')
cv_f1 = f1_score(y_train, y_cv_pred, average='binary')
print(f"Cross-Validation - Accuracy: {cv_acc:.4f}, Precision: {cv_prec:.4f}, Recall: {cv_rec:.4f}, F1 Score: {cv_f1:.4f}")

# Evaluate on the Validation Set
print("Validation Set Evaluation:")
val_acc, val_prec, val_rec, val_f1, y_val_pred = evaluate_model(best_log_reg, X_val, y_val, "Validation Set")

# Evaluate on the Test Set
print("Test Set Evaluation:")
test_acc, test_prec, test_rec, test_f1, y_test_pred = evaluate_model(best_log_reg, X_test, y_test, "Test Set")

# Step 4: Plot ROC curves and calculate AUC for each set
def plot_roc_curve(model, X, y, dataset_name="Dataset"):
    """Plots ROC curve and calculates AUC for the specified dataset."""
    y_prob = model.predict_proba(X)[:, 1]  # Get the probabilities for the positive class
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {roc_auc:.4f})')
    return roc_auc

plt.figure(figsize=(8, 6))
# Training ROC Curve
train_auc = plot_roc_curve(best_log_reg, X_train, y_train, "Training Set")
# Cross-Validation ROC Curve using cross_val_predict with probabilities
y_cv_prob = cross_val_predict(best_log_reg, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
fpr_cv, tpr_cv, _ = roc_curve(y_train, y_cv_prob)
cv_auc = auc(fpr_cv, tpr_cv)
plt.plot(fpr_cv, tpr_cv, label=f'Cross-Validation (AUC = {cv_auc:.4f})')

# Validation ROC Curve
val_auc = plot_roc_curve(best_log_reg, X_val, y_val, "Validation Set")
# Test ROC Curve
test_auc = plot_roc_curve(best_log_reg, X_test, y_test, "Test Set")

# Finalize the plot
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Step 5: Save the model
joblib.dump(best_log_reg, r'C:\Users\USER\Desktop\credit_card_fraud_detection\action_project_flask_app\best_logistic_regression_model.joblib')
print("Model saved to 'C:\\Users\\USER\\Desktop\\credit_card_fraud_detection\action_project_flask_app\\best_logistic_regression_model.joblib'.")

# Save evaluation metrics to a DataFrame and then to a CSV file
metrics_data = {
    'Dataset': ['Training', 'Cross-Validation', 'Validation', 'Test'],
    'Accuracy': [train_acc, cv_acc, val_acc, test_acc],
    'Precision': [train_prec, cv_prec, val_prec, test_prec],
    'Recall': [train_rec, cv_rec, val_rec, test_rec],
    'F1 Score': [train_f1, cv_f1, val_f1, test_f1]
}
metrics_df = pd.DataFrame(metrics_data)

# Save metrics to a CSV file
metrics_df.to_csv(r'C:\Users\USER\Desktop\credit_card_fraud_detection\logistic_regression_model_evaluation_metrics.csv', index=False)
print("Evaluation metrics saved to 'C:\\Users\\USER\\Desktop\\credit_card_fraud_detection\\logistic_regression_model_evaluation_metrics.csv'.")


