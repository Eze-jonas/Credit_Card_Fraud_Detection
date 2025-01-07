import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Import the upsampled data
# specified file path
file_path = r"C:\Users\USER\Desktop\credit_card_fraud_detection\upsampled_data.csv"
# Load data into a dataframe; df
df = pd.read_csv(file_path)
print(df)

# Modeling
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Target

# Step 1: Split the dataset into 70% training, 15% validation, and 15% test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


# Define the ANN model with Input layer to remove the warning
def create_ann_model():
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Define input shape with Input layer
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Step 2: Perform 4-Fold Cross-Validation
k = 4
cv = StratifiedKFold(n_splits=k)
cv_scores = []
y_cv_preds = np.empty(len(X_train))

for train_idx, val_idx in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = create_ann_model()  # Create a new model for each fold
    model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0)
    y_val_pred_fold = (model.predict(X_val_fold) > 0.5).astype("int32").ravel()

    y_cv_preds[val_idx] = y_val_pred_fold
    cv_scores.append(f1_score(y_val_fold, y_val_pred_fold))

print(f"Cross-Validation - Average F1 Score: {np.mean(cv_scores):.4f}")

# Final model training on the full training set
final_model = create_ann_model()
final_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# Step 3: Save the final model
final_model.save(
    r'C:\Users\USER\Desktop\credit_card_fraud_detection\action_project_flask_app\best_ann_model.keras')  # Save using the specified path
print(
    "Keras model saved to 'C:\\Users\\USER\\Desktop\\credit_card_fraud_detection\action_project_flask_app\\best_ann_model.keras'.")


# Evaluation function
def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{dataset_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    return acc, prec, rec, f1


# Evaluate on Training Set
y_train_pred = (final_model.predict(X_train) > 0.5).astype("int32")
print("ANN Results:")
print("Training Set Evaluation:")
train_acc, train_prec, train_rec, train_f1 = evaluate_model(y_train, y_train_pred, "Training Set")

# Evaluate on Cross-Validation Set
print("Cross-Validation Evaluation:")
cv_acc, cv_prec, cv_rec, cv_f1 = evaluate_model(y_train, y_cv_preds, "Cross-Validation Set")

# Evaluate on Validation Set
y_val_pred = (final_model.predict(X_val) > 0.5).astype("int32")
print("Validation Set Evaluation:")
val_acc, val_prec, val_rec, val_f1 = evaluate_model(y_val, y_val_pred, "Validation Set")

# Evaluate on Test Set
y_test_pred = (final_model.predict(X_test) > 0.5).astype("int32")
print("Test Set Evaluation:")
test_acc, test_prec, test_rec, test_f1 = evaluate_model(y_test, y_test_pred, "Test Set")


# Step 4: Plot ROC Curve and AUC for Training, Cross-Validation, Validation, and Test Sets
def plot_roc_curve(y_true, y_probs, dataset_name="Dataset"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {roc_auc:.4f})')
    return roc_auc


plt.figure(figsize=(8, 6))

# Training ROC Curve
y_train_probs = final_model.predict(X_train).ravel()
train_auc = plot_roc_curve(y_train, y_train_probs, "Training Set")

# Cross-Validation ROC Curve
cv_auc = plot_roc_curve(y_train, y_cv_preds, "Cross-Validation Set")

# Validation ROC Curve
y_val_probs = final_model.predict(X_val).ravel()
val_auc = plot_roc_curve(y_val, y_val_probs, "Validation Set")

# Test ROC Curve
y_test_probs = final_model.predict(X_test).ravel()
test_auc = plot_roc_curve(y_test, y_test_probs, "Test Set")

# Finalize the plot
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ANN ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Step 5: Save evaluation metrics to a DataFrame and then to a CSV file
metrics_data = {
    'Dataset': ['Training', 'Cross-Validation', 'Validation', 'Test'],
    'Accuracy': [train_acc, cv_acc, val_acc, test_acc],
    'Precision': [train_prec, cv_prec, val_prec, test_prec],
    'Recall': [train_rec, cv_rec, val_rec, test_rec],
    'F1 Score': [train_f1, cv_f1, val_f1, test_f1]
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_data)

# Save metrics to a CSV file
metrics_df.to_csv(r'C:\Users\USER\Desktop\credit_card_fraud_detection\ann_model_evaluation_metrics.csv', index=False)
print(
    "Evaluation metrics saved to 'C:\\Users\\USER\\Desktop\\credit_card_fraud_detection\\ann_model_evaluation_metrics.csv'.")
