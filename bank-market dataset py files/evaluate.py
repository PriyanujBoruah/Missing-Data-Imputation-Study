import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model_path, features_path, target_path, task_type):
    """
    Loads a model and provides a comprehensive evaluation on the test set.
    
    Args:
        model_path (str): Path to the saved model file.
        features_path (str): Path to the test features.
        target_path (str): Path to the test target.
        task_type (str): 'classification' or 'regression'.
    """
    print("Loading model and test data for comprehensive evaluation...")
    # Load the model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(features_path)
    y_test_df = pd.read_csv(target_path)

    # Make predictions
    predictions = model.predict(X_test)

    # --- REGRESSION EVALUATION ---
    if task_type == 'regression':
        score = mean_squared_error(y_test_df, predictions, squared=False) # RMSE
        print(f"--- Baseline Performance ---")
        print(f"RMSE: {score:.4f}")
        return

    # --- CLASSIFICATION EVALUATION ---
    # Ensure target is encoded numerically
    le = LabelEncoder()
    y_test = le.fit_transform(y_test_df.iloc[:, 0])
    
    # Define settings for binary vs multi-class
    num_classes = len(np.unique(y_test))
    if num_classes > 2: # Multi-class (e.g., Iris)
        average_method = 'weighted'
        pos_label = None # Not applicable for multi-class F1/Precision/Recall
    else: # Binary (e.g., Bank Marketing)
        average_method = 'binary'
        # Determine the positive label for clarity (e.g., the encoded version of 'yes')
        pos_label = 1 if 'yes' in le.classes_ else 1

    # --- Calculate and Print All Metrics ---
    print("\n--- Baseline Performance Metrics ---")
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy:  {accuracy:.4f}")

    precision = precision_score(y_test, predictions, average=average_method, pos_label=pos_label, zero_division=0)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test, predictions, average=average_method, pos_label=pos_label, zero_division=0)
    print(f"Recall:    {recall:.4f}")
    
    f1 = f1_score(y_test, predictions, average=average_method, pos_label=pos_label, zero_division=0)
    print(f"F1-Score:  {f1:.4f}")

    # AUC-ROC requires probability scores
    if task_type == 'classification' and num_classes == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC:   {auc_roc:.4f}")

    # --- Display Confusion Matrix ---
    print("\nDisplaying Confusion Matrix...")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=le.classes_)
    plt.title("Baseline Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    # --- CONFIGURE YOUR TASK HERE ---
    TASK_TYPE = 'classification' # 'classification' or 'regression'

    MODEL_PATH = 'xgboost_model.joblib'
    TEST_FEATURES_PATH = 'test_features.csv'
    TEST_TARGET_PATH = 'test_target.csv'
    evaluate_model(MODEL_PATH, TEST_FEATURES_PATH, TEST_TARGET_PATH, TASK_TYPE)
