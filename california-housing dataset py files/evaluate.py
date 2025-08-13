import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_squared_error, # We will use this for MSE
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np # Import numpy for the square root function
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
    print("Loading model and test data for evaluation...")
    # Load the model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(features_path)
    y_test_df = pd.read_csv(target_path)

    # Make predictions
    predictions = model.predict(X_test)

    # --- REGRESSION EVALUATION ---
    if task_type == 'regression':
        # --- FIX APPLIED HERE ---
        # Calculate MSE first, then take the square root to get RMSE
        mse = mean_squared_error(y_test_df, predictions)
        rmse = np.sqrt(mse)
        
        print(f"--- Baseline Performance ---")
        print(f"RMSE: {rmse:.4f}")
        return

    # --- CLASSIFICATION EVALUATION ---
    # (The classification part remains unchanged)
    le = LabelEncoder()
    y_test = le.fit_transform(y_test_df.iloc[:, 0])
    
    num_classes = len(np.unique(y_test))
    average_method = 'weighted' if num_classes > 2 else 'binary'
    pos_label = 1 if average_method == 'binary' and 'yes' in le.classes_ else 1

    print("\n--- Baseline Performance Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, average=average_method, pos_label=pos_label, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, average=average_method, pos_label=pos_label, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, predictions, average=average_method, pos_label=pos_label, zero_division=0):.4f}")

    if num_classes == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nDisplaying Confusion Matrix...")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=le.classes_)
    plt.title("Baseline Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    # --- CONFIGURED FOR CALIFORNIA HOUSING ---
    TASK_TYPE = 'regression'

    MODEL_PATH = 'xgboost_model.joblib'
    TEST_FEATURES_PATH = 'test_features.csv'
    TEST_TARGET_PATH = 'test_target.csv'
    evaluate_model(MODEL_PATH, TEST_FEATURES_PATH, TEST_TARGET_PATH, TASK_TYPE)
