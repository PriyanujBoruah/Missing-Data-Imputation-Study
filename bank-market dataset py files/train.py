# train.py
import pandas as pd
import xgboost as xgb
import joblib # For saving the model

def train_model(features_path, target_path, task_type):
    """
    Trains an XGBoost model and saves it.
    
    Args:
        features_path (str): Path to the training features.
        target_path (str): Path to the training target.
        task_type (str): 'classification' or 'regression'.
    """
    print(f"Loading training data for {task_type} task...")
    X_train = pd.read_csv(features_path)
    y_train = pd.read_csv(target_path)
    
    # Initialize the appropriate XGBoost model
    if task_type == 'classification':
        # You might need to encode the target variable if it's text (e.g., 'yes'/'no')
        # For simplicity, we assume it's already 0/1 for binary, or 0/1/2 for multi-class
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train.iloc[:, 0])
        
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        
    elif task_type == 'regression':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
        
    # Train the model
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, 'xgboost_model.joblib')
    print("Model training complete. Saved 'xgboost_model.joblib'.")

if __name__ == '__main__':
    # --- CONFIGURE YOUR TASK HERE ---
    # For Bank Marketing or Iris, use 'classification'
    # For California Housing, use 'regression'
    TASK_TYPE = 'classification' 
    
    TRAIN_FEATURES_PATH = 'train_features.csv'
    TRAIN_TARGET_PATH = 'train_target.csv'
    train_model(TRAIN_FEATURES_PATH, TRAIN_TARGET_PATH, TASK_TYPE)