# train.py
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

def train_model(features_path, target_path, task_type):
    print(f"Loading training data for {task_type} task...")
    X_train = pd.read_csv(features_path)
    y_train_df = pd.read_csv(target_path)
    
    if task_type == 'classification':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_df.iloc[:, 0])
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif task_type == 'regression':
        y_train = y_train_df
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
        
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'xgboost_model.joblib')
    print("Model training complete. Saved 'xgboost_model.joblib'.")

if __name__ == '__main__':
    # --- CONFIGURED FOR IRIS ---
    TASK_TYPE = 'classification' 
    
    TRAIN_FEATURES_PATH = 'train_features.csv'
    TRAIN_TARGET_PATH = 'train_target.csv'
    train_model(TRAIN_FEATURES_PATH, TRAIN_TARGET_PATH, TASK_TYPE)