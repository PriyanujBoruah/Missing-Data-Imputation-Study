# split.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(features_path, target_path):
    print("Loading preprocessed data for splitting...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train.to_csv('train_features.csv', index=False)
    X_test.to_csv('test_features.csv', index=False)
    y_train.to_csv('train_target.csv', index=False)
    y_test.to_csv('test_target.csv', index=False)
    
    print("Data splitting complete. Saved train and test files.")

if __name__ == '__main__':
    FEATURES_PATH = 'processed_features.csv'
    TARGET_PATH = 'processed_target.csv'
    split_data(FEATURES_PATH, TARGET_PATH)
