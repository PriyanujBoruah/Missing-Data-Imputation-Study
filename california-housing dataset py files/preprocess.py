# preprocess.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re # Import the regular expression module

def preprocess_data(dataset_path, target_column):
    print(f"Loading and preprocessing {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Get new column names if categorical features exist
    if len(categorical_features) > 0:
        new_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        new_col_names = list(numerical_features) + list(new_cat_features)
    else:
        new_col_names = list(numerical_features)

    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=new_col_names, index=X.index)
    
    # --- FIX ADDED HERE ---
    # Sanitize column names for XGBoost compatibility
    X_preprocessed_df.columns = [re.sub(r'[\[\]<]', '', col) for col in X_preprocessed_df.columns]
    
    # Save the processed data
    X_preprocessed_df.to_csv('processed_features.csv', index=False)
    y.to_csv('processed_target.csv', index=False)
    
    print("Preprocessing complete. Saved 'processed_features.csv' and 'processed_target.csv'.")

if __name__ == '__main__':
    # --- CONFIGURED FOR CALIFORNIA HOUSING ---
    DATASET_PATH = 'housing.csv' 
    TARGET_COLUMN = 'median_house_value' # Confirm this is the name in your CSV
    
    preprocess_data(DATASET_PATH, TARGET_COLUMN)
