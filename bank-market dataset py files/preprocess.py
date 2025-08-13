# preprocess.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(dataset_path, target_column):
    """
    Loads data, separates features and target, and applies preprocessing.
    
    Args:
        dataset_path (str): The path to the raw CSV file.
        target_column (str): The name of the target variable column.
        
    Returns:
        pd.DataFrame: Preprocessed features.
        pd.Series: Target variable.
    """
    print(f"Loading and preprocessing {dataset_path}...")
    
    # Load the dataset
    df = pd.read_csv(dataset_path, delimiter=';')
    
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    
    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Convert the processed features back to a DataFrame for clarity
    # Get new column names after one-hot encoding
    new_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    new_col_names = list(numerical_features) + list(new_cat_features)
    
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=new_col_names, index=X.index)
    
    # Save the processed data
    X_preprocessed_df.to_csv('processed_features.csv', index=False)
    y.to_csv('processed_target.csv', index=False)
    
    print("Preprocessing complete. Saved 'processed_features.csv' and 'processed_target.csv'.")
    return X_preprocessed_df, y

if __name__ == '__main__':
    # --- CONFIGURE YOUR DATASET HERE ---
    # Example for the Bank Marketing dataset
    DATASET_PATH = 'bank-full.csv' # Change to your dataset file
    TARGET_COLUMN = 'y'          # Change to your target column name
    
    preprocess_data(DATASET_PATH, TARGET_COLUMN)