import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer # Required for MICE
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

def impute_data(input_path, output_path, method='mean'):
    """
    Loads a dataset with missing values and applies an imputation technique.

    Args:
        input_path (str): Path to the degraded CSV file.
        output_path (str): Path to save the new, imputed CSV file.
        method (str): The imputation method to use ('mean', 'median', 'knn', 'mice').
    """
    print(f"Imputing data from {input_path} using the '{method}' method...")
    
    # Load the degraded dataset
    df = pd.read_csv(input_path)
    
    # Choose the imputer based on the method
    if method in ['mean', 'median']:
        imputer = SimpleImputer(strategy=method)
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'mice':
        # MICE is powerful but can be slow on large datasets
        imputer = IterativeImputer(max_iter=10, random_state=42)
    else:
        raise ValueError("Method not recognized. Choose from 'mean', 'median', 'knn', 'mice'.")

    # Fit the imputer and transform the data
    # The result is a numpy array, so we need to convert it back to a DataFrame
    df_imputed_array = imputer.fit_transform(df)
    
    # Create the new imputed DataFrame, preserving column names
    df_imputed = pd.DataFrame(df_imputed_array, columns=df.columns)
    
    # Save the new imputed dataframe
    df_imputed.to_csv(output_path, index=False)
    
    print(f"Saved imputed file to {output_path}")

if __name__ == '__main__':
    # --- CONFIGURE YOUR IMPUTATION TEST ---
    # This is the file created by degrade_data.py
    INPUT_FILE = 'train_features_degraded_5perc.csv'
    
    # Choose which method to test first
    IMPUTATION_METHOD = 'mean' # Options: 'mean', 'median', 'knn', 'mice'
    
    OUTPUT_FILE = f'train_features_imputed_{IMPUTATION_METHOD}.csv'
    
    impute_data(INPUT_FILE, OUTPUT_FILE, IMPUTATION_METHOD)
