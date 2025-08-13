import pandas as pd
import numpy as np

def degrade_data(input_path, output_path, degradation_level):
    """
    Loads a dataset and introduces missing values (NaNs).

    Args:
        input_path (str): Path to the clean CSV file.
        output_path (str): Path to save the new, degraded CSV file.
        degradation_level (float): The fraction of data to remove (e.g., 0.05 for 5%).
    """
    print(f"Degrading data from {input_path} by {degradation_level*100:.2f}%...")
    
    # Load the clean dataset
    df = pd.read_csv(input_path)
    
    # Create a copy to degrade
    df_degraded = df.copy()
    
    # Create a boolean mask. True means the cell will be replaced with NaN.
    # The 'p' argument sets the probability for True (degradation_level) and False.
    mask = np.random.choice([True, False], 
                            size=df_degraded.shape, 
                            p=[degradation_level, 1 - degradation_level])
    
    # Apply the mask to set values to NaN
    df_degraded[mask] = np.nan
    
    # Save the new degraded dataframe
    df_degraded.to_csv(output_path, index=False)
    
    print(f"Saved degraded file to {output_path}")

if __name__ == '__main__':
    # --- CONFIGURE YOUR DEGRADATION TEST ---
    # We'll use the training features from one of your baseline runs
    INPUT_FILE = 'train_features.csv'
    
    # Let's start by testing a 5% degradation
    DEGRADATION_PERCENT = 0.05 
    
    OUTPUT_FILE = f'train_features_degraded_{int(DEGRADATION_PERCENT*100)}perc.csv'
    
    degrade_data(INPUT_FILE, OUTPUT_FILE, DEGRADATION_PERCENT)
