"""
Quick check of real calibration data
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    DATA_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025/data")
    
    print("=== Quick Real Data Check ===")
    
    files = ['dark.parquet', 'dead.parquet', 'flat.parquet', 'read.parquet']
    
    for filename in files:
        file_path = DATA_DIR / filename
        if file_path.exists():
            print(f"\\n--- {filename} ---")
            try:
                df = pd.read_parquet(file_path)
                print(f"Shape: {df.shape}")
                print(f"Data type: {df.dtypes.iloc[0]}")
                
                if filename == 'dead.parquet':
                    # Dead pixels (boolean)
                    dead_count = df.sum().sum()
                    total = df.shape[0] * df.shape[1]
                    print(f"Dead pixels: {dead_count}/{total} ({dead_count/total:.4f})")
                else:
                    # Numerical data
                    values = df.values.flatten()
                    print(f"Range: [{values.min():.6f}, {values.max():.6f}]")
                    print(f"Mean: {values.mean():.6f}")
                    print(f"Std: {values.std():.6f}")
                
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"{filename} not found")
    
    print(f"\\n=== Summary ===")
    print("FGS1 detector: 32x32 pixels")
    print("Available calibration files:")
    for filename in files:
        exists = (DATA_DIR / filename).exists()
        print(f"  {filename}: {'OK' if exists else 'Missing'}")

if __name__ == "__main__":
    main()