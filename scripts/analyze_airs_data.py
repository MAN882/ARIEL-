"""
Analyze AIRS-CH0 detector data structure
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_airs_data():
    """Analyze AIRS-CH0 calibration data"""
    
    print("=== AIRS-CH0 Detector Analysis ===")
    
    data_dir = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025/data/AIRS_CH0_cal0")
    
    files = ['dark.parquet', 'dead.parquet', 'flat.parquet', 'read.parquet', 'linear_corr.parquet']
    
    for filename in files:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"\\n--- {filename} ---")
            try:
                df = pd.read_parquet(file_path)
                print(f"Shape: {df.shape}")
                print(f"Data type: {df.dtypes.iloc[0]}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                if filename == 'dead.parquet':
                    # Dead pixels (boolean)
                    dead_count = df.sum().sum()
                    total = df.shape[0] * df.shape[1]
                    print(f"Dead pixels: {dead_count}/{total} ({dead_count/total:.6f})")
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
    
    print(f"\\n=== Detector Comparison ===")
    
    # Compare with FGS1
    fgs1_shape = (32, 32)  # 1,024 pixels
    
    try:
        airs_dead = pd.read_parquet(data_dir / 'dead.parquet')
        airs_shape = airs_dead.shape
        airs_pixels = airs_shape[0] * airs_shape[1]
        fgs1_pixels = fgs1_shape[0] * fgs1_shape[1]
        
        print(f"FGS1: {fgs1_shape[0]}x{fgs1_shape[1]} = {fgs1_pixels:,} pixels")
        print(f"AIRS-CH0: {airs_shape[0]}x{airs_shape[1]} = {airs_pixels:,} pixels")
        print(f"Size ratio: {airs_pixels/fgs1_pixels:.1f}x larger")
        
        return airs_shape, airs_pixels
        
    except Exception as e:
        print(f"Error analyzing detector: {e}")
        return None, None

if __name__ == "__main__":
    analyze_airs_data()