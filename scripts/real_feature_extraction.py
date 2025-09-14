"""
Real feature extraction from Ariel calibration data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_calibration_features(data_dir: Path) -> pd.DataFrame:
    """Extract features from real calibration data"""
    
    print("=== Real Feature Extraction ===")
    
    features = {}
    
    # Load calibration files
    cal_files = {
        'dark': 'dark.parquet',
        'dead': 'dead.parquet',
        'flat': 'flat.parquet', 
        'read': 'read.parquet'
    }
    
    cal_data = {}
    for cal_type, filename in cal_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            cal_data[cal_type] = pd.read_parquet(file_path)
    
    print(f"Loaded {len(cal_data)} calibration files")
    
    # Extract features from each calibration type
    for cal_type, df in cal_data.items():
        print(f"\\nExtracting {cal_type} features...")
        
        if cal_type == 'dead':
            # Dead pixel features
            total_dead = df.sum().sum()
            total_pixels = df.shape[0] * df.shape[1]
            features[f'{cal_type}_count'] = total_dead
            features[f'{cal_type}_fraction'] = total_dead / total_pixels
            
            # Spatial distribution
            row_dead = df.sum(axis=1).values
            col_dead = df.sum(axis=0).values
            features[f'{cal_type}_max_row'] = row_dead.max()
            features[f'{cal_type}_max_col'] = col_dead.max()
            features[f'{cal_type}_row_std'] = row_dead.std()
            features[f'{cal_type}_col_std'] = col_dead.std()
            
        else:
            # Numerical calibration data
            values = df.values.flatten()
            
            # Basic statistics
            features[f'{cal_type}_mean'] = values.mean()
            features[f'{cal_type}_std'] = values.std()
            features[f'{cal_type}_median'] = np.median(values)
            features[f'{cal_type}_min'] = values.min()
            features[f'{cal_type}_max'] = values.max()
            features[f'{cal_type}_range'] = features[f'{cal_type}_max'] - features[f'{cal_type}_min']
            
            # Percentiles
            for p in [10, 25, 75, 90]:
                features[f'{cal_type}_p{p}'] = np.percentile(values, p)
            
            # Distribution shape
            features[f'{cal_type}_skew'] = pd.Series(values).skew()
            features[f'{cal_type}_kurtosis'] = pd.Series(values).kurtosis()
            
            # Spatial uniformity
            row_means = df.mean(axis=1).values
            col_means = df.mean(axis=0).values
            features[f'{cal_type}_row_uniformity'] = row_means.std() / row_means.mean()
            features[f'{cal_type}_col_uniformity'] = col_means.std() / col_means.mean()
            
            # Center vs edge analysis
            center_slice = slice(8, 24)  # Central 16x16 region
            center_data = df.iloc[center_slice, center_slice].values
            edge_mask = np.ones(df.shape, dtype=bool)
            edge_mask[center_slice, center_slice] = False
            edge_data = df.values[edge_mask]
            
            features[f'{cal_type}_center_mean'] = center_data.mean()
            features[f'{cal_type}_edge_mean'] = edge_data.mean()
            features[f'{cal_type}_center_edge_ratio'] = (
                features[f'{cal_type}_center_mean'] / features[f'{cal_type}_edge_mean']
                if features[f'{cal_type}_edge_mean'] != 0 else 0
            )
    
    # Cross-calibration features
    if 'dark' in cal_data and 'read' in cal_data:
        dark_vals = cal_data['dark'].values.flatten()
        read_vals = cal_data['read'].values.flatten()
        
        # Correlation
        correlation = np.corrcoef(dark_vals, read_vals)[0, 1]
        features['dark_read_corr'] = correlation if not np.isnan(correlation) else 0
        
        # SNR estimation
        features['snr_estimate'] = dark_vals.mean() / read_vals.mean()
    
    if 'flat' in cal_data:
        flat_vals = cal_data['flat'].values.flatten()
        features['detector_response_uniformity'] = flat_vals.std() / flat_vals.mean()
    
    print(f"\\nExtracted {len(features)} features total")
    
    return pd.DataFrame([features])


def create_real_submission(features_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Create submission using real calibration features"""
    
    print("\\n=== Creating Real Data Submission ===")
    
    # Load sample submission format
    sample_df = pd.read_csv(base_dir / 'sample_submission.csv')
    
    # Get base values
    wl_cols = [col for col in sample_df.columns if col.startswith('wl_')]
    sigma_cols = [col for col in sample_df.columns if col.startswith('sigma_')]
    
    submission = sample_df.copy()
    
    # Use extracted features to modify predictions
    # This is a simplified approach - in reality you'd train ML models
    
    np.random.seed(42)
    
    # Extract key calibration characteristics
    if 'flat_response_uniformity' in features_df.columns:
        flat_uniformity = features_df['detector_response_uniformity'].iloc[0]
    else:
        flat_uniformity = 0.05  # Default
    
    if 'read_mean' in features_df.columns:
        read_noise_level = features_df['read_mean'].iloc[0]
    else:
        read_noise_level = 13.0  # Default based on observed data
    
    if 'dark_mean' in features_df.columns:
        dark_level = features_df['dark_mean'].iloc[0]  
    else:
        dark_level = 0.007  # Default
    
    print(f"Using calibration characteristics:")
    print(f"  Flat field uniformity: {flat_uniformity:.6f}")
    print(f"  Read noise level: {read_noise_level:.6f}")
    print(f"  Dark current level: {dark_level:.6f}")
    
    # Modify base prediction using calibration info
    base_wl = sample_df[wl_cols].iloc[0].values
    base_sigma = sample_df[sigma_cols].iloc[0].values
    
    # Wavelength corrections based on flat field
    wl_correction = flat_uniformity * np.sin(2 * np.pi * np.arange(len(wl_cols)) / 50)
    corrected_wl = base_wl * (1 + wl_correction)
    
    # Uncertainty scaling based on read noise
    noise_scaling = read_noise_level / 13.0  # Normalize to expected value
    scaled_sigma = base_sigma * noise_scaling
    
    # Add dark current influence
    dark_influence = dark_level * 0.1  # Small influence
    final_wl = corrected_wl + dark_influence
    
    # Apply to submission
    submission[wl_cols] = final_wl
    submission[sigma_cols] = scaled_sigma
    
    # Ensure physical constraints
    submission[sigma_cols] = np.maximum(submission[sigma_cols].values, 0.001)
    
    return submission


def visualize_features(features_df: pd.DataFrame, save_path: Path):
    """Visualize extracted features"""
    
    print("\\n=== Feature Visualization ===")
    
    # Select interesting features for visualization
    feature_groups = {
        'Dark Current': [col for col in features_df.columns if col.startswith('dark_')],
        'Read Noise': [col for col in features_df.columns if col.startswith('read_')],
        'Flat Field': [col for col in features_df.columns if col.startswith('flat_')],
        'Dead Pixels': [col for col in features_df.columns if col.startswith('dead_')]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (group_name, feature_cols) in enumerate(feature_groups.items()):
        if i >= len(axes) or not feature_cols:
            continue
        
        ax = axes[i]
        
        # Get feature values
        values = [features_df[col].iloc[0] for col in feature_cols]
        labels = [col.replace(group_name.lower().replace(' ', '_'), '').strip('_') 
                 for col in feature_cols]
        
        # Create bar plot
        bars = ax.bar(range(len(values)), values, alpha=0.7)
        ax.set_title(f'{group_name} Features')
        ax.set_ylabel('Value')
        
        # Rotate x-labels if too many
        if len(labels) > 3:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
        
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(feature_groups), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Real Calibration Features - FGS1 Detector', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution"""
    
    BASE_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025")
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    SUBMISSIONS_DIR = BASE_DIR / "submissions"
    
    print("=== Real Data Feature Extraction & Submission ===")
    
    # 1. Extract features from real calibration data
    features_df = extract_calibration_features(DATA_DIR)
    
    # 2. Save features
    features_path = RESULTS_DIR / "real_calibration_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"\\nFeatures saved: {features_path}")
    
    # 3. Create submission based on real data
    submission = create_real_submission(features_df, BASE_DIR)
    
    # 4. Save submission
    submission_path = SUBMISSIONS_DIR / "real_data_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved: {submission_path}")
    
    # 5. Visualize features
    vis_path = RESULTS_DIR / "real_calibration_features.png"
    visualize_features(features_df, vis_path)
    print(f"Visualization saved: {vis_path}")
    
    # 6. Validation
    print("\\n=== Submission Validation ===")
    wl_cols = [col for col in submission.columns if col.startswith('wl_')]
    sigma_cols = [col for col in submission.columns if col.startswith('sigma_')]
    
    wl_data = submission[wl_cols].values
    sigma_data = submission[sigma_cols].values
    
    print(f"Shape: {submission.shape}")
    print(f"Wavelength range: {wl_data.min():.6f} to {wl_data.max():.6f}")
    print(f"Uncertainty range: {sigma_data.min():.6f} to {sigma_data.max():.6f}")
    print(f"All uncertainties positive: {(sigma_data > 0).all()}")
    print(f"No missing values: {not submission.isnull().any().any()}")
    
    print("\\n=== Real Data Analysis Complete ===")
    return features_df, submission


if __name__ == "__main__":
    main()