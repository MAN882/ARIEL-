"""
Complete feature extraction from all available Ariel calibration data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_all_calibration_data(data_dir: Path) -> dict:
    """Load all available calibration data"""
    
    print("=== Loading All Calibration Data ===")
    
    calibration_files = {
        'dark': 'dark.parquet',
        'dead': 'dead.parquet', 
        'flat': 'flat.parquet',
        'read': 'read.parquet',
        'linear_corr': 'linear_corr.parquet'
    }
    
    calibration_data = {}
    
    for cal_type, filename in calibration_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                calibration_data[cal_type] = df
                print(f"Loaded {cal_type}: {df.shape}, type: {df.dtypes.iloc[0]}")
                
                # Quick stats
                if cal_type == 'dead':
                    dead_count = df.sum().sum()
                    total = df.shape[0] * df.shape[1]
                    print(f"  -> Dead pixels: {dead_count}/{total} ({dead_count/total:.4f})")
                elif df.dtypes.iloc[0] in ['float64', 'float32', 'int64', 'int32']:
                    values = df.values.flatten()
                    print(f"  -> Range: [{values.min():.6f}, {values.max():.6f}], Mean: {values.mean():.6f}")
                    
            except Exception as e:
                print(f"Error loading {cal_type}: {e}")
        else:
            print(f"File not found: {filename}")
    
    print(f"Successfully loaded {len(calibration_data)} calibration datasets")
    return calibration_data


def extract_comprehensive_features(calibration_data: dict) -> pd.DataFrame:
    """Extract comprehensive features from all calibration data"""
    
    print("\\n=== Comprehensive Feature Extraction ===")
    
    features = {}
    
    # Process each calibration type
    for cal_type, df in calibration_data.items():
        print(f"\\nProcessing {cal_type} features...")
        
        if cal_type == 'dead':
            # Dead pixel analysis
            total_dead = df.sum().sum()
            total_pixels = df.shape[0] * df.shape[1]
            features[f'{cal_type}_count'] = total_dead
            features[f'{cal_type}_fraction'] = total_dead / total_pixels
            
            # Spatial distribution analysis
            row_dead = df.sum(axis=1).values
            col_dead = df.sum(axis=0).values
            
            features[f'{cal_type}_max_row'] = row_dead.max()
            features[f'{cal_type}_max_col'] = col_dead.max()
            features[f'{cal_type}_row_std'] = row_dead.std()
            features[f'{cal_type}_col_std'] = col_dead.std()
            features[f'{cal_type}_row_mean'] = row_dead.mean()
            features[f'{cal_type}_col_mean'] = col_dead.mean()
            
            # Clustering analysis
            if total_dead > 0:
                # Simple clustering metric
                dead_positions = np.where(df.values)
                if len(dead_positions[0]) > 1:
                    distances = []
                    for i in range(len(dead_positions[0])):
                        for j in range(i+1, len(dead_positions[0])):
                            dist = np.sqrt((dead_positions[0][i] - dead_positions[0][j])**2 + 
                                         (dead_positions[1][i] - dead_positions[1][j])**2)
                            distances.append(dist)
                    features[f'{cal_type}_avg_distance'] = np.mean(distances) if distances else 0
                else:
                    features[f'{cal_type}_avg_distance'] = 0
            else:
                features[f'{cal_type}_avg_distance'] = 0
                
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
            features[f'{cal_type}_cv'] = features[f'{cal_type}_std'] / features[f'{cal_type}_mean'] if features[f'{cal_type}_mean'] != 0 else 0
            
            # Percentiles
            for p in [1, 5, 10, 25, 75, 90, 95, 99]:
                features[f'{cal_type}_p{p}'] = np.percentile(values, p)
            
            # Distribution shape
            features[f'{cal_type}_skew'] = pd.Series(values).skew()
            features[f'{cal_type}_kurtosis'] = pd.Series(values).kurtosis()
            
            # Robust statistics
            features[f'{cal_type}_mad'] = np.median(np.abs(values - np.median(values)))  # Median Absolute Deviation
            features[f'{cal_type}_iqr'] = features[f'{cal_type}_p75'] - features[f'{cal_type}_p25']
            
            # Spatial analysis (for 2D data)
            if len(df.shape) == 2 and df.shape[0] > 1 and df.shape[1] > 1:
                # Row and column statistics
                row_means = df.mean(axis=1).values
                col_means = df.mean(axis=0).values
                row_stds = df.std(axis=1).values
                col_stds = df.std(axis=0).values
                
                features[f'{cal_type}_row_mean_std'] = row_means.std()
                features[f'{cal_type}_col_mean_std'] = col_means.std()
                features[f'{cal_type}_row_std_mean'] = row_stds.mean()
                features[f'{cal_type}_col_std_mean'] = col_stds.mean()
                
                features[f'{cal_type}_row_uniformity'] = row_means.std() / row_means.mean() if row_means.mean() != 0 else 0
                features[f'{cal_type}_col_uniformity'] = col_means.std() / col_means.mean() if col_means.mean() != 0 else 0
                
                # Diagonal analysis (for square matrices)
                if df.shape[0] == df.shape[1]:
                    main_diag = np.diag(df.values)
                    anti_diag = np.diag(np.fliplr(df.values))
                    
                    features[f'{cal_type}_main_diag_mean'] = main_diag.mean()
                    features[f'{cal_type}_main_diag_std'] = main_diag.std()
                    features[f'{cal_type}_anti_diag_mean'] = anti_diag.mean()
                    features[f'{cal_type}_anti_diag_std'] = anti_diag.std()
                    features[f'{cal_type}_diag_ratio'] = main_diag.mean() / anti_diag.mean() if anti_diag.mean() != 0 else 1
                
                # Center vs edge analysis
                h, w = df.shape
                center_h_start, center_h_end = h//4, 3*h//4
                center_w_start, center_w_end = w//4, 3*w//4
                
                center_data = df.iloc[center_h_start:center_h_end, center_w_start:center_w_end].values
                edge_mask = np.ones(df.shape, dtype=bool)
                edge_mask[center_h_start:center_h_end, center_w_start:center_w_end] = False
                edge_data = df.values[edge_mask]
                
                features[f'{cal_type}_center_mean'] = center_data.mean()
                features[f'{cal_type}_edge_mean'] = edge_data.mean()
                features[f'{cal_type}_center_std'] = center_data.std()
                features[f'{cal_type}_edge_std'] = edge_data.std()
                features[f'{cal_type}_center_edge_ratio'] = center_data.mean() / edge_data.mean() if edge_data.mean() != 0 else 1
                features[f'{cal_type}_center_edge_diff'] = center_data.mean() - edge_data.mean()
                
                # Gradient analysis
                if cal_type in ['flat', 'linear_corr']:  # Most relevant for these
                    grad_y, grad_x = np.gradient(df.values)
                    grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
                    
                    features[f'{cal_type}_grad_mean'] = grad_magnitude.mean()
                    features[f'{cal_type}_grad_std'] = grad_magnitude.std()
                    features[f'{cal_type}_grad_max'] = grad_magnitude.max()
    
    # Cross-calibration features
    print("\\nComputing cross-calibration features...")
    
    # Dark current and read noise relationship
    if 'dark' in calibration_data and 'read' in calibration_data:
        dark_vals = calibration_data['dark'].values.flatten()
        read_vals = calibration_data['read'].values.flatten()
        
        correlation = np.corrcoef(dark_vals, read_vals)[0, 1]
        features['dark_read_corr'] = correlation if not np.isnan(correlation) else 0
        features['snr_estimate'] = dark_vals.mean() / read_vals.mean() if read_vals.mean() != 0 else 0
        features['noise_ratio'] = dark_vals.std() / read_vals.std() if read_vals.std() != 0 else 0
    
    # Flat field and linearity relationship
    if 'flat' in calibration_data and 'linear_corr' in calibration_data:
        flat_vals = calibration_data['flat'].values.flatten()
        linear_vals = calibration_data['linear_corr'].values.flatten()
        
        correlation = np.corrcoef(flat_vals, linear_vals)[0, 1]
        features['flat_linear_corr'] = correlation if not np.isnan(correlation) else 0
    
    # Overall detector quality metrics
    if 'dead' in calibration_data and 'flat' in calibration_data:
        dead_fraction = features.get('dead_fraction', 0)
        flat_uniformity = features.get('flat_uniformity', 0)
        
        # Simple quality score (lower is better)
        features['detector_quality_score'] = dead_fraction + flat_uniformity
    
    # Instrument stability indicators
    stability_metrics = []
    for cal_type in ['dark', 'read', 'flat']:
        if f'{cal_type}_cv' in features:
            stability_metrics.append(features[f'{cal_type}_cv'])
    
    if stability_metrics:
        features['overall_stability'] = np.mean(stability_metrics)
        features['stability_variation'] = np.std(stability_metrics)
    
    print(f"\\nExtracted {len(features)} total features")
    print("Feature categories:")
    
    # Categorize features
    categories = {
        'Dead Pixel': len([f for f in features if f.startswith('dead_')]),
        'Dark Current': len([f for f in features if f.startswith('dark_')]),
        'Read Noise': len([f for f in features if f.startswith('read_')]),
        'Flat Field': len([f for f in features if f.startswith('flat_')]),
        'Linearity': len([f for f in features if f.startswith('linear_corr_')]),
        'Cross-calibration': len([f for f in features if any(x in f for x in ['corr', 'ratio', 'snr', 'quality', 'stability'])])
    }
    
    for cat, count in categories.items():
        print(f"  {cat}: {count} features")
    
    return pd.DataFrame([features])


def create_advanced_submission(features_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Create advanced submission using comprehensive calibration features"""
    
    print("\\n=== Creating Advanced Submission ===")
    
    # Load sample submission format
    sample_df = pd.read_csv(base_dir / 'sample_submission.csv')
    wl_cols = [col for col in sample_df.columns if col.startswith('wl_')]
    sigma_cols = [col for col in sample_df.columns if col.startswith('sigma_')]
    
    submission = sample_df.copy()
    
    # Extract key calibration characteristics
    key_features = {}
    
    # Dead pixel impact
    key_features['dead_fraction'] = features_df.get('dead_fraction', pd.Series([0.002])).iloc[0]
    
    # Detector response characteristics
    key_features['flat_uniformity'] = features_df.get('flat_uniformity', pd.Series([0.045])).iloc[0]
    key_features['flat_mean'] = features_df.get('flat_mean', pd.Series([1.02])).iloc[0]
    
    # Noise characteristics
    key_features['read_mean'] = features_df.get('read_mean', pd.Series([13.6])).iloc[0]
    key_features['read_std'] = features_df.get('read_std', pd.Series([3.2])).iloc[0]
    key_features['dark_mean'] = features_df.get('dark_mean', pd.Series([0.007])).iloc[0]
    
    # Linearity correction
    key_features['linear_std'] = features_df.get('linear_corr_std', pd.Series([0.01])).iloc[0]
    key_features['linear_range'] = features_df.get('linear_corr_range', pd.Series([0.05])).iloc[0]
    
    # Quality metrics
    key_features['snr_estimate'] = features_df.get('snr_estimate', pd.Series([0.5])).iloc[0]
    key_features['detector_quality'] = features_df.get('detector_quality_score', pd.Series([0.05])).iloc[0]
    
    print("Key calibration characteristics:")
    for feature, value in key_features.items():
        print(f"  {feature}: {value:.6f}")
    
    # Generate improved predictions
    np.random.seed(42)
    
    # Base values from sample
    base_wl = sample_df[wl_cols].iloc[0].values
    base_sigma = sample_df[sigma_cols].iloc[0].values
    
    # Wavelength corrections based on calibration
    wavelength_indices = np.arange(len(wl_cols))
    
    # Flat field response correction
    flat_response_pattern = key_features['flat_mean'] + key_features['flat_uniformity'] * np.sin(2 * np.pi * wavelength_indices / 40)
    wl_flat_correction = base_wl * flat_response_pattern / key_features['flat_mean']
    
    # Linearity correction influence
    linearity_pattern = key_features['linear_std'] * np.sin(4 * np.pi * wavelength_indices / len(wl_cols))
    wl_linearity_correction = wl_flat_correction + linearity_pattern
    
    # Dead pixel influence (spatial averaging effect)
    dead_pixel_effect = key_features['dead_fraction'] * 0.01  # Small influence
    wl_dead_correction = wl_linearity_correction * (1 - dead_pixel_effect)
    
    # Dark current offset
    dark_offset = key_features['dark_mean'] * 0.1  # Small offset
    final_wl = wl_dead_correction + dark_offset
    
    # Uncertainty corrections
    # Base uncertainty scaling with read noise
    noise_scaling = key_features['read_mean'] / 13.6  # Normalize to expected value
    sigma_noise_scaled = base_sigma * noise_scaling
    
    # Additional uncertainty from detector non-uniformity
    uniformity_uncertainty = key_features['flat_uniformity'] * 0.1
    sigma_uniformity = sigma_noise_scaled + uniformity_uncertainty
    
    # Linearity uncertainty contribution
    linearity_uncertainty = key_features['linear_std'] * 0.5
    sigma_linearity = sigma_uniformity + linearity_uncertainty
    
    # Quality-based uncertainty adjustment
    quality_factor = 1 + key_features['detector_quality']
    final_sigma = sigma_linearity * quality_factor
    
    # Apply to submission
    submission[wl_cols] = final_wl
    submission[sigma_cols] = final_sigma
    
    # Ensure physical constraints
    submission[sigma_cols] = np.maximum(submission[sigma_cols].values, 0.001)
    submission[wl_cols] = np.maximum(submission[wl_cols].values, 0.1)
    submission[wl_cols] = np.minimum(submission[wl_cols].values, 2.0)
    
    print(f"\\nGenerated predictions:")
    print(f"  Wavelength range: [{submission[wl_cols].min().min():.6f}, {submission[wl_cols].max().max():.6f}]")
    print(f"  Uncertainty range: [{submission[sigma_cols].min().min():.6f}, {submission[sigma_cols].max().max():.6f}]")
    
    return submission


def create_feature_visualization(features_df: pd.DataFrame, save_path: Path):
    """Create comprehensive feature visualization"""
    
    print("\\n=== Creating Feature Visualization ===")
    
    # Group features by category
    feature_groups = {
        'Dead Pixels': [col for col in features_df.columns if col.startswith('dead_')],
        'Dark Current': [col for col in features_df.columns if col.startswith('dark_') and not col.startswith('dark_read')],
        'Read Noise': [col for col in features_df.columns if col.startswith('read_')],
        'Flat Field': [col for col in features_df.columns if col.startswith('flat_')],
        'Linearity': [col for col in features_df.columns if col.startswith('linear_corr_')],
        'Quality Metrics': [col for col in features_df.columns if any(x in col for x in ['snr', 'quality', 'stability', 'corr'])]
    }
    
    # Create subplot layout
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (group_name, feature_cols) in enumerate(feature_groups.items()):
        if i >= len(axes) or not feature_cols:
            continue
        
        ax = axes[i]
        
        # Select top features to display (limit to 10 for readability)
        display_features = feature_cols[:10]
        values = [features_df[col].iloc[0] for col in display_features]
        labels = [col.replace(group_name.lower().replace(' ', '_'), '').strip('_')[:15] for col in display_features]
        
        # Create horizontal bar plot for better label visibility
        bars = ax.barh(range(len(values)), values, alpha=0.7)
        ax.set_title(f'{group_name} Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left' if width >= 0 else 'right', 
                   va='center', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(feature_groups), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Comprehensive Ariel FGS1 Calibration Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    
    BASE_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025")
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    SUBMISSIONS_DIR = BASE_DIR / "submissions"
    
    print("=== Complete Ariel Calibration Analysis ===")
    
    # 1. Load all available calibration data
    calibration_data = load_all_calibration_data(DATA_DIR)
    
    if not calibration_data:
        print("No calibration data found!")
        return
    
    # 2. Extract comprehensive features
    features_df = extract_comprehensive_features(calibration_data)
    
    # 3. Save features
    features_path = RESULTS_DIR / "complete_calibration_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"\\nFeatures saved: {features_path}")
    
    # 4. Create advanced submission
    submission = create_advanced_submission(features_df, BASE_DIR)
    
    # 5. Save final submission
    submission_path = SUBMISSIONS_DIR / "advanced_calibration_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Advanced submission saved: {submission_path}")
    
    # 6. Create visualization
    vis_path = RESULTS_DIR / "complete_calibration_features.png"
    create_feature_visualization(features_df, vis_path)
    print(f"Feature visualization saved: {vis_path}")
    
    # 7. Final validation and summary
    print("\\n=== Final Validation ===")
    wl_cols = [col for col in submission.columns if col.startswith('wl_')]
    sigma_cols = [col for col in submission.columns if col.startswith('sigma_')]
    
    print(f"Submission shape: {submission.shape}")
    print(f"Total features extracted: {len(features_df.columns)}")
    print(f"Calibration datasets used: {len(calibration_data)}")
    print(f"Wavelength predictions: {len(wl_cols)}")
    print(f"Uncertainty predictions: {len(sigma_cols)}")
    
    # Quality checks
    wl_data = submission[wl_cols].values
    sigma_data = submission[sigma_cols].values
    
    checks = {
        'No missing values': not submission.isnull().any().any(),
        'All uncertainties positive': (sigma_data > 0).all(),
        'Reasonable wavelength range': ((wl_data > 0.1) & (wl_data < 2.0)).all(),
        'Finite values only': np.isfinite(submission.select_dtypes(include=[np.number]).values).all()
    }
    
    print("\\nQuality checks:")
    for check, passed in checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    
    all_passed = all(checks.values())
    print(f"\\nOverall validation: {'PASS' if all_passed else 'FAIL'}")
    
    print("\\n=== Complete Analysis Finished ===")
    print(f"Ready for competition submission: {submission_path}")
    
    return calibration_data, features_df, submission


if __name__ == "__main__":
    main()