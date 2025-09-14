"""
Ariel Data Challenge 2025 - Real Data Analysis
実際のコンペデータの詳細分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def load_calibration_files(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """校正ファイルの読み込み"""
    
    print("Loading real calibration data...")
    
    files = {
        'dark': 'dark.parquet',
        'dead': 'dead.parquet', 
        'flat': 'flat.parquet',
        'read': 'read.parquet'
    }
    
    calibration_data = {}
    
    for cal_type, filename in files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                calibration_data[cal_type] = df
                print(f"Loaded {cal_type}: {df.shape}, dtype: {df.dtypes.iloc[0]}")
            except Exception as e:
                print(f"Error loading {cal_type}: {e}")
        else:
            print(f"File not found: {filename}")
    
    return calibration_data


def analyze_detector_data(calibration_data: Dict[str, pd.DataFrame]):
    """検出器データの詳細分析"""
    
    print("\\n=== Detector Data Analysis ===")
    
    for cal_type, df in calibration_data.items():
        print(f"\\n--- {cal_type.upper()} ---")
        print(f"Shape: {df.shape}")
        print(f"Data type: {df.dtypes.iloc[0]}")
        
        if cal_type == 'dead':
            # デッドピクセル解析
            dead_pixels = df.sum().sum()  # True=1として計算
            total_pixels = df.shape[0] * df.shape[1]
            dead_fraction = dead_pixels / total_pixels
            print(f"Dead pixels: {dead_pixels}/{total_pixels} ({dead_fraction:.4f})")
            
        else:
            # 数値データの統計
            flat_data = df.values.flatten()
            print(f"Min: {flat_data.min():.6f}")
            print(f"Max: {flat_data.max():.6f}")
            print(f"Mean: {flat_data.mean():.6f}")
            print(f"Std: {flat_data.std():.6f}")
            print(f"Median: {np.median(flat_data):.6f}")
            
            # 分布の特徴
            print(f"Skewness: {pd.Series(flat_data).skew():.6f}")
            print(f"Kurtosis: {pd.Series(flat_data).kurtosis():.6f}")


def extract_real_features(calibration_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """実際の校正データから特徴量を抽出"""
    
    print("\\n=== Real Feature Extraction ===")
    
    features = {}
    
    for cal_type, df in calibration_data.items():
        print(f"\\nExtracting features from {cal_type}...")
        
        if cal_type == 'dead':
            # デッドピクセル特徴量
            features[f'{cal_type}_total_dead'] = df.sum().sum()
            features[f'{cal_type}_dead_fraction'] = features[f'{cal_type}_total_dead'] / (df.shape[0] * df.shape[1])
            
            # 空間分布の特徴
            row_dead_counts = df.sum(axis=1)
            col_dead_counts = df.sum(axis=0)
            
            features[f'{cal_type}_max_row_dead'] = row_dead_counts.max()
            features[f'{cal_type}_max_col_dead'] = col_dead_counts.max()
            features[f'{cal_type}_dead_row_var'] = row_dead_counts.var()
            features[f'{cal_type}_dead_col_var'] = col_dead_counts.var()
            
        else:
            # 数値データの特徴量
            flat_data = df.values.flatten()
            
            # 基本統計量
            features[f'{cal_type}_mean'] = flat_data.mean()
            features[f'{cal_type}_std'] = flat_data.std()
            features[f'{cal_type}_median'] = np.median(flat_data)
            features[f'{cal_type}_min'] = flat_data.min()
            features[f'{cal_type}_max'] = flat_data.max()
            features[f'{cal_type}_range'] = features[f'{cal_type}_max'] - features[f'{cal_type}_min']
            
            # パーセンタイル
            for p in [5, 10, 25, 75, 90, 95]:
                features[f'{cal_type}_p{p}'] = np.percentile(flat_data, p)
            
            # 分布の形状
            features[f'{cal_type}_skew'] = pd.Series(flat_data).skew()
            features[f'{cal_type}_kurtosis'] = pd.Series(flat_data).kurtosis()
            
            # 空間的特徴量（2D構造の分析）
            if len(df.shape) == 2:
                # 行・列方向の統計
                row_means = df.mean(axis=1)
                col_means = df.mean(axis=0)
                
                features[f'{cal_type}_row_mean_std'] = row_means.std()
                features[f'{cal_type}_col_mean_std'] = col_means.std()
                features[f'{cal_type}_row_mean_range'] = row_means.max() - row_means.min()
                features[f'{cal_type}_col_mean_range'] = col_means.max() - col_means.min()
                
                # 対角線プロファイル
                if df.shape[0] == df.shape[1]:  # 正方形の場合
                    main_diag = np.diag(df.values)
                    anti_diag = np.diag(np.fliplr(df.values))
                    
                    features[f'{cal_type}_main_diag_mean'] = main_diag.mean()
                    features[f'{cal_type}_main_diag_std'] = main_diag.std()
                    features[f'{cal_type}_anti_diag_mean'] = anti_diag.mean()
                    features[f'{cal_type}_anti_diag_std'] = anti_diag.std()
                
                # 中心と端の比較
                center_region = df.iloc[df.shape[0]//4:3*df.shape[0]//4, 
                                       df.shape[1]//4:3*df.shape[1]//4]
                center_mean = center_region.values.mean()
                edge_mask = np.ones(df.shape, dtype=bool)
                edge_mask[df.shape[0]//4:3*df.shape[0]//4, 
                         df.shape[1]//4:3*df.shape[1]//4] = False
                edge_mean = df.values[edge_mask].mean()
                
                features[f'{cal_type}_center_mean'] = center_mean
                features[f'{cal_type}_edge_mean'] = edge_mean
                features[f'{cal_type}_center_edge_ratio'] = center_mean / edge_mean if edge_mean != 0 else 0
    
    # 相互関係の特徴量
    if 'dark' in calibration_data and 'read' in calibration_data:
        dark_data = calibration_data['dark'].values.flatten()
        read_data = calibration_data['read'].values.flatten()
        
        # 相関
        correlation = np.corrcoef(dark_data, read_data)[0, 1]
        features['dark_read_correlation'] = correlation if not np.isnan(correlation) else 0
        
        # SNR推定
        snr_estimate = np.mean(dark_data) / np.mean(read_data) if np.mean(read_data) != 0 else 0
        features['estimated_snr'] = snr_estimate
    
    if 'flat' in calibration_data:
        flat_data = calibration_data['flat'].values.flatten()
        # フラットフィールドの均一性
        features['flat_uniformity'] = flat_data.std() / flat_data.mean() if flat_data.mean() != 0 else 0
    
    print(f"\\nExtracted {len(features)} features")
    print("Sample features:", list(features.keys())[:10])
    
    # DataFrameに変換（1行のデータ）
    features_df = pd.DataFrame([features])
    
    return features_df


def visualize_real_data(calibration_data: Dict[str, pd.DataFrame], save_path: Path = None):
    """実データの可視化"""
    
    print("\\n=== Real Data Visualization ===")
    
    n_types = len(calibration_data)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (cal_type, df) in enumerate(calibration_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if cal_type == 'dead':
            # デッドピクセルマップ
            im = ax.imshow(df.values.astype(int), cmap='RdYlBu_r', aspect='auto')
            ax.set_title(f'{cal_type.upper()} - Dead Pixel Map\\n({df.sum().sum()} dead pixels)')
            plt.colorbar(im, ax=ax, label='Dead Pixel (1=Dead)')
            
        else:
            # 数値データの2D表示
            im = ax.imshow(df.values, cmap='viridis', aspect='auto')
            ax.set_title(f'{cal_type.upper()} - 2D View\\nRange: [{df.min().min():.3f}, {df.max().max():.3f}]')
            plt.colorbar(im, ax=ax, label='Value')
        
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    # 未使用のサブプロット非表示
    for i in range(len(calibration_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Ariel FGS1 Calibration Data - Real Data Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()


def analyze_instrument_characteristics(calibration_data: Dict[str, pd.DataFrame]):
    """機器特性の詳細分析"""
    
    print("\\n=== Instrument Characteristics Analysis ===")
    
    # 検出器サイズ
    if calibration_data:
        sample_shape = list(calibration_data.values())[0].shape
        print(f"Detector size: {sample_shape[0]} x {sample_shape[1]} pixels")
        print(f"Total pixels: {sample_shape[0] * sample_shape[1]}")
    
    # ノイズ特性
    if 'read' in calibration_data:
        read_noise = calibration_data['read'].values.flatten()
        print(f"\\nRead noise characteristics:")
        print(f"  RMS read noise: {read_noise.std():.6f}")
        print(f"  Mean read noise: {read_noise.mean():.6f}")
        print(f"  Read noise uniformity (CV): {read_noise.std()/read_noise.mean():.6f}")
    
    # ダーク電流
    if 'dark' in calibration_data:
        dark_current = calibration_data['dark'].values.flatten()
        print(f"\\nDark current characteristics:")
        print(f"  Mean dark current: {dark_current.mean():.6f}")
        print(f"  Dark current variation: {dark_current.std():.6f}")
        
        # ホットピクセル検出
        hot_threshold = np.percentile(dark_current, 99)
        hot_pixels = np.sum(dark_current > hot_threshold)
        print(f"  Hot pixels (>99th percentile): {hot_pixels} ({hot_pixels/len(dark_current)*100:.2f}%)")
    
    # フラットフィールド
    if 'flat' in calibration_data:
        flat_field = calibration_data['flat'].values.flatten()
        print(f"\\nFlat field characteristics:")
        print(f"  Flat field uniformity: {flat_field.std()/flat_field.mean():.6f}")
        print(f"  Mean response: {flat_field.mean():.6f}")
        print(f"  Response variation: {flat_field.std():.6f}")
    
    # デッドピクセル分析
    if 'dead' in calibration_data:
        dead_map = calibration_data['dead']
        total_dead = dead_map.sum().sum()
        total_pixels = dead_map.shape[0] * dead_map.shape[1]
        print(f"\\nDead pixel analysis:")
        print(f"  Total dead pixels: {total_dead}")
        print(f"  Dead pixel fraction: {total_dead/total_pixels:.6f}")
        print(f"  Detector yield: {(1-total_dead/total_pixels)*100:.2f}%")


def main():
    """メイン実行関数"""
    
    BASE_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025")
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    
    print("=== Ariel Data Challenge 2025 - Real Data Analysis ===")
    
    # 1. 校正データの読み込み
    calibration_data = load_calibration_files(DATA_DIR)
    
    if not calibration_data:
        print("No calibration data found!")
        return
    
    # 2. 検出器データの分析
    analyze_detector_data(calibration_data)
    
    # 3. 機器特性の詳細分析
    analyze_instrument_characteristics(calibration_data)
    
    # 4. 特徴量抽出
    features_df = extract_real_features(calibration_data)
    
    # 5. 可視化
    vis_path = RESULTS_DIR / "real_calibration_analysis.png"
    visualize_real_data(calibration_data, vis_path)
    
    # 6. 特徴量の保存
    features_path = RESULTS_DIR / "extracted_features.csv"
    features_df.to_csv(features_path, index=False)
    
    print(f"\\n=== Analysis Complete ===")
    print(f"Calibration files analyzed: {len(calibration_data)}")
    print(f"Features extracted: {len(features_df.columns)}")
    print(f"Features saved: {features_path}")
    print(f"Visualization saved: {vis_path}")
    
    return calibration_data, features_df


if __name__ == "__main__":
    main()