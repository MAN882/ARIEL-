"""
Ariel Data Challenge 2025 - Real Data Loader
実際のコンペデータの読み込みと分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ArielDataLoader:
    """Arielコンペデータローダー"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.planet_data = {}
        
    def explore_data_structure(self):
        """データ構造の探索"""
        print("=== Ariel Data Structure Exploration ===")
        
        # 利用可能なファイルを確認
        parquet_files = list(self.data_dir.glob("**/*.parquet"))
        print(f"Found {len(parquet_files)} parquet files")
        
        if parquet_files:
            # 最初のファイルを読み込んでデータ構造を確認
            sample_file = parquet_files[0]
            print(f"\\nAnalyzing sample file: {sample_file}")
            
            try:
                df = pd.read_parquet(sample_file)
                print(f"Sample data shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"Data types: {df.dtypes}")
                print(f"\\nFirst few rows:")
                print(df.head())
                
                # 統計サマリー
                print(f"\\nStatistical summary:")
                print(df.describe())
                
                return df
                
            except Exception as e:
                print(f"Error reading {sample_file}: {e}")
                return None
        else:
            print("No parquet files found in data directory")
            return None
    
    def download_calibration_files(self, planet_id: str = "1103775", 
                                 instrument: str = "FGS1", 
                                 calibration_set: str = "calibration_0"):
        """校正ファイルのダウンロード"""
        
        print(f"\\nDownloading calibration files for {planet_id}/{instrument}/{calibration_set}")
        
        calibration_types = ["dark", "dead", "flat", "linear_corr", "read"]
        downloaded_files = {}
        
        for cal_type in calibration_types:
            file_path = f"test/{planet_id}/{instrument}_{calibration_set}/{cal_type}.parquet"
            
            try:
                # Kaggle CLIでダウンロード
                import subprocess
                result = subprocess.run([
                    'kaggle', 'competitions', 'download', '-c', 'ariel-data-challenge-2025', 
                    '-f', file_path, '-p', str(self.data_dir)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    downloaded_file = self.data_dir / f"{cal_type}.parquet"
                    if downloaded_file.exists():
                        downloaded_files[cal_type] = downloaded_file
                        print(f"OK Downloaded {cal_type}.parquet")
                    else:
                        print(f"ERROR Download reported success but file not found: {cal_type}")
                else:
                    print(f"ERROR Failed to download {cal_type}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"ERROR Timeout downloading {cal_type}")
            except Exception as e:
                print(f"ERROR Error downloading {cal_type}: {e}")
        
        return downloaded_files
    
    def load_calibration_data(self, files_dict: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
        """校正データの読み込み"""
        
        print(f"\\nLoading calibration data...")
        calibration_data = {}
        
        for cal_type, file_path in files_dict.items():
            try:
                df = pd.read_parquet(file_path)
                calibration_data[cal_type] = df
                print(f"OK Loaded {cal_type}: {df.shape}")
                
                # データサンプルを表示
                print(f"  Columns: {list(df.columns)[:5]}...")
                print(f"  Data range: [{df.min().min():.6f}, {df.max().max():.6f}]")
                
            except Exception as e:
                print(f"ERROR Error loading {cal_type}: {e}")
        
        return calibration_data
    
    def analyze_calibration_data(self, calibration_data: Dict[str, pd.DataFrame]):
        """校正データの詳細分析"""
        
        print(f"\\n=== Calibration Data Analysis ===")
        
        for cal_type, df in calibration_data.items():
            print(f"\\n--- {cal_type.upper()} Analysis ---")
            
            # 基本統計
            print(f"Shape: {df.shape}")
            print(f"Data type: {df.dtypes.iloc[0]}")
            
            # 数値データの場合の統計
            if df.select_dtypes(include=[np.number]).shape[1] > 0:
                numeric_df = df.select_dtypes(include=[np.number])
                
                print(f"Value range: [{numeric_df.min().min():.6f}, {numeric_df.max().max():.6f}]")
                print(f"Mean: {numeric_df.mean().mean():.6f}")
                print(f"Std: {numeric_df.std().mean():.6f}")
                
                # 異常値検出
                flat_values = numeric_df.values.flatten()
                q99 = np.percentile(flat_values, 99)
                q1 = np.percentile(flat_values, 1)
                outlier_fraction = np.sum((flat_values > q99) | (flat_values < q1)) / len(flat_values)
                print(f"Outlier fraction (beyond 1-99 percentile): {outlier_fraction:.4f}")
                
    def visualize_calibration_data(self, calibration_data: Dict[str, pd.DataFrame], 
                                 save_path: Path = None):
        """校正データの可視化"""
        
        print(f"\\nVisualizing calibration data...")
        
        n_types = len(calibration_data)
        if n_types == 0:
            print("No data to visualize")
            return
        
        # 2D画像として表示可能かチェック
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (cal_type, df) in enumerate(calibration_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # 数値データを取得
                numeric_df = df.select_dtypes(include=[np.number])
                
                if numeric_df.shape[1] == 1:
                    # 1次元データ：ヒストグラム
                    data = numeric_df.iloc[:, 0].values
                    ax.hist(data.flatten(), bins=50, alpha=0.7)
                    ax.set_title(f'{cal_type.upper()} Distribution')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    
                elif numeric_df.shape[1] > 1:
                    # 多次元データ：最初の2列の散布図か、2D形状なら画像
                    if len(df.shape) == 2 and df.shape[0] * df.shape[1] > 100:
                        # 2D画像として表示を試行
                        try:
                            # 正方形に近い形に reshape を試行
                            data = numeric_df.values.flatten()
                            side_length = int(np.sqrt(len(data)))
                            if side_length * side_length == len(data):
                                img_data = data.reshape(side_length, side_length)
                                im = ax.imshow(img_data, cmap='viridis', aspect='auto')
                                ax.set_title(f'{cal_type.upper()} 2D View')
                                plt.colorbar(im, ax=ax)
                            else:
                                # 1D プロット
                                ax.plot(data[:1000])  # 最初の1000点のみ
                                ax.set_title(f'{cal_type.upper()} Signal')
                                ax.set_xlabel('Index')
                                ax.set_ylabel('Value')
                        except:
                            # フォールバック：ヒストグラム
                            ax.hist(numeric_df.values.flatten(), bins=50, alpha=0.7)
                            ax.set_title(f'{cal_type.upper()} Distribution')
                    else:
                        # 散布図
                        if numeric_df.shape[1] >= 2:
                            ax.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], alpha=0.5)
                            ax.set_title(f'{cal_type.upper()} Scatter')
                            ax.set_xlabel(numeric_df.columns[0])
                            ax.set_ylabel(numeric_df.columns[1])
                        else:
                            ax.hist(numeric_df.iloc[:, 0], bins=50, alpha=0.7)
                            ax.set_title(f'{cal_type.upper()} Distribution')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Visualization Error\\n{cal_type}\\n{str(e)[:50]}...', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{cal_type.upper()} (Error)')
        
        # 未使用のサブプロットを非表示
        for i in range(len(calibration_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()


def main():
    """メイン実行関数"""
    
    BASE_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025")
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    
    print("=== Ariel Data Challenge 2025 - Real Data Analysis ===")
    
    # データローダーの初期化
    loader = ArielDataLoader(DATA_DIR)
    
    # 1. 現在のデータ構造を探索
    print("\\n1. Exploring current data structure...")
    sample_data = loader.explore_data_structure()
    
    # 2. 追加の校正ファイルをダウンロード
    print("\\n2. Downloading additional calibration files...")
    downloaded_files = loader.download_calibration_files()
    
    if downloaded_files:
        # 3. 校正データの読み込み
        print("\\n3. Loading calibration data...")
        calibration_data = loader.load_calibration_data(downloaded_files)
        
        # 4. データ分析
        print("\\n4. Analyzing calibration data...")
        loader.analyze_calibration_data(calibration_data)
        
        # 5. 可視化
        print("\\n5. Visualizing calibration data...")
        vis_path = RESULTS_DIR / "real_calibration_data_analysis.png"
        loader.visualize_calibration_data(calibration_data, vis_path)
        
        print(f"\\n=== Real Data Analysis Complete ===")
        print(f"Calibration files loaded: {len(calibration_data)}")
        print(f"Visualization saved: {vis_path}")
        
        return calibration_data
    
    else:
        print("No calibration files were successfully downloaded")
        return None


if __name__ == "__main__":
    main()