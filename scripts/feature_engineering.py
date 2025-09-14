"""
Ariel Data Challenge 2025 - Feature Engineering
系外惑星観測データの特徴量エンジニアリング
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class CalibrationFeatureExtractor:
    """校正データから特徴量を抽出するクラス"""
    
    def __init__(self):
        self.feature_names = []
        self.statistics = {}
    
    def extract_pixel_statistics(self, calibration_data: np.ndarray, 
                               prefix: str = "") -> Dict[str, float]:
        """ピクセル統計特徴量の抽出"""
        features = {}
        
        # 基本統計量
        features[f'{prefix}_mean'] = np.mean(calibration_data)
        features[f'{prefix}_std'] = np.std(calibration_data)
        features[f'{prefix}_median'] = np.median(calibration_data)
        features[f'{prefix}_min'] = np.min(calibration_data)
        features[f'{prefix}_max'] = np.max(calibration_data)
        features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
        
        # パーセンタイル
        for p in [10, 25, 75, 90]:
            features[f'{prefix}_p{p}'] = np.percentile(calibration_data, p)
        
        # 分布の形状
        features[f'{prefix}_skewness'] = skew(calibration_data.flatten())
        features[f'{prefix}_kurtosis'] = kurtosis(calibration_data.flatten())
        
        # ピクセル値の分散（空間的）
        if calibration_data.ndim > 1:
            features[f'{prefix}_spatial_var'] = np.var(np.mean(calibration_data, axis=0))
        
        return features
    
    def extract_noise_characteristics(self, read_noise: np.ndarray,
                                    dark_current: np.ndarray) -> Dict[str, float]:
        """ノイズ特性の抽出"""
        features = {}
        
        # 読み出しノイズ特性
        features['read_noise_level'] = np.mean(read_noise)
        features['read_noise_uniformity'] = np.std(read_noise) / np.mean(read_noise)
        
        # ダーク電流特性
        features['dark_current_level'] = np.mean(dark_current)
        features['dark_current_variation'] = np.std(dark_current)
        
        # ホットピクセル（異常に高い値）の検出
        dark_threshold = np.percentile(dark_current, 95)
        features['hot_pixel_fraction'] = np.sum(dark_current > dark_threshold) / dark_current.size
        
        return features
    
    def extract_flatfield_features(self, flatfield: np.ndarray) -> Dict[str, float]:
        """フラットフィールド補正から特徴量を抽出"""
        features = {}
        
        # フラットフィールドの均一性
        features['flatfield_uniformity'] = np.std(flatfield) / np.mean(flatfield)
        features['flatfield_mean'] = np.mean(flatfield)
        
        # 2D構造解析（空間依存性）
        if flatfield.ndim == 2:
            # 行・列方向の変動
            row_variation = np.std(np.mean(flatfield, axis=1))
            col_variation = np.std(np.mean(flatfield, axis=0))
            features['flatfield_row_variation'] = row_variation
            features['flatfield_col_variation'] = col_variation
            
            # 対角線プロファイル
            diag_profile = np.diag(flatfield)
            features['flatfield_diag_variation'] = np.std(diag_profile)
        
        return features
    
    def extract_linearity_features(self, linearity_correction: np.ndarray) -> Dict[str, float]:
        """線形性補正から特徴量を抽出"""
        features = {}
        
        # 線形性の偏差
        features['linearity_deviation'] = np.mean(np.abs(linearity_correction))
        features['linearity_max_deviation'] = np.max(np.abs(linearity_correction))
        
        # 非線形性の空間パターン
        if linearity_correction.ndim == 2:
            features['linearity_spatial_pattern'] = np.std(linearity_correction)
        
        return features


class SpectralFeatureExtractor:
    """分光データの特徴量抽出クラス"""
    
    def extract_spectral_features(self, wavelengths: np.ndarray, 
                                spectrum: np.ndarray) -> Dict[str, float]:
        """分光データから特徴量を抽出"""
        features = {}
        
        # 基本的な分光統計量
        features['spectrum_mean'] = np.mean(spectrum)
        features['spectrum_std'] = np.std(spectrum)
        features['spectrum_median'] = np.median(spectrum)
        features['spectrum_min'] = np.min(spectrum)
        features['spectrum_max'] = np.max(spectrum)
        
        # スペクトル勾配
        gradient = np.gradient(spectrum)
        features['spectrum_gradient_mean'] = np.mean(gradient)
        features['spectrum_gradient_std'] = np.std(gradient)
        
        # ピーク検出
        peaks, _ = signal.find_peaks(spectrum, height=np.mean(spectrum))
        features['spectrum_num_peaks'] = len(peaks)
        
        if len(peaks) > 0:
            features['spectrum_peak_height_mean'] = np.mean(spectrum[peaks])
            features['spectrum_peak_height_std'] = np.std(spectrum[peaks])
        else:
            features['spectrum_peak_height_mean'] = 0
            features['spectrum_peak_height_std'] = 0
        
        # 分光幅の特徴量
        if len(wavelengths) == len(spectrum):
            # 重心波長
            features['spectrum_centroid'] = np.average(wavelengths, weights=spectrum)
            
            # 分光幅（標準偏差）
            features['spectrum_width'] = np.sqrt(np.average((wavelengths - features['spectrum_centroid'])**2, 
                                                         weights=spectrum))
        
        # フーリエ変換による周波数成分
        fft = np.fft.fft(spectrum)
        fft_power = np.abs(fft)**2
        features['spectrum_fft_peak'] = np.max(fft_power[1:len(fft_power)//2])
        
        return features


class AdvancedFeatureEngineering:
    """高度な特徴量エンジニアリング"""
    
    def __init__(self, n_components_pca: int = 50):
        self.n_components_pca = n_components_pca
        self.pca = None
        self.ica = None
        self.feature_selector = None
        self.scaler = None
        
    def apply_dimensionality_reduction(self, X: pd.DataFrame, 
                                     method: str = 'pca') -> Tuple[np.ndarray, Any]:
        """次元削減を適用"""
        
        if method == 'pca':
            if self.pca is None:
                self.pca = PCA(n_components=self.n_components_pca)
                X_reduced = self.pca.fit_transform(X)
                print(f"PCA applied: {X.shape} -> {X_reduced.shape}")
                print(f"Cumulative explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
            else:
                X_reduced = self.pca.transform(X)
            
            return X_reduced, self.pca
            
        elif method == 'ica':
            if self.ica is None:
                self.ica = FastICA(n_components=self.n_components_pca, random_state=42)
                X_reduced = self.ica.fit_transform(X)
                print(f"ICA applied: {X.shape} -> {X_reduced.shape}")
            else:
                X_reduced = self.ica.transform(X)
            
            return X_reduced, self.ica
        
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'f_regression', k: int = 100) -> pd.DataFrame:
        """特徴量選択"""
        
        if method == 'f_regression':
            if self.feature_selector is None:
                self.feature_selector = SelectKBest(f_regression, k=k)
                X_selected = self.feature_selector.fit_transform(X, y)
                selected_features = X.columns[self.feature_selector.get_support()]
                print(f"Feature selection using F-regression: {X.shape[1]} -> {len(selected_features)} features")
            else:
                X_selected = self.feature_selector.transform(X)
                selected_features = X.columns[self.feature_selector.get_support()]
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            print(f"Feature selection using mutual information: {X.shape[1]} -> {len(selected_features)} features")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        elif method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # 重要度でソートして上位k個選択
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(k)['feature'].tolist()
            print(f"Feature selection using Random Forest importance: {X.shape[1]} -> {len(selected_features)} features")
            
            return X[selected_features]
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def create_interaction_features(self, X: pd.DataFrame, 
                                  max_interactions: int = 10) -> pd.DataFrame:
        """相互作用特徴量の作成"""
        
        X_new = X.copy()
        
        # 最も重要な特徴量間の相互作用を作成
        feature_cols = X.columns[:max_interactions]  # 上位N個の特徴量
        
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                col_i, col_j = feature_cols[i], feature_cols[j]
                
                # 積
                X_new[f'{col_i}_x_{col_j}'] = X[col_i] * X[col_j]
                
                # 比率（ゼロ除算回避）
                X_new[f'{col_i}_div_{col_j}'] = X[col_i] / (X[col_j] + 1e-8)
                
                # 差
                X_new[f'{col_i}_diff_{col_j}'] = X[col_i] - X[col_j]
        
        print(f"Interaction features created: {X.shape[1]} -> {X_new.shape[1]} features")
        return X_new
    
    def apply_scaling(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """スケーリングを適用"""
        
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
                
        elif method == 'robust':
            if self.scaler is None:
                self.scaler = RobustScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
                
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def create_mock_calibration_features(n_samples: int = 100) -> pd.DataFrame:
    """模擬校正特徴量データを作成"""
    
    np.random.seed(42)
    
    # 校正特徴量抽出器
    extractor = CalibrationFeatureExtractor()
    
    features_list = []
    
    for i in range(n_samples):
        # 模擬校正データを生成
        dark_data = np.random.exponential(scale=10, size=(64, 64)) + np.random.normal(0, 2, (64, 64))
        read_noise = np.random.normal(5, 1, (64, 64))
        flatfield = 1.0 + np.random.normal(0, 0.05, (64, 64))
        linearity_corr = np.random.normal(0, 0.02, (64, 64))
        
        # 特徴量抽出
        features = {}
        
        # ピクセル統計（各校正データから）
        features.update(extractor.extract_pixel_statistics(dark_data, 'dark'))
        features.update(extractor.extract_pixel_statistics(read_noise, 'read'))
        features.update(extractor.extract_pixel_statistics(flatfield, 'flat'))
        features.update(extractor.extract_pixel_statistics(linearity_corr, 'linear'))
        
        # ノイズ特性
        features.update(extractor.extract_noise_characteristics(read_noise, dark_data))
        
        # フラットフィールド特性
        features.update(extractor.extract_flatfield_features(flatfield))
        
        # 線形性特性
        features.update(extractor.extract_linearity_features(linearity_corr))
        
        features_list.append(features)
    
    # データフレーム化
    features_df = pd.DataFrame(features_list)
    
    print(f"Calibration features generated: {features_df.shape}")
    print(f"Example features: {list(features_df.columns)[:10]}")
    
    return features_df


def main():
    """Main execution function"""
    
    print("=== Ariel Data Challenge 2025 - Feature Engineering ===")
    
    # Generate mock calibration features
    print("\n1. Feature extraction from calibration data")
    calibration_features = create_mock_calibration_features(n_samples=200)
    
    # Load sample spectral data (target variable)
    sample_df = pd.read_csv("../sample_submission.csv")
    wl_cols = [col for col in sample_df.columns if col.startswith('wl_')]
    
    # Create mock target variable (in reality, predict all wavelengths)
    y_sample = np.random.normal(0.456, 0.01, len(calibration_features))  # One wavelength sample
    
    print(f"Target variable sample: shape={y_sample.shape}")
    
    # Advanced feature engineering
    print("\n2. Advanced feature engineering")
    
    feature_engineer = AdvancedFeatureEngineering(n_components_pca=30)
    
    # Scaling
    print("\n2-1. Scaling")
    X_scaled = feature_engineer.apply_scaling(calibration_features, method='standard')
    
    # Feature selection
    print("\n2-2. Feature selection")
    X_selected = feature_engineer.select_features(X_scaled, y_sample, method='f_regression', k=50)
    
    # Interaction features
    print("\n2-3. Creating interaction features")
    X_interactions = feature_engineer.create_interaction_features(X_selected, max_interactions=5)
    
    # Dimensionality reduction
    print("\n2-4. Dimensionality reduction (PCA)")
    X_pca, pca_transformer = feature_engineer.apply_dimensionality_reduction(X_interactions, method='pca')
    
    print(f"\nFinal features: {X_pca.shape}")
    
    # Feature visualization
    print("\n3. Feature visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original feature distribution
    axes[0,0].hist(calibration_features.iloc[:, 0], bins=20, alpha=0.7)
    axes[0,0].set_title('Original Feature Distribution Example')
    axes[0,0].set_xlabel('Value')
    axes[0,0].set_ylabel('Frequency')
    
    # Distribution after scaling
    axes[0,1].hist(X_scaled.iloc[:, 0], bins=20, alpha=0.7)
    axes[0,1].set_title('Distribution After Scaling')
    axes[0,1].set_xlabel('Value')
    axes[0,1].set_ylabel('Frequency')
    
    # Number of features after each step
    axes[1,0].bar(range(3), [calibration_features.shape[1], X_selected.shape[1], X_interactions.shape[1]],
                  tick_label=['Original', 'Selected', 'With Interactions'])
    axes[1,0].set_title('Number of Features at Each Engineering Step')
    axes[1,0].set_ylabel('Number of Features')
    
    # PCA explained variance ratio
    if len(pca_transformer.explained_variance_ratio_) > 10:
        axes[1,1].plot(range(1, 11), pca_transformer.explained_variance_ratio_[:10], 'bo-')
        axes[1,1].set_title('PCA Explained Variance Ratio (Top 10 Components)')
        axes[1,1].set_xlabel('Principal Component')
        axes[1,1].set_ylabel('Explained Variance Ratio')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../results/feature_engineering_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Feature engineering completed ===")
    print(f"Final feature set: {X_pca.shape}")
    print("Figure saved to results/feature_engineering_analysis.png")
    
    return X_pca, y_sample, feature_engineer


if __name__ == "__main__":
    main()