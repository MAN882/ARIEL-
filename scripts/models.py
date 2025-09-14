"""
Ariel Data Challenge 2025 - モデル開発
系外惑星大気の分光データ予測モデル
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SpectralDataProcessor:
    """分光データの前処理クラス"""
    
    def __init__(self, n_components_pca: int = 50, use_pca: bool = True):
        self.n_components_pca = n_components_pca
        self.use_pca = use_pca
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components_pca) if use_pca else None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'SpectralDataProcessor':
        """前処理パラメータをフィット"""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.use_pca:
            self.pca.fit(X_scaled)
            
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """データを変換"""
        if not self.is_fitted:
            raise ValueError("Processorはまずfitされる必要があります")
            
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
            
        return X_transformed
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """フィットと変換を同時に実行"""
        return self.fit(X).transform(X)


class WavelengthSigmaPredictor(BaseEstimator, RegressorMixin):
    """波長と不確実性の同時予測モデル"""
    
    def __init__(self, 
                 wl_model=None, 
                 sigma_model=None,
                 processor_params: Dict[str, Any] = None):
        self.wl_model = wl_model
        self.sigma_model = sigma_model
        self.processor_params = processor_params or {}
        self.wl_processor = None
        self.sigma_processor = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """モデルを訓練"""
        # 波長と不確実性のカラムを分離
        wl_cols = [col for col in y.columns if col.startswith('wl_')]
        sigma_cols = [col for col in y.columns if col.startswith('sigma_')]
        
        y_wl = y[wl_cols]
        y_sigma = y[sigma_cols]
        
        # 前処理器の初期化と訓練
        self.wl_processor = SpectralDataProcessor(**self.processor_params)
        self.sigma_processor = SpectralDataProcessor(**self.processor_params)
        
        X_wl = self.wl_processor.fit_transform(X)
        X_sigma = self.sigma_processor.fit_transform(X)
        
        # モデルの訓練
        print(f"波長予測モデルを訓練中... 入力次元: {X_wl.shape}, 出力次元: {y_wl.shape}")
        self.wl_model.fit(X_wl, y_wl)
        
        print(f"不確実性予測モデルを訓練中... 入力次元: {X_sigma.shape}, 出力次元: {y_sigma.shape}")
        self.sigma_model.fit(X_sigma, y_sigma)
        
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """予測を実行"""
        X_wl = self.wl_processor.transform(X)
        X_sigma = self.sigma_processor.transform(X)
        
        wl_pred = self.wl_model.predict(X_wl)
        sigma_pred = self.sigma_model.predict(X_sigma)
        
        # 結果を結合
        wl_cols = [f'wl_{i+1}' for i in range(wl_pred.shape[1])]
        sigma_cols = [f'sigma_{i+1}' for i in range(sigma_pred.shape[1])]
        
        result = pd.DataFrame(
            np.hstack([wl_pred, sigma_pred]),
            columns=wl_cols + sigma_cols,
            index=X.index
        )
        
        return result


class ModelFactory:
    """様々な回帰モデルを生成するファクトリ"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseEstimator:
        """指定されたタイプのモデルを作成"""
        
        if model_type == 'ridge':
            return MultiOutputRegressor(Ridge(**kwargs))
            
        elif model_type == 'lasso':
            return MultiOutputRegressor(Lasso(**kwargs))
            
        elif model_type == 'elastic_net':
            return MultiOutputRegressor(ElasticNet(**kwargs))
            
        elif model_type == 'random_forest':
            return RandomForestRegressor(**kwargs)
            
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**kwargs)
            
        elif model_type == 'lightgbm':
            if not HAS_LGB:
                raise ImportError("LightGBMがインストールされていません")
            return lgb.LGBMRegressor(**kwargs)
            
        elif model_type == 'xgboost':
            if not HAS_XGB:
                raise ImportError("XGBoostがインストールされていません")
            return xgb.XGBRegressor(**kwargs)
            
        else:
            raise ValueError(f"未知のモデルタイプ: {model_type}")


class ModelEvaluator:
    """モデル評価クラス"""
    
    @staticmethod
    def evaluate_model(model: BaseEstimator, 
                      X: pd.DataFrame, 
                      y: pd.DataFrame,
                      cv_folds: int = 5) -> Dict[str, float]:
        """クロスバリデーションでモデルを評価"""
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # MSEでの評価
        mse_scores = cross_val_score(
            model, X, y, 
            cv=kfold, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return {
            'mse_mean': -mse_scores.mean(),
            'mse_std': mse_scores.std(),
            'mse_scores': -mse_scores
        }
    
    @staticmethod
    def detailed_evaluation(y_true: pd.DataFrame, 
                          y_pred: pd.DataFrame) -> Dict[str, Any]:
        """詳細なモデル評価"""
        
        wl_cols = [col for col in y_true.columns if col.startswith('wl_')]
        sigma_cols = [col for col in y_true.columns if col.startswith('sigma_')]
        
        results = {}
        
        # 全体評価
        results['overall'] = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        # 波長部分の評価
        if wl_cols:
            results['wavelength'] = {
                'mse': mean_squared_error(y_true[wl_cols], y_pred[wl_cols]),
                'mae': mean_absolute_error(y_true[wl_cols], y_pred[wl_cols]),
                'rmse': np.sqrt(mean_squared_error(y_true[wl_cols], y_pred[wl_cols]))
            }
        
        # 不確実性部分の評価
        if sigma_cols:
            results['uncertainty'] = {
                'mse': mean_squared_error(y_true[sigma_cols], y_pred[sigma_cols]),
                'mae': mean_absolute_error(y_true[sigma_cols], y_pred[sigma_cols]),
                'rmse': np.sqrt(mean_squared_error(y_true[sigma_cols], y_pred[sigma_cols]))
            }
        
        return results


class EnsemblePredictor:
    """アンサンブル予測クラス"""
    
    def __init__(self, models: List[BaseEstimator], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """全てのモデルを訓練"""
        for i, model in enumerate(self.models):
            print(f"モデル {i+1}/{len(self.models)} を訓練中...")
            model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """重み付きアンサンブル予測"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            if isinstance(pred, np.ndarray):
                pred = pd.DataFrame(pred, index=X.index)
            predictions.append(pred)
        
        # 重み付き平均
        ensemble_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        
        return ensemble_pred


def create_baseline_models() -> Dict[str, BaseEstimator]:
    """ベースラインモデルセットを作成"""
    
    models = {
        'ridge': WavelengthSigmaPredictor(
            wl_model=ModelFactory.create_model('ridge', alpha=1.0),
            sigma_model=ModelFactory.create_model('ridge', alpha=1.0),
            processor_params={'n_components_pca': 50, 'use_pca': True}
        ),
        
        'random_forest': WavelengthSigmaPredictor(
            wl_model=ModelFactory.create_model('random_forest', 
                                             n_estimators=100, 
                                             max_depth=10,
                                             random_state=42),
            sigma_model=ModelFactory.create_model('random_forest', 
                                                n_estimators=100, 
                                                max_depth=10,
                                                random_state=42),
            processor_params={'n_components_pca': 50, 'use_pca': True}
        ),
        
        'gradient_boosting': WavelengthSigmaPredictor(
            wl_model=ModelFactory.create_model('gradient_boosting',
                                             n_estimators=100,
                                             max_depth=8,
                                             learning_rate=0.1,
                                             random_state=42),
            sigma_model=ModelFactory.create_model('gradient_boosting',
                                                n_estimators=100,
                                                max_depth=8,
                                                learning_rate=0.1,
                                                random_state=42),
            processor_params={'n_components_pca': 50, 'use_pca': True}
        )
    }
    
    return models


if __name__ == "__main__":
    # テスト用コード
    print("Ariel Data Challenge 2025 - モデル開発モジュール")
    print("利用可能なモデル:")
    
    models = create_baseline_models()
    for name in models.keys():
        print(f"  - {name}")
    
    print(f"\nSpectralDataProcessorの特徴:")
    print("  - 標準化（StandardScaler）")
    print("  - PCA次元削減（オプション）")
    print("  - 波長と不確実性の個別処理")
    
    print(f"\nWavelengthSigmaPredictorの特徴:")
    print("  - 波長と不確実性の同時予測")
    print("  - 個別の前処理パイプライン")
    print("  - 様々なベースモデル対応")