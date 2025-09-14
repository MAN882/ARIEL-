"""
Ariel Data Challenge 2025 - モデル訓練スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split
from models import create_baseline_models, ModelEvaluator, EnsemblePredictor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """モデル訓練管理クラス"""
    
    def __init__(self, data_path: Path, models_dir: Path, results_dir: Path):
        self.data_path = data_path
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # ディレクトリ作成
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        
    def load_data(self) -> tuple:
        """Load data (using sample data as substitute)"""
        print("Loading data...")
        
        # Sample data (in actual competition, features would be extracted from calibration data)
        sample_df = pd.read_csv(self.data_path / 'sample_submission.csv')
        
        # Generate mock feature data (in reality extracted from calibration data)
        np.random.seed(42)
        n_samples = len(sample_df)
        n_features = 100  # Number of mock features
        
        # Features (simulating features extracted from calibration data)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i+1}' for i in range(n_features)],
            index=sample_df['planet_id']
        )
        
        # Target (wavelength and uncertainty)
        y = sample_df.drop('planet_id', axis=1)
        
        print(f"Data size: X{X.shape}, y{y.shape}")
        return X, y
        
    def train_all_models(self, X: pd.DataFrame, y: pd.DataFrame):
        """全ベースラインモデルを訓練"""
        print("\\n=== ベースラインモデル訓練開始 ===")
        
        # 訓練・検証分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # ベースラインモデル作成
        baseline_models = create_baseline_models()
        
        for model_name, model in baseline_models.items():
            print(f"\\n--- {model_name} を訓練中 ---")
            
            try:
                # モデル訓練
                model.fit(X_train, y_train)
                
                # 予測
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                # 評価
                train_metrics = ModelEvaluator.detailed_evaluation(y_train, y_pred_train)
                val_metrics = ModelEvaluator.detailed_evaluation(y_val, y_pred_val)
                
                # 結果保存
                self.models[model_name] = model
                self.results[model_name] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'model_params': str(model)
                }
                
                print(f"訓練RMSE: {train_metrics['overall']['rmse']:.6f}")
                print(f"検証RMSE: {val_metrics['overall']['rmse']:.6f}")
                
                # モデル保存
                model_path = self.models_dir / f'{model_name}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"モデル保存: {model_path}")
                
            except Exception as e:
                print(f"エラー発生: {e}")
                continue
                
        print("\\n=== 全モデル訓練完了 ===")
        
    def create_ensemble(self, X: pd.DataFrame, y: pd.DataFrame):
        """アンサンブルモデルの作成"""
        print("\\n=== アンサンブルモデル作成 ===")
        
        if len(self.models) < 2:
            print("アンサンブルには最低2つのモデルが必要です")
            return
            
        # 検証セットでの性能に基づいて重みを決定
        weights = []
        model_list = []
        
        for model_name, model in self.models.items():
            val_rmse = self.results[model_name]['val_metrics']['overall']['rmse']
            # RMSEが低いほど重みを大きく（逆数を使用）
            weight = 1.0 / (val_rmse + 1e-6)
            weights.append(weight)
            model_list.append(model)
            print(f"{model_name}: 重み = {weight:.4f} (RMSE: {val_rmse:.6f})")
            
        # 重みの正規化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # アンサンブル作成
        ensemble = EnsemblePredictor(model_list, weights.tolist())
        
        # アンサンブル保存
        ensemble_path = self.models_dir / 'ensemble_model.pkl'
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble, f)
            
        self.models['ensemble'] = ensemble
        print(f"アンサンブルモデル保存: {ensemble_path}")
        
    def save_results(self):
        """結果を保存"""
        results_path = self.results_dir / 'training_results.json'
        
        # JSON用にデータを準備
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = {
                'train_rmse': result['train_metrics']['overall']['rmse'],
                'val_rmse': result['val_metrics']['overall']['rmse'],
                'train_mae': result['train_metrics']['overall']['mae'],
                'val_mae': result['val_metrics']['overall']['mae']
            }
            
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"結果保存: {results_path}")
        
    def plot_results(self):
        """結果の可視化"""
        if not self.results:
            print("表示する結果がありません")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = list(self.results.keys())
        train_rmse = [self.results[name]['train_metrics']['overall']['rmse'] for name in model_names]
        val_rmse = [self.results[name]['val_metrics']['overall']['rmse'] for name in model_names]
        
        x = range(len(model_names))
        
        # RMSE比較
        axes[0].bar([i-0.2 for i in x], train_rmse, width=0.4, label='Train RMSE', alpha=0.7)
        axes[0].bar([i+0.2 for i in x], val_rmse, width=0.4, label='Validation RMSE', alpha=0.7)
        axes[0].set_xlabel('モデル')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('モデル性能比較（RMSE）')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE比較
        train_mae = [self.results[name]['train_metrics']['overall']['mae'] for name in model_names]
        val_mae = [self.results[name]['val_metrics']['overall']['mae'] for name in model_names]
        
        axes[1].bar([i-0.2 for i in x], train_mae, width=0.4, label='Train MAE', alpha=0.7)
        axes[1].bar([i+0.2 for i in x], val_mae, width=0.4, label='Validation MAE', alpha=0.7)
        axes[1].set_xlabel('モデル')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('モデル性能比較（MAE）')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_summary(self):
        """結果サマリーを出力"""
        if not self.results:
            print("結果がありません")
            return
            
        print("\\n=== 訓練結果サマリー ===")
        print(f"{'モデル名':<15} {'Train RMSE':<12} {'Val RMSE':<12} {'オーバーフィット':<10}")
        print("-" * 55)
        
        for model_name, result in self.results.items():
            train_rmse = result['train_metrics']['overall']['rmse']
            val_rmse = result['val_metrics']['overall']['rmse']
            overfit = val_rmse / train_rmse
            
            print(f"{model_name:<15} {train_rmse:<12.6f} {val_rmse:<12.6f} {overfit:<10.2f}")
            
        # 最良モデルを特定
        best_model = min(self.results.keys(), 
                        key=lambda x: self.results[x]['val_metrics']['overall']['rmse'])
        best_rmse = self.results[best_model]['val_metrics']['overall']['rmse']
        
        print(f"\\n最良モデル: {best_model} (Validation RMSE: {best_rmse:.6f})")


def main():
    """メイン実行関数"""
    # パス設定
    BASE_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025")
    DATA_PATH = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # 結果ディレクトリ作成
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 訓練器初期化
    trainer = ModelTrainer(DATA_PATH, MODELS_DIR, RESULTS_DIR)
    
    # データ読み込み
    X, y = trainer.load_data()
    
    # 全モデル訓練
    trainer.train_all_models(X, y)
    
    # アンサンブル作成
    trainer.create_ensemble(X, y)
    
    # 結果保存・可視化
    trainer.save_results()
    trainer.plot_results()
    trainer.print_summary()
    
    print("\\n=== 訓練完了 ===")
    print(f"モデル保存先: {MODELS_DIR}")
    print(f"結果保存先: {RESULTS_DIR}")


if __name__ == "__main__":
    main()