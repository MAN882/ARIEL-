"""
Ariel Data Challenge 2025 - Simple Model Training Script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_sample_data(data_path):
    """Load and generate sample data"""
    print("Loading data...")
    
    sample_df = pd.read_csv(data_path / 'sample_submission.csv')
    
    # Generate more samples for training (simulate multiple observations)
    np.random.seed(42)
    n_samples = 100  # Generate 100 mock observations
    n_features = 50  # Reduced for simplicity
    
    print(f"Generating {n_samples} mock observations...")
    
    # Mock features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i+1}' for i in range(n_features)]
    )
    
    # Add some realistic patterns to simulate real calibration features
    for i in range(n_samples):
        # Add correlated features
        X.loc[i, 'feature_2'] = X.loc[i, 'feature_1'] * 0.5 + np.random.normal(0, 0.2)
        X.loc[i, 'feature_3'] = X.loc[i, 'feature_1'] * 0.3 + X.loc[i, 'feature_2'] * 0.2 + np.random.normal(0, 0.1)
    
    # Target values - create realistic spectral data
    target_cols = [col for col in sample_df.columns if col != 'planet_id']
    y = pd.DataFrame(index=range(n_samples), columns=target_cols)
    
    # Get base values from sample
    base_wl = sample_df[[col for col in target_cols if col.startswith('wl_')]].iloc[0]
    base_sigma = sample_df[[col for col in target_cols if col.startswith('sigma_')]].iloc[0]
    
    # Generate realistic spectral variations
    for i in range(n_samples):
        # Wavelength values with systematic variations
        wl_variation = np.random.normal(0, 0.02, len(base_wl))  # 2% variation
        spectral_trend = 0.01 * np.sin(np.linspace(0, 4*np.pi, len(base_wl)))  # Spectral features
        
        for j, col in enumerate(base_wl.index):
            y.loc[i, col] = base_wl.iloc[j] + wl_variation[j] + spectral_trend[j]
        
        # Uncertainty values with correlated variations
        sigma_variation = np.random.normal(0, 0.005, len(base_sigma))  # Smaller uncertainty variation
        
        for j, col in enumerate(base_sigma.index):
            y.loc[i, col] = base_sigma.iloc[j] + sigma_variation[j]
    
    print(f"Data size: X{X.shape}, y{y.shape}")
    return X, y


def train_simple_models(X, y, models_dir, results_dir):
    """Train simple baseline models"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # PCA for dimension reduction
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    print(f"After PCA: {X_train_pca.shape}")
    
    # Models to train
    models = {
        'ridge': MultiOutputRegressor(Ridge(alpha=1.0)),
        'random_forest': RandomForestRegressor(
            n_estimators=50, 
            max_depth=8, 
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            # Train
            model.fit(X_train_pca, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train_pca)
            y_pred_val = model.predict(X_val_pca)
            
            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            
            print(f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
            print(f"Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}")
            
            # Save results
            results[model_name] = {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae
            }
            
            # Save model with preprocessing
            model_data = {
                'model': model,
                'scaler': scaler,
                'pca': pca
            }
            
            with open(models_dir / f'{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved: {models_dir / f'{model_name}_model.pkl'}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Save results
    with open(results_dir / 'simple_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(results, results_dir)
    
    return results


def plot_results(results, results_dir):
    """Plot training results"""
    if not results:
        return
    
    model_names = list(results.keys())
    train_rmse = [results[name]['train_rmse'] for name in model_names]
    val_rmse = [results[name]['val_rmse'] for name in model_names]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = range(len(model_names))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], train_rmse, width, label='Train RMSE', alpha=0.8)
    ax.bar([i + width/2 for i in x], val_rmse, width, label='Validation RMSE', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'simple_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_submission(models_dir, data_path, results_dir):
    """Create sample submission"""
    print("\nCreating sample submission...")
    
    # Load sample submission format
    sample_df = pd.read_csv(data_path / 'sample_submission.csv')
    
    # Load best model (assume ridge for now)
    try:
        with open(models_dir / 'ridge_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        pca = model_data['pca']
        
        # Generate mock test features (same as training for demo)
        np.random.seed(123)  # Different seed for test
        n_samples = len(sample_df)
        n_features = 50
        
        X_test = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        
        # Preprocess
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Predict
        predictions = model.predict(X_test_pca)
        
        # Create submission
        submission = sample_df.copy()
        target_cols = [col for col in submission.columns if col != 'planet_id']
        submission[target_cols] = predictions
        
        # Save submission
        submission_path = results_dir / 'sample_submission_predicted.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"Submission saved: {submission_path}")
        print(f"Submission shape: {submission.shape}")
        
        return submission
        
    except Exception as e:
        print(f"Error creating submission: {e}")
        return None


def main():
    """Main execution function"""
    # Path setup
    BASE_DIR = Path("C:/Users/ichry/OneDrive/Desktop/kaggle_competition/ariel_data_challenge_2025")
    DATA_PATH = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_sample_data(DATA_PATH)
    
    # Train models
    results = train_simple_models(X, y, MODELS_DIR, RESULTS_DIR)
    
    # Create submission
    submission = create_submission(MODELS_DIR, DATA_PATH, RESULTS_DIR)
    
    # Print summary
    print("\n=== Training Summary ===")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"  Overfit ratio: {metrics['val_rmse']/metrics['train_rmse']:.2f}")
    
    print(f"\nModels saved in: {MODELS_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()