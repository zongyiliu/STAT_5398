#just TEST whether XGBoost with GPUis faster than CPU

import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from multiprocessing import cpu_count

n_cpus = cpu_count() - 1
print(f"\n{'='*60}")
print(f"XGBoost Speed Test")
print(f"{'='*60}")
print(f"CPU Cores Available: {n_cpus}")
print(f"{'='*60}\n")

print("Generating test data...")
np.random.seed(42)
n_samples = 350
n_features = 6

X_train = np.random.randn(n_samples, n_features)
y_train = np.random.randn(n_samples)

X_test = np.random.randn(100, n_features)
y_test = np.random.randn(100)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}\n")

param_grid_gbm = {
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 250, 500, 1000]
}

n_models = 1
for key, val in param_grid_gbm.items():
    n_models *= len(val)

n_jobs_per_model = min(max(1, n_cpus // n_models), n_cpus)

print(f"GridSearch Configuration:")
print(f"  - Parameter combinations: {n_models}")
print(f"  - Cross-validation folds: 3")
print(f"  - Total training runs: {n_models * 3} = {n_models} params × 3 folds")
print(f"  - n_jobs per model: {n_jobs_per_model}")
print(f"  - GridSearchCV n_jobs: {n_cpus // n_jobs_per_model}")
print(f"\n{'='*60}\n")

# Test 1: GPU RTX 3080Ti
print("XGBoost with GPU")
print("-" * 60)

try:
    start_time = time.time()
    
    xgb_gpu = XGBRegressor(
        random_state=42, 
        n_jobs=n_jobs_per_model,
        device='cuda',
        tree_method='hist'
    )
    
    xgb_regressor_gpu = GridSearchCV(
        estimator=xgb_gpu, 
        param_grid=param_grid_gbm,
        cv=3, 
        n_jobs=n_cpus // n_jobs_per_model, 
        scoring='neg_mean_squared_error', 
        verbose=1
    )
    
    xgb_regressor_gpu.fit(X_train, y_train)
    
    gpu_time = time.time() - start_time
  
    pred_start = time.time()
    y_pred_gpu = xgb_regressor_gpu.best_estimator_.predict(X_test)
    pred_time_gpu = time.time() - pred_start

    print(f"   Best parameters: {xgb_regressor_gpu.best_params_}")
    print(f"   Best CV score: {-xgb_regressor_gpu.best_score_:.6f}")
    print(f"   Training time: {gpu_time:.2f} seconds")
    print(f"   Prediction time: {pred_time_gpu*1000:.2f} ms")
    
except Exception as e:
    print(f"GPU test failed: {str(e)}")
    gpu_time = None

print(f"\n{'='*60}\n")

# Test 2: CPU
print("Test 2: XGBoost with CPU")
print("-" * 60)

start_time = time.time()

xgb_cpu = XGBRegressor(
    random_state=42, 
    n_jobs=n_jobs_per_model
)

xgb_regressor_cpu = GridSearchCV(
    estimator=xgb_cpu, 
    param_grid=param_grid_gbm,
    cv=3, 
    n_jobs=n_cpus // n_jobs_per_model, 
    scoring='neg_mean_squared_error', 
    verbose=1
)

xgb_regressor_cpu.fit(X_train, y_train)

cpu_time = time.time() - start_time

pred_start = time.time()
y_pred_cpu = xgb_regressor_cpu.best_estimator_.predict(X_test)
pred_time_cpu = time.time() - pred_start

print(f"   Best parameters: {xgb_regressor_cpu.best_params_}")
print(f"   Best CV score: {-xgb_regressor_cpu.best_score_:.6f}")
print(f"   Training time: {cpu_time:.2f} seconds")
print(f"   Prediction time: {pred_time_cpu*1000:.2f} ms")

print(f"\n{'='*60}\n")

print("COMPARISON")
print("=" * 60)

if gpu_time is not None:
    speedup = cpu_time / gpu_time
    
    print(f"GPU:")
    print(f"   Training: {gpu_time:.2f}s")
    print(f"   Prediction: {pred_time_gpu*1000:.2f}ms")
    
    print(f"CPU:")
    print(f"   Training: {cpu_time:.2f}s")
    print(f"   Prediction: {pred_time_cpu*1000:.2f}ms")
    
    print(f"Speedup:")
    if speedup > 1:
        print(f" GPU is {speedup:.2f}x FASTER than CPU")
        print(f"   Time saved: {cpu_time - gpu_time:.2f}s ({(1-gpu_time/cpu_time)*100:.1f}%)")
    else:
        print(f" GPU is {1/speedup:.2f}x SLOWER than CPU")
        print(f"   Extra time: {gpu_time - cpu_time:.2f}s ({(gpu_time/cpu_time-1)*100:.1f}%)")
        print(f"\n   💡 Reason: Small dataset ({n_samples} samples)")
        print(f"      GPU overhead > computation benefit")
else:
    print(f"CPU:")
    print(f"   Training: {cpu_time:.2f}s")
    print(f"   Prediction: {pred_time_cpu*1000:.2f}ms")
    print(f"GPU test failed - only CPU results available")

print(f"\n{'='*60}")
