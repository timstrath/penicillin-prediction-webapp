#!/usr/bin/env python3
"""
Debug script to investigate the suspicious metrics discrepancy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def debug_predictions():
    """Debug the prediction metrics to understand the discrepancy"""
    
    # Load data
    print("Loading data...")
    data = pd.read_csv('test_data/test_samples.csv')
    target_col = 'Penicillin concentration(P:g/L)'
    ground_truth = data[target_col].values
    
    print(f"Data shape: {data.shape}")
    print(f"Ground truth range: {ground_truth.min():.3f} - {ground_truth.max():.3f}")
    print(f"Ground truth mean: {ground_truth.mean():.3f}")
    print(f"Ground truth std: {ground_truth.std():.3f}")
    
    # Load models
    print("\nLoading models...")
    try:
        pipeline = joblib.load('app/models/preprocessing_pipeline.pkl')
        elastic_model = joblib.load('app/models/elasticnet_penicillin.pkl')
        pls_model = joblib.load('app/models/pls_penicillin.pkl')
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    try:
        # Fill NaN values
        data_copy = data.copy()
        data_copy = data_copy.fillna(0)
        
        # Convert object columns to numeric
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce').fillna(0)
        
        # Reorder columns to match pipeline
        if hasattr(pipeline, 'feature_names_in_'):
            data_copy = data_copy.reindex(columns=pipeline.feature_names_in_, fill_value=0)
        
        preprocessed_data = pipeline.transform(data_copy)
        print(f"✅ Data preprocessed: {preprocessed_data.shape}")
    except Exception as e:
        print(f"❌ Error preprocessing: {e}")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        elasticnet_pred = elastic_model.predict(preprocessed_data)
        pls_pred = pls_model.predict(preprocessed_data)
        print(f"✅ Predictions made")
        print(f"ElasticNet predictions range: {elasticnet_pred.min():.3f} - {elasticnet_pred.max():.3f}")
        print(f"PLS predictions range: {pls_pred.min():.3f} - {pls_pred.max():.3f}")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # ElasticNet metrics
    elasticnet_mse = mean_squared_error(ground_truth, elasticnet_pred)
    elasticnet_mae = mean_absolute_error(ground_truth, elasticnet_pred)
    elasticnet_r2 = r2_score(ground_truth, elasticnet_pred)
    elasticnet_rmse = np.sqrt(elasticnet_mse)
    
    # PLS metrics
    pls_mse = mean_squared_error(ground_truth, pls_pred)
    pls_mae = mean_absolute_error(ground_truth, pls_pred)
    pls_r2 = r2_score(ground_truth, pls_pred)
    pls_rmse = np.sqrt(pls_mse)
    
    print(f"\n📊 METRICS COMPARISON:")
    print(f"{'Metric':<15} {'ElasticNet':<12} {'PLS':<12} {'Difference':<12}")
    print("-" * 55)
    print(f"{'RMSE':<15} {elasticnet_rmse:<12.3f} {pls_rmse:<12.3f} {abs(elasticnet_rmse-pls_rmse):<12.3f}")
    print(f"{'MAE':<15} {elasticnet_mae:<12.3f} {pls_mae:<12.3f} {abs(elasticnet_mae-pls_mae):<12.3f}")
    print(f"{'R²':<15} {elasticnet_r2:<12.3f} {pls_r2:<12.3f} {abs(elasticnet_r2-pls_r2):<12.3f}")
    print(f"{'MSE':<15} {elasticnet_mse:<12.3f} {pls_mse:<12.3f} {abs(elasticnet_mse-pls_mse):<12.3f}")
    
    # Check for suspicious patterns
    print(f"\n🔍 ANALYSIS:")
    
    # Check if predictions are too similar to ground truth
    elasticnet_corr = np.corrcoef(ground_truth, elasticnet_pred)[0,1]
    pls_corr = np.corrcoef(ground_truth, pls_pred)[0,1]
    print(f"Correlation with ground truth:")
    print(f"  ElasticNet: {elasticnet_corr:.6f}")
    print(f"  PLS: {pls_corr:.6f}")
    
    # Check if predictions are too close to each other
    pred_corr = np.corrcoef(elasticnet_pred, pls_pred)[0,1]
    print(f"Correlation between predictions: {pred_corr:.6f}")
    
    # Check prediction differences
    pred_diff = np.abs(elasticnet_pred - pls_pred)
    print(f"Mean absolute difference between predictions: {pred_diff.mean():.3f}")
    print(f"Max absolute difference between predictions: {pred_diff.max():.3f}")
    
    # Check if R² is suspiciously high
    if elasticnet_r2 > 0.99 and pls_r2 > 0.99:
        print(f"\n⚠️  WARNING: R² scores are suspiciously high (>0.99)")
        print(f"   This suggests possible data leakage or overfitting")
    
    # Check if the models are predicting very similar values
    if pred_corr > 0.95:
        print(f"\n⚠️  WARNING: Models are predicting very similar values")
        print(f"   This suggests they might be overfitted to this specific test set")
    
    # Show some example predictions
    print(f"\n📋 SAMPLE PREDICTIONS (first 10):")
    print(f"{'Index':<6} {'Ground Truth':<12} {'ElasticNet':<12} {'PLS':<12} {'ElasticNet Diff':<15} {'PLS Diff':<12}")
    print("-" * 80)
    for i in range(min(10, len(ground_truth))):
        elasticnet_diff = abs(ground_truth[i] - elasticnet_pred[i])
        pls_diff = abs(ground_truth[i] - pls_pred[i])
        print(f"{i:<6} {ground_truth[i]:<12.3f} {elasticnet_pred[i]:<12.3f} {pls_pred[i]:<12.3f} {elasticnet_diff:<15.3f} {pls_diff:<12.3f}")

if __name__ == "__main__":
    debug_predictions()
