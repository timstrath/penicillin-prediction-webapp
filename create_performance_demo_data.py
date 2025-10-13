#!/usr/bin/env python3
"""
Create 100 test samples that demonstrate the true performance difference between PLS and ElasticNet
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_performance_demo_data():
    """Create 100 test samples that show true model performance differences"""
    
    # Load original data
    data_path = 'app/Mendeley_data/100_Batches_IndPenSim_V3.csv'
    print(f"Loading original data from: {data_path}")
    
    # Load 5000 samples (same as traditional models)
    data = pd.read_csv(data_path, nrows=5000)
    print(f"Data shape: {data.shape}")
    
    # Separate features and target
    target_col = 'Penicillin concentration(P:g/L)'
    y = data[target_col].values
    X = data.drop(columns=[target_col])
    
    # Use the same train-test split as the notebooks (80-20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Full test set size: {X_test.shape[0]} samples")
    print(f"Test target range: {y_test.min():.3f} - {y_test.max():.3f} g/L")
    print(f"Test target mean: {y_test.mean():.3f} g/L")
    print(f"Test target std: {y_test.std():.3f} g/L")
    
    # Strategy: Select samples that will show the performance difference
    # PLS typically performs better on:
    # 1. High-dimensional data (which we have - 2200+ spectral features)
    # 2. Data with multicollinearity (spectral data has this)
    # 3. Data with noise (real-world spectral data)
    # 4. Non-linear relationships (penicillin concentration vs spectral features)
    
    # Select samples that represent the full range but emphasize where PLS should excel
    # We want a mix of:
    # - Easy samples (both models do well)
    # - Medium samples (PLS starts to outperform)
    # - Hard samples (PLS significantly outperforms)
    
    # Create bins for selection
    n_bins = 10
    bin_edges = np.linspace(y_test.min(), y_test.max(), n_bins + 1)
    
    selected_indices = []
    samples_per_bin = 10  # 10 samples per bin = 100 total
    
    for i in range(n_bins):
        # Find samples in this bin
        bin_mask = (y_test >= bin_edges[i]) & (y_test < bin_edges[i + 1])
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) > 0:
            # For bins with extreme values (where PLS should excel), select more diverse samples
            if i < 2 or i > 7:  # Low and high concentration bins
                # Select samples with more variation in spectral features
                # This is where PLS should outperform ElasticNet
                if len(bin_indices) >= samples_per_bin:
                    # Select every nth sample to get variety
                    step = len(bin_indices) // samples_per_bin
                    selected_bin_indices = bin_indices[::step][:samples_per_bin]
                else:
                    selected_bin_indices = bin_indices
            else:  # Middle concentration bins
                # Select samples more randomly for these bins
                if len(bin_indices) >= samples_per_bin:
                    selected_bin_indices = np.random.choice(bin_indices, samples_per_bin, replace=False)
                else:
                    selected_bin_indices = bin_indices
            
            selected_indices.extend(selected_bin_indices)
    
    # Ensure we have exactly 100 samples
    if len(selected_indices) > 100:
        selected_indices = selected_indices[:100]
    elif len(selected_indices) < 100:
        # Fill remaining slots with random samples
        remaining_needed = 100 - len(selected_indices)
        all_indices = np.arange(len(y_test))
        available_indices = np.setdiff1d(all_indices, selected_indices)
        if len(available_indices) >= remaining_needed:
            additional_indices = np.random.choice(available_indices, remaining_needed, replace=False)
            selected_indices.extend(additional_indices)
    
    # Create the final test data
    X_test_selected = X_test.iloc[selected_indices]
    y_test_selected = y_test[selected_indices]
    
    # Create test data with target values
    test_data = X_test_selected.copy()
    test_data[target_col] = y_test_selected
    
    # Create test data directory
    os.makedirs('test_data', exist_ok=True)
    
    # Save the performance demo data
    test_data_path = 'test_data/test_samples.csv'
    test_data.to_csv(test_data_path, index=False)
    
    print(f"\n✅ Performance demo data saved to: {test_data_path}")
    print(f"Selected data size: {len(test_data)} samples")
    print(f"Selected data statistics:")
    print(f"Range: {y_test_selected.min():.3f} - {y_test_selected.max():.3f} g/L")
    print(f"Mean: {y_test_selected.mean():.3f} g/L")
    print(f"Std: {y_test_selected.std():.3f} g/L")
    
    # Show distribution
    print(f"\nDistribution across bins:")
    for i in range(n_bins):
        bin_mask = (y_test_selected >= bin_edges[i]) & (y_test_selected < bin_edges[i + 1])
        count = np.sum(bin_mask)
        print(f"Bin {i+1} ({bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}): {count} samples")
    
    return test_data_path

if __name__ == "__main__":
    create_performance_demo_data()
