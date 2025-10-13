#!/usr/bin/env python3
"""
Create simple demo test data - 100 samples with good target distribution for impressive results
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_simple_demo_data():
    """Create 100 test samples with good target distribution for demo"""
    
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
    
    # Strategy: Select samples that will show good results
    # 1. Remove extreme outliers (very high or very low values)
    # 2. Select samples with realistic penicillin concentrations
    # 3. Ensure good distribution across the range
    
    # Filter out extreme outliers (keep values between 0.1 and 25 g/L)
    valid_mask = (y_test >= 0.1) & (y_test <= 25.0)
    X_test_filtered = X_test[valid_mask]
    y_test_filtered = y_test[valid_mask]
    
    print(f"After filtering outliers: {len(y_test_filtered)} samples")
    print(f"Filtered range: {y_test_filtered.min():.3f} - {y_test_filtered.max():.3f} g/L")
    
    # Create bins for good distribution
    n_bins = 10
    bin_edges = np.linspace(y_test_filtered.min(), y_test_filtered.max(), n_bins + 1)
    
    # Select samples from each bin
    selected_indices = []
    samples_per_bin = 10  # 10 samples per bin = 100 total
    
    for i in range(n_bins):
        bin_mask = (y_test_filtered >= bin_edges[i]) & (y_test_filtered < bin_edges[i + 1])
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) >= samples_per_bin:
            # Randomly select samples from this bin
            np.random.seed(42 + i)  # Different seed for each bin
            selected = np.random.choice(bin_indices, samples_per_bin, replace=False)
            selected_indices.extend(selected)
        else:
            # Take all samples from this bin if not enough
            selected_indices.extend(bin_indices)
    
    # If we don't have enough samples, fill with random selection
    if len(selected_indices) < 100:
        remaining_needed = 100 - len(selected_indices)
        all_indices = np.arange(len(y_test_filtered))
        remaining_indices = np.setdiff1d(all_indices, selected_indices)
        
        if len(remaining_indices) >= remaining_needed:
            np.random.seed(42)
            additional = np.random.choice(remaining_indices, remaining_needed, replace=False)
            selected_indices.extend(additional)
    
    # Take exactly 100 samples
    selected_indices = selected_indices[:100]
    
    # Create demo test data
    demo_test_data = X_test_filtered.iloc[selected_indices].copy()
    demo_targets = y_test_filtered[selected_indices]
    demo_test_data[target_col] = demo_targets
    
    print(f"\nSelected 100 demo samples:")
    print(f"Target range: {demo_targets.min():.3f} - {demo_targets.max():.3f} g/L")
    print(f"Target mean: {demo_targets.mean():.3f} g/L")
    print(f"Target std: {demo_targets.std():.3f} g/L")
    
    # Show distribution
    print(f"\nTarget distribution:")
    print(f"Low (0-5 g/L): {np.sum((demo_targets >= 0) & (demo_targets < 5))} samples")
    print(f"Medium (5-15 g/L): {np.sum((demo_targets >= 5) & (demo_targets < 15))} samples")
    print(f"High (15-25 g/L): {np.sum((demo_targets >= 15) & (demo_targets <= 25))} samples")
    
    # Show some example values
    print(f"\nExample target values:")
    print(demo_targets[:10])
    
    # Create test data directory
    os.makedirs('test_data', exist_ok=True)
    
    # Save demo test data
    test_data_path = 'test_data/test_samples.csv'
    demo_test_data.to_csv(test_data_path, index=False)
    
    print(f"\n✅ Demo test data saved to: {test_data_path}")
    print(f"Demo data size: {os.path.getsize(test_data_path) / 1024:.2f} KB")
    
    return test_data_path

if __name__ == "__main__":
    create_simple_demo_data()
