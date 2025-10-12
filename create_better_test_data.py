#!/usr/bin/env python3
"""
Create a better test dataset with realistic penicillin concentration values
"""

import pandas as pd
import numpy as np
import os

def create_better_test_data():
    """Create a test dataset with realistic penicillin concentration values"""
    
    # Load original data to get structure
    data_path = 'app/Mendeley_data/100_Batches_IndPenSim_V3.csv'
    print(f"Loading original data structure from: {data_path}")
    
    # Read a larger sample to find better data
    data_full = pd.read_csv(data_path, nrows=2000)
    
    # Find samples with meaningful penicillin concentrations
    target_col = 'Penicillin concentration(P:g/L)'
    if target_col in data_full.columns:
        # Filter for samples with concentration > 0.1 g/L
        meaningful_data = data_full[data_full[target_col] > 0.1]
        print(f"Found {len(meaningful_data)} samples with concentration > 0.1 g/L")
        
        if len(meaningful_data) >= 50:
            # Take first 50 meaningful samples
            test_data = meaningful_data.head(50)
        else:
            # Take all meaningful samples plus some random ones
            test_data = meaningful_data.copy()
            remaining_needed = 50 - len(meaningful_data)
            if remaining_needed > 0:
                # Add some random samples from the rest
                other_data = data_full[data_full[target_col] <= 0.1].sample(n=min(remaining_needed, len(data_full[data_full[target_col] <= 0.1])))
                test_data = pd.concat([test_data, other_data], ignore_index=True)
    else:
        # Fallback: take random samples
        test_data = data_full.sample(n=min(50, len(data_full)))
    
    print(f"Test data shape: {test_data.shape}")
    
    # Create test data directory
    os.makedirs('test_data', exist_ok=True)
    
    # Save test data
    test_data_path = 'test_data/test_samples.csv'
    test_data.to_csv(test_data_path, index=False)
    
    print(f"✅ Test data saved to: {test_data_path}")
    print(f"Test data size: {os.path.getsize(test_data_path) / 1024:.2f} KB")
    
    # Show basic statistics
    if target_col in test_data.columns:
        y = test_data[target_col]
        print(f"\nTest data statistics:")
        print(f"Range: {y.min():.3f} - {y.max():.3f} g/L")
        print(f"Mean: {y.mean():.3f} g/L")
        print(f"Std: {y.std():.3f} g/L")
        print(f"Median: {y.median():.3f} g/L")
        
        # Show distribution
        print(f"\nConcentration distribution:")
        print(f"Samples with concentration > 1.0 g/L: {len(y[y > 1.0])}")
        print(f"Samples with concentration > 5.0 g/L: {len(y[y > 5.0])}")
        print(f"Samples with concentration > 10.0 g/L: {len(y[y > 10.0])}")
    
    return test_data_path

if __name__ == "__main__":
    create_better_test_data()
