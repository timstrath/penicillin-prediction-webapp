#!/usr/bin/env python3
"""
Create a small test dataset for web app deployment
"""

import pandas as pd
import numpy as np
import os

def create_test_data():
    """Create a small test dataset for web app demonstration"""
    
    # Load original data to get structure
    data_path = 'app/Mendeley_data/100_Batches_IndPenSim_V3.csv'
    print(f"Loading original data structure from: {data_path}")
    
    # Read only first 100 samples for testing
    data_test = pd.read_csv(data_path, nrows=100)
    print(f"Test data shape: {data_test.shape}")
    
    # Create test data directory
    os.makedirs('test_data', exist_ok=True)
    
    # Save test data
    test_data_path = 'test_data/test_samples.csv'
    data_test.to_csv(test_data_path, index=False)
    
    print(f"✅ Test data saved to: {test_data_path}")
    print(f"Test data size: {os.path.getsize(test_data_path) / 1024:.2f} KB")
    
    # Show basic statistics
    target_col = 'Penicillin concentration(P:g/L)'
    if target_col in data_test.columns:
        y = data_test[target_col]
        print(f"\nTest data statistics:")
        print(f"Range: {y.min():.3f} - {y.max():.3f} g/L")
        print(f"Mean: {y.mean():.3f} g/L")
        print(f"Std: {y.std():.3f} g/L")
    
    return test_data_path

if __name__ == "__main__":
    create_test_data()
