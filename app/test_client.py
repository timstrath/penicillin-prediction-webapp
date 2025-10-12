import requests
import json
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def test_single_prediction():
    """Test single spectrum prediction"""
    # Create dummy spectral data (2239 values)
    dummy_spectrum = [0.0] * 2239
    
    # Make request
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"spectral_values": dummy_spectrum}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Single prediction successful!")
        print(f"Predicted concentration: {result['predicted_concentration']} g/L")
        print(f"Model used: {result['model_used']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Processing time: {result['processing_time_ms']} ms")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_batch_prediction():
    """Test batch prediction"""
    # Create dummy batch data (3 spectra)
    dummy_batch = [[0.0] * 2239, [0.0] * 2239, [0.0] * 2239]
    
    # Make request
    response = requests.post(
        f"{BASE_URL}/batch_predict",
        json={"spectral_data": dummy_batch}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Batch prediction successful!")
        print(f"Predictions: {result['predictions']}")
        print(f"Total spectra: {result['total_spectra']}")
        print(f"Processing time: {result['processing_time_ms']} ms")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Health check successful!")
        print(f"Status: {result['status']}")
        print(f"Models loaded: {result['models_loaded']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model_info")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Model info retrieved!")
        print(f"Model type: {result['model_type']}")
        print(f"Training RMSE: {result['training_rmse']}")
        print(f"Input dimensions: {result['input_dimensions']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_with_real_data():
    """Test with real spectral data from your dataset"""
    try:
        import pandas as pd
        import numpy as np
        
        # Load a real spectrum from your data
        all_data = pd.read_csv('../Mendeley_data/100_Batches_IndPenSim_V3.csv', nrows=1)
        real_spectrum = all_data.iloc[0].tolist()
        
        # Clean the data for JSON serialization
        # Replace inf, -inf, and NaN with 0
        cleaned_spectrum = []
        for value in real_spectrum:
            if np.isnan(value) or np.isinf(value):
                cleaned_spectrum.append(0.0)
            else:
                cleaned_spectrum.append(float(value))
        
        print(f"📊 Loaded real spectrum with {len(cleaned_spectrum)} data points")
        print(f"🔧 Cleaned {sum(1 for v in real_spectrum if np.isnan(v) or np.isinf(v))} invalid values")
        
        # Make request
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"spectral_values": cleaned_spectrum}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Real data prediction successful!")
            print(f"Predicted concentration: {result['predicted_concentration']} g/L")
            print(f"Model used: {result['model_used']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Processing time: {result['processing_time_ms']} ms")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error loading real data: {str(e)}")

if __name__ == "__main__":
    print("Testing FastAPI endpoints...")
    print("\n1. Health Check:")
    test_health_check()
    
    print("\n2. Model Info:")
    test_model_info()
    
    print("\n3. Single Prediction (Dummy Data):")
    test_single_prediction()
    
    print("\n4. Batch Prediction:")
    test_batch_prediction()
    
    print("\n5. Real Data Prediction:")
    test_with_real_data()
