from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Penicillin Concentration Prediction API",
    description="API for predicting penicillin concentration from Raman spectroscopy data",
    version="1.0.0"
)

# Load models and pipeline
try:
    # Adjust path to models directory (go up one level from app/)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    loaded_pipeline = joblib.load(os.path.join(models_dir, "preprocessing_pipeline.pkl"))
    loaded_elastic_model = joblib.load(os.path.join(models_dir, "elasticnet_penicillin.pkl"))
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Model files not found: {str(e)}")
    print("Please ensure you have trained and saved your models first.")
    loaded_pipeline = None
    loaded_elastic_model = None
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    print("This might be due to dependency conflicts. Try updating packages:")
    print("pip install --upgrade numpy pandas scikit-learn")
    loaded_pipeline = None
    loaded_elastic_model = None

# Pydantic models for request/response
class SpectralData(BaseModel):
    """Input model for spectral data"""
    spectral_values: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "spectral_values": [0.0] * 2239  # Example with 2239 spectral points
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_concentration: float
    model_used: str
    confidence: str
    processing_time_ms: float

class BatchPredictionRequest(BaseModel):
    """Input model for batch predictions"""
    spectral_data: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {
                "spectral_data": [[0.0] * 2239, [0.0] * 2239]  # Example with 2 spectra
            }
        }

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[float]
    model_used: str
    total_spectra: int
    processing_time_ms: float

# Utility functions
def preprocess_spectral_data(spectral_values: List[float]) -> np.ndarray:
    """Preprocess spectral data for prediction"""
    # Convert to DataFrame
    df = pd.DataFrame([spectral_values])
    
    # Handle NaN values
    df = df.fillna(0)
    
    # Apply preprocessing pipeline
    processed_data = loaded_pipeline.transform(df)
    
    return processed_data

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Penicillin Concentration Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if loaded_pipeline is None or loaded_elastic_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "pipeline_ready": loaded_pipeline is not None,
        "elasticnet_ready": loaded_elastic_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_concentration(spectral_data: SpectralData):
    """Predict penicillin concentration for a single spectrum"""
    import time
    start_time = time.time()
    
    # Check if models are loaded
    if loaded_pipeline is None or loaded_elastic_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate input
    if len(spectral_data.spectral_values) != 2239:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 2239 spectral values, got {len(spectral_data.spectral_values)}"
        )
    
    try:
        # Preprocess data
        processed_data = preprocess_spectral_data(spectral_data.spectral_values)
        
        # Make prediction
        prediction = loaded_elastic_model.predict(processed_data)[0]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Determine confidence based on prediction value
        if 0 <= prediction <= 40:
            confidence = "high"
        elif 40 < prediction <= 60:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            predicted_concentration=round(prediction, 2),
            model_used="ElasticNet",
            confidence=confidence,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_concentration(batch_data: BatchPredictionRequest):
    """Predict penicillin concentration for multiple spectra"""
    import time
    start_time = time.time()
    
    # Check if models are loaded
    if loaded_pipeline is None or loaded_elastic_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate input
    if not batch_data.spectral_data:
        raise HTTPException(status_code=400, detail="No spectral data provided")
    
    for i, spectrum in enumerate(batch_data.spectral_data):
        if len(spectrum) != 2239:
            raise HTTPException(
                status_code=400, 
                detail=f"Spectrum {i} has {len(spectrum)} values, expected 2239"
            )
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(batch_data.spectral_data)
        
        # Handle NaN values
        df = df.fillna(0)
        
        # Apply preprocessing pipeline
        processed_data = loaded_pipeline.transform(df)
        
        # Make predictions
        predictions = loaded_elastic_model.predict(processed_data)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=[round(pred, 2) for pred in predictions],
            model_used="ElasticNet",
            total_spectra=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if loaded_elastic_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "ElasticNet",
        "model_parameters": {
            "alpha": getattr(loaded_elastic_model, 'alpha', 'N/A'),
            "l1_ratio": getattr(loaded_elastic_model, 'l1_ratio', 'N/A'),
            "max_iter": getattr(loaded_elastic_model, 'max_iter', 'N/A')
        },
        "preprocessing_steps": [
            "RangeCut (350-1750 cm⁻¹)",
            "Linear Correction",
            "Savitzky-Golay Filter",
            "Norris-Williams Derivative",
            "Standard Scaling"
        ],
        "training_rmse": 0.33,
        "input_dimensions": 2239,
        "output_dimensions": 1
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
