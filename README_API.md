# Penicillin Concentration Prediction API

A FastAPI-based web service for predicting penicillin concentration from Raman spectroscopy data using machine learning models.

## Features

- **Single Spectrum Prediction**: Predict concentration for individual spectra
- **Batch Prediction**: Process multiple spectra simultaneously
- **Health Monitoring**: Health check endpoints for monitoring
- **Model Information**: Get details about the loaded model
- **Auto Documentation**: Interactive API documentation with Swagger UI

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure Models are Available**:
Make sure you have the following files in the `models/` directory:
- `preprocessing_pipeline.pkl`
- `elasticnet_penicillin.pkl`

## Running the API

### Development Mode
```bash
python app.py
```

### Production Mode
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: GET
- **Description**: API information and available endpoints

### 2. Health Check
- **URL**: `/health`
- **Method**: GET
- **Description**: Check if models are loaded and API is healthy

### 3. Single Prediction
- **URL**: `/predict`
- **Method**: POST
- **Description**: Predict penicillin concentration for a single spectrum
- **Input**: JSON with `spectral_values` (list of 2239 float values)
- **Output**: Predicted concentration, model info, confidence, processing time

### 4. Batch Prediction
- **URL**: `/batch_predict`
- **Method**: POST
- **Description**: Predict concentrations for multiple spectra
- **Input**: JSON with `spectral_data` (list of lists, each with 2239 float values)
- **Output**: List of predictions, model info, processing time

### 5. Model Information
- **URL**: `/model_info`
- **Method**: GET
- **Description**: Get information about the loaded model

## API Documentation

Once the API is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing the API

Run the test client to verify all endpoints:

```bash
python test_client.py
```

## Example Usage

### Python Requests
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"spectral_values": [0.0] * 2239}
)
result = response.json()
print(f"Predicted concentration: {result['predicted_concentration']} g/L")

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"spectral_data": [[0.0] * 2239, [0.0] * 2239]}
)
result = response.json()
print(f"Predictions: {result['predictions']}")
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"spectral_values": [0.0, 0.0, ...]}'
```

## Model Details

- **Model Type**: ElasticNet Regression
- **Training RMSE**: 0.33 g/L
- **Input Dimensions**: 2239 spectral points
- **Preprocessing**: RangeCut, Linear Correction, Savitzky-Golay Filter, Norris-Williams Derivative, Standard Scaling
- **Wavelength Range**: 350-1750 cm⁻¹

## Error Handling

The API includes comprehensive error handling for:
- Missing or invalid input data
- Model loading failures
- Preprocessing errors
- Prediction failures

## Performance

- **Single Prediction**: Typically < 100ms
- **Batch Processing**: Optimized for multiple spectra
- **Memory Efficient**: Handles large datasets

## Deployment

For production deployment, consider:
- Using a production ASGI server (e.g., Gunicorn with Uvicorn workers)
- Setting up proper logging
- Implementing authentication/authorization
- Adding rate limiting
- Using environment variables for configuration

## Support

For issues or questions, please check the API documentation at `/docs` or contact the development team.
