# FastAPI Application

This directory contains the FastAPI application for the Penicillin Concentration Prediction API.

## Structure

```
app/
├── __init__.py          # Package initialization
├── main.py              # Main FastAPI application
├── run_api.py           # Script to run API from app directory
├── test_client.py       # Test client for API endpoints
└── README.md            # This file
```

## Running the API

### From Root Directory (Recommended)
```bash
# From the project root directory
python run_api.py
```

### From App Directory
```bash
# From the app directory
cd app
python run_api.py
```

### Direct Uvicorn
```bash
# From the project root directory
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

### From Root Directory
```bash
python test_client.py
```

### From App Directory
```bash
cd app
python test_client.py
```

## API Endpoints

- **Root**: `/` - API information
- **Health**: `/health` - Health check
- **Predict**: `/predict` - Single spectrum prediction
- **Batch Predict**: `/batch_predict` - Multiple spectra prediction
- **Model Info**: `/model_info` - Model information
- **Documentation**: `/docs` - Interactive API documentation

## Configuration

The API automatically loads models from the `../models/` directory relative to the app folder.

## Dependencies

- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Pydantic
