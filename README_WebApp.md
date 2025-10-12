# Penicillin Concentration Prediction Web App

A Streamlit-based web application for visualizing preprocessing steps and comparing PLS vs ElasticNet predictions for penicillin concentration from Raman spectroscopy data.

## Features

### 🔬 Preprocessing Tab
- **Raw Raman Spectra Visualization**: Interactive plots of original spectral data
- **Preprocessed Spectra Visualization**: Shows spectra after preprocessing pipeline
- **Preprocessing Steps**: Detailed information about each preprocessing step

### 📊 Results & Predictions Tab
- **Batch Prediction**: Run predictions on selected number of spectra
- **Model Comparison**: Side-by-side comparison of PLS vs ElasticNet predictions
- **Results Table**: Detailed results with statistics
- **Interactive Charts**: Scatter plots and distribution histograms
- **Real-time Progress**: Progress tracking during batch processing

### 📈 History Tab
- **Prediction History**: Track all previous prediction runs
- **Performance Over Time**: Visualize model performance trends
- **Export History**: Download prediction history

### ⚙️ Settings Tab
- **Model Configuration**: View current model parameters
- **Data Configuration**: Display current data settings
- **Display Settings**: Customize chart appearance
- **Export Options**: Download results as CSV
- **System Information**: Version and dependency info

## Installation

### Prerequisites
- Python 3.8+
- Conda environment with required packages

### Setup
1. **Activate your conda environment**:
   ```bash
   conda activate indpensim
   ```

2. **Install web app dependencies**:
   ```bash
   pip install -r requirements_webapp.txt
   ```

3. **Ensure required files exist**:
   - `models/preprocessing_pipeline.pkl`
   - `models/elasticnet_penicillin.pkl`
   - `Mendeley_data/100_Batches_IndPenSim_V3.csv`

## Running the Application

### Option 1: Using the run script (Recommended)
```bash
python run_webapp.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run web_app.py
```

### Option 3: With custom settings
```bash
streamlit run web_app.py --server.port 8501 --server.address localhost
```

## Usage

1. **Open your browser** and navigate to `http://localhost:8501`

2. **Preprocessing Tab**: 
   - View raw and preprocessed Raman spectra
   - Understand the preprocessing pipeline

3. **Results Tab**:
   - Select number of spectra to predict
   - Click "Run Predictions" to start batch processing
   - View results table and comparison charts

4. **History Tab**:
   - Review previous prediction runs
   - Track model performance over time

5. **Settings Tab**:
   - Configure display options
   - Export results
   - View system information

## Data Flow

```
Raw Data → Preprocessing → Model Prediction → Results Display
    ↓           ↓              ↓                ↓
CSV File → Pipeline → ElasticNet/PLS → Interactive Charts
```

## Key Features

- **No File Upload**: Uses your existing data file directly
- **Real-time Processing**: Live progress updates during predictions
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover
- **Model Comparison**: Side-by-side PLS vs ElasticNet evaluation
- **Session Persistence**: Maintains state across page interactions
- **Export Functionality**: Download results and history

## Technical Details

### Architecture
- **Frontend**: Streamlit (Python-based)
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and NumPy
- **ML Models**: Scikit-learn (ElasticNet, PLS)
- **Caching**: Streamlit's built-in caching for performance

### Performance
- **Data Loading**: Cached for fast subsequent loads
- **Preprocessing**: Cached pipeline application
- **Predictions**: Real-time batch processing
- **Visualizations**: Interactive and responsive

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check if model files exist in `models/` directory
   - Ensure models were trained and saved correctly

2. **Data not loading**:
   - Verify CSV file exists in `Mendeley_data/` directory
   - Check file permissions

3. **Dependencies missing**:
   - Run `pip install -r requirements_webapp.txt`
   - Ensure you're in the correct conda environment

4. **Port already in use**:
   - Change port: `streamlit run web_app.py --server.port 8502`
   - Or kill existing process

### Performance Tips

- Use smaller batch sizes for faster predictions
- Clear browser cache if experiencing slow loading
- Close other applications to free up memory

## File Structure

```
indpensim-notebook/
├── web_app.py                 # Main Streamlit application
├── run_webapp.py             # Run script
├── requirements_webapp.txt   # Web app dependencies
├── README_WebApp.md          # This file
├── models/                   # Trained models
│   ├── preprocessing_pipeline.pkl
│   └── elasticnet_penicillin.pkl
└── Mendeley_data/            # Data files
    └── 100_Batches_IndPenSim_V3.csv
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure data and model files are present
4. Check the Streamlit documentation for advanced configuration
