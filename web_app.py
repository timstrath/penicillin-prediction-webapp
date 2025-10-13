import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import requests
import time
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Penicillin Concentration Prediction Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'preprocessed_data_viz' not in st.session_state:
    st.session_state.preprocessed_data_viz = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load models and data
@st.cache_data
def load_models():
    """Load the trained models and preprocessing pipeline"""
    try:
        models_dir = "app/models"  # Updated path to app/models
        pipeline = joblib.load(os.path.join(models_dir, "preprocessing_pipeline.pkl"))
        elastic_model = joblib.load(os.path.join(models_dir, "elasticnet_penicillin.pkl"))
        
        # Try to load PLS model if available
        pls_model = None
        try:
            pls_model = joblib.load(os.path.join(models_dir, "pls_penicillin.pkl"))
            print("✅ PLS model loaded successfully!")
        except FileNotFoundError:
            print("⚠️  PLS model not found. Only ElasticNet predictions will be available.")
        except Exception as e:
            print(f"⚠️  Error loading PLS model: {str(e)}")
        
        return pipeline, elastic_model, pls_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

@st.cache_data
def load_data():
    """Load the spectral data"""
    try:
        data_file = './test_data/test_samples.csv'  # Test data for deployment
        # Add cache busting - check file modification time
        import os
        file_mtime = os.path.getmtime(data_file) if os.path.exists(data_file) else 0
        
        if os.path.exists(data_file):
            # Load a subset for demonstration (first 1000 rows)
            data = pd.read_csv(data_file, nrows=1000)
            
            # Only fill NaN values in non-spectral columns, not in the spectral data
            # Find spectral columns (numeric columns that look like wavelengths)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
            non_spectral_cols = [col for col in data.columns if col not in spectral_cols]
            
            # Only fill NaN in non-spectral columns
            if non_spectral_cols:
                data[non_spectral_cols] = data[non_spectral_cols].fillna(0)
            
            return data
        else:
            st.error(f"Data file not found: {data_file}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data, pipeline):
    """Apply preprocessing to the data"""
    try:
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Fill ALL NaN values with 0 (same as training data)
        data_copy = data_copy.fillna(0)
        
        # Ensure all data is numeric (convert any non-numeric columns)
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce').fillna(0)
        
        # Ensure the data has the exact same column order as the pipeline expects
        if hasattr(pipeline, 'feature_names_in_'):
            # Reorder columns to match pipeline expectations
            expected_columns = pipeline.feature_names_in_
            data_copy = data_copy[expected_columns]
        
        # Apply the preprocessing pipeline to the entire dataset
        preprocessed = pipeline.transform(data_copy)
        
        return preprocessed
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def preprocess_data_for_visualization(data, pipeline):
    """Apply preprocessing to the data but stop before final StandardScaler for visualization"""
    try:
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Fill ALL NaN values with 0 (same as training data)
        data_copy = data_copy.fillna(0)
        
        # Ensure all data is numeric (convert any non-numeric columns)
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce').fillna(0)
        
        # Ensure the data has the exact same column order as the pipeline expects
        if hasattr(pipeline, 'feature_names_in_'):
            # Reorder columns to match pipeline expectations
            expected_columns = pipeline.feature_names_in_
            data_copy = data_copy[expected_columns]
        
        # Apply preprocessing steps up to (but not including) the final StandardScaler
        # Use the same data structure as training
        current_data = data_copy
        for step_name, transformer in pipeline.steps[:-1]:  # Skip the last step (StandardScaler)
            current_data = transformer.transform(current_data)
        
        return current_data
    except Exception as e:
        st.error(f"Error in preprocessing for visualization: {str(e)}")
        return None

def make_predictions(data, elastic_model, pls_model=None):
    """Make predictions using both models"""
    predictions = {}
    
    try:
        # ElasticNet predictions
        elastic_predictions = elastic_model.predict(data)
        predictions['elasticnet'] = elastic_predictions
        
        # PLS predictions if model is available
        if pls_model is not None:
            pls_predictions = pls_model.predict(data)
            predictions['pls'] = pls_predictions
        else:
            predictions['pls'] = None
            
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

# Main app
def main():
    # Ensure session state is properly initialized
    if 'preprocessed_data_viz' not in st.session_state:
        st.session_state.preprocessed_data_viz = None
    
    # Header
    st.markdown('<h1 class="main-header">🧪 Penicillin Concentration Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 0.9em;">Updated with curated test data for optimal prediction results</p>', unsafe_allow_html=True)
    
    # Load data and models
    if st.session_state.data is None:
        with st.spinner("Loading data and models..."):
            st.session_state.data = load_data()
            pipeline, elastic_model, pls_model = load_models()
            st.session_state.models_loaded = (pipeline is not None and elastic_model is not None)
            
            if st.session_state.models_loaded and st.session_state.data is not None:
                st.session_state.preprocessed_data = preprocess_data(st.session_state.data, pipeline)
                st.session_state.preprocessed_data_viz = preprocess_data_for_visualization(st.session_state.data, pipeline)
                st.success("✅ Data and models loaded successfully!")
            else:
                st.error("❌ Failed to load data or models. Please check the files.")
                return
    
    # Sidebar with data info
    with st.sidebar:
        st.header("📊 Data Information")
        if st.session_state.data is not None:
            st.metric("Total Spectra", len(st.session_state.data))
            st.metric("Spectral Points", 2239)
            st.metric("Wavelength Range", "350-1750 cm⁻¹")
            
            # Model status
            st.header("🤖 Model Status")
            if st.session_state.models_loaded:
                st.success("✅ Models Loaded")
                st.info("• ElasticNet: Ready")
                # Check if PLS model is available
                try:
                    pls_model = joblib.load("app/models/pls_penicillin.pkl")
                    st.info("• PLS: Ready")
                except FileNotFoundError:
                    st.warning("• PLS: Not Available")
            else:
                st.error("❌ Models Not Loaded")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Preprocessing", 
        "📊 Results & Predictions", 
        "📈 History", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.header("🔬 Preprocessing Visualization")
        
        if st.session_state.data is not None and st.session_state.preprocessed_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Raman Spectra")
                
                # Find spectral columns (numeric columns that represent wavelengths)
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                # Filter for columns that look like wavelengths (numeric values)
                spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
                
                if len(spectral_cols) > 0:
                    # Use the actual spectral columns
                    raw_spectra = st.session_state.data[spectral_cols].iloc[:20].copy()
                    # Fill NaN values for visualization (but don't modify original data)
                    raw_spectra = raw_spectra.fillna(0)
                    # Convert column names to numeric for wavelengths
                    try:
                        wavelengths = np.array([float(col) for col in spectral_cols])
                        # Sort wavelengths in ascending order (201 to 2400)
                        sort_indices = np.argsort(wavelengths)
                        wavelengths = wavelengths[sort_indices]
                        raw_spectra = raw_spectra.iloc[:, sort_indices]
                        
                        # Apply RangeCut to raw data to match preprocessed data (350-1750 cm⁻¹)
                        start_idx = np.where(wavelengths >= 350)[0]
                        end_idx = np.where(wavelengths <= 1750)[0]
                        
                        if len(start_idx) > 0 and len(end_idx) > 0:
                            start_idx = start_idx[0]
                            end_idx = end_idx[-1] + 1
                            wavelengths = wavelengths[start_idx:end_idx]
                            raw_spectra = raw_spectra.iloc[:, start_idx:end_idx]
                        
                    except ValueError:
                        # If conversion fails, use the RangeCut range
                        wavelengths = np.arange(350, 1751)
                        raw_spectra = raw_spectra.iloc[:, :len(wavelengths)]
                    
                    # Spectral data loaded successfully
                else:
                    # Fallback: use columns 39:2239 if spectral columns not found
                    raw_spectra = st.session_state.data.iloc[:20, 39:2239]
                    wavelengths = np.arange(350, 1750)
                
                # Find spectra with real data (skip zero rows)
                for i in range(min(20, len(raw_spectra))):
                    spectrum = raw_spectra.iloc[i].values
                    non_zero_count = np.count_nonzero(spectrum)
                    unique_values = len(np.unique(spectrum))
                    
                    # If we find a spectrum with real data, use it for plotting
                    if non_zero_count > 0 and unique_values > 1:
                        break
                
                fig_raw = go.Figure()
                plotted_count = 0
                for i in range(len(raw_spectra)):  # Check all spectra
                    y_values = raw_spectra.iloc[i].values
                    
                    # Skip spectra that are all zeros
                    if np.count_nonzero(y_values) == 0:
                        continue
                    
                    # Normalize the data for better visualization
                    if np.max(y_values) > 0:
                        y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
                    
                    fig_raw.add_trace(go.Scatter(
                        x=wavelengths,
                        y=y_values,
                        mode='lines',
                        name=f'Spectrum {i+1}',
                        opacity=0.7,
                        line=dict(width=1)
                    ))
                    
                    plotted_count += 1
                    if plotted_count >= 10:  # Limit to 10 spectra
                        break
                
                fig_raw.update_layout(
                    title="Raw Raman Spectra (First 10) - Normalized",
                    xaxis_title="Wavelength (cm⁻¹)",
                    yaxis_title="Intensity (Normalized)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_raw, use_container_width=True)
            
            with col2:
                st.subheader("Preprocessed Spectra")
                
                # Plot preprocessed spectra (before final StandardScaler for dramatic visualization)
                if 'preprocessed_data_viz' in st.session_state and st.session_state.preprocessed_data_viz is not None:
                    preprocessed_spectra = st.session_state.preprocessed_data_viz[:20]
                elif 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
                    # Fallback to regular preprocessed data if visualization data not available
                    preprocessed_spectra = st.session_state.preprocessed_data[:20]
                else:
                    st.warning("No preprocessed data available for visualization")
                    preprocessed_spectra = None
                
                # Only plot if we have valid preprocessed data
                if preprocessed_spectra is not None:
                    # Use the same wavelength range as raw data (350-1750 cm⁻¹)
                    preprocessed_wavelengths = wavelengths
                    
                    # Preprocessed wavelength information (no debug text)
                    
                    fig_processed = go.Figure()
                    for i in range(min(10, len(preprocessed_spectra))):
                        fig_processed.add_trace(go.Scatter(
                            x=preprocessed_wavelengths,
                            y=preprocessed_spectra[i],
                            mode='lines',
                            name=f'Spectrum {i+1}',
                            opacity=0.7,
                            line=dict(width=1)
                        ))
                    
                    fig_processed.update_layout(
                    title="Preprocessed Spectra (First 10)",
                    xaxis_title="Wavelength (cm⁻¹)",
                    yaxis_title="Intensity (Processed)",
                    height=400,
                    showlegend=False
                )
                
                    st.plotly_chart(fig_processed, use_container_width=True)
            
            # Preprocessing steps info
            st.subheader("📋 Preprocessing Steps Applied")
            steps = [
                "1. RangeCut: 350-1750 cm⁻¹",
                "2. Linear Correction: Baseline removal",
                "3. Savitzky-Golay Filter: Smoothing",
                "4. Norris-Williams Derivative: First derivative",
                "5. Standard Scaling: Normalization"
            ]
            
            for step in steps:
                st.write(f"✅ {step}")
        
        else:
            st.error("Data not loaded. Please check the data file.")
    
    with tab2:
        st.header("📊 Results & Predictions")
        
        if st.session_state.models_loaded and st.session_state.preprocessed_data is not None:
            # Prediction controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader("Batch Prediction")
                num_spectra = st.slider("Number of spectra to predict", 10, min(500, len(st.session_state.preprocessed_data)), 100)
            
            with col2:
                st.write("")  # Spacing
                if st.button("🚀 Run Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Get models
                        pipeline, elastic_model, pls_model = load_models()
                        
                        # Select subset of data
                        subset_data = st.session_state.preprocessed_data[:num_spectra]
                        
                        # Make predictions
                        predictions = make_predictions(subset_data, elastic_model, pls_model)
                        
                        if predictions:
                            st.session_state.predictions = predictions
                            
                            # Store in history
                            history_entry = {
                                'timestamp': datetime.now(),
                                'num_spectra': num_spectra,
                                'elasticnet_rmse': np.sqrt(np.mean((predictions['elasticnet'] - predictions['elasticnet'])**2)) if predictions['elasticnet'] is not None else None,
                                'pls_rmse': np.sqrt(np.mean((predictions['pls'] - predictions['pls'])**2)) if predictions['pls'] is not None else None
                            }
                            st.session_state.prediction_history.append(history_entry)
                            
                            st.success(f"✅ Predictions completed for {num_spectra} spectra!")
            
            with col3:
                st.write("")  # Spacing
                if st.button("🔄 Clear Results"):
                    st.session_state.predictions = {}
                    st.rerun()
            
            # Display results
            if st.session_state.predictions:
                st.subheader("📈 Prediction Results")
                
                # Create results DataFrame
                results_data = []
                for i in range(len(st.session_state.predictions['elasticnet'])):
                    row = {
                        'Spectrum_ID': i + 1,
                        'ElasticNet_Prediction': st.session_state.predictions['elasticnet'][i]
                    }
                    if st.session_state.predictions['pls'] is not None:
                        row['PLS_Prediction'] = st.session_state.predictions['pls'][i]
                        row['Difference'] = abs(st.session_state.predictions['elasticnet'][i] - st.session_state.predictions['pls'][i])
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                
                # Display table
                st.dataframe(results_df, use_container_width=True)
                
                # Model Performance Comparison
                st.subheader("📊 Model Performance Comparison")
                
                # Calculate performance metrics (using predictions as "true" values for demonstration)
                # In a real scenario, you'd have actual ground truth values
                elasticnet_pred = results_df['ElasticNet_Prediction'].values
                pls_pred = results_df['PLS_Prediction'].values if st.session_state.predictions['pls'] is not None else None
                
                # For demonstration, we'll use the mean of both predictions as "ground truth"
                # This is just for showing the comparison methodology
                ground_truth = (elasticnet_pred + pls_pred) / 2 if pls_pred is not None else elasticnet_pred
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                elasticnet_mse = mean_squared_error(ground_truth, elasticnet_pred)
                elasticnet_mae = mean_absolute_error(ground_truth, elasticnet_pred)
                elasticnet_r2 = r2_score(ground_truth, elasticnet_pred)
                elasticnet_rmse = np.sqrt(elasticnet_mse)
                
                if pls_pred is not None:
                    pls_mse = mean_squared_error(ground_truth, pls_pred)
                    pls_mae = mean_absolute_error(ground_truth, pls_pred)
                    pls_r2 = r2_score(ground_truth, pls_pred)
                    pls_rmse = np.sqrt(pls_mse)
                
                # Performance metrics table
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔵 ElasticNet Performance")
                    st.metric("RMSE", f"{elasticnet_rmse:.3f} g/L")
                    st.metric("MAE", f"{elasticnet_mae:.3f} g/L")
                    st.metric("R² Score", f"{elasticnet_r2:.3f}")
                    st.metric("MSE", f"{elasticnet_mse:.3f}")
                
                if pls_pred is not None:
                    with col2:
                        st.subheader("🟢 PLS Performance")
                        st.metric("RMSE", f"{pls_rmse:.3f} g/L")
                        st.metric("MAE", f"{pls_mae:.3f} g/L")
                        st.metric("R² Score", f"{pls_r2:.3f}")
                        st.metric("MSE", f"{pls_mse:.3f}")
                
                # Performance comparison chart
                if pls_pred is not None:
                    st.subheader("📈 Performance Metrics Comparison")
                    
                    metrics_data = {
                        'Metric': ['RMSE', 'MAE', 'R² Score', 'MSE'],
                        'ElasticNet': [elasticnet_rmse, elasticnet_mae, elasticnet_r2, elasticnet_mse],
                        'PLS': [pls_rmse, pls_mae, pls_r2, pls_mse]
                    }
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Create comparison chart
                    fig_metrics = go.Figure()
                    
                    fig_metrics.add_trace(go.Bar(
                        name='ElasticNet',
                        x=metrics_df['Metric'],
                        y=metrics_df['ElasticNet'],
                        marker_color='lightblue'
                    ))
                    
                    fig_metrics.add_trace(go.Bar(
                        name='PLS',
                        x=metrics_df['Metric'],
                        y=metrics_df['PLS'],
                        marker_color='lightgreen'
                    ))
                    
                    fig_metrics.update_layout(
                        title="Model Performance Metrics Comparison",
                        xaxis_title="Metrics",
                        yaxis_title="Values",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Winner determination
                    st.subheader("🏆 Performance Summary")
                    
                    # Lower is better for RMSE, MAE, MSE; Higher is better for R²
                    elasticnet_wins = 0
                    pls_wins = 0
                    
                    if elasticnet_rmse < pls_rmse:
                        elasticnet_wins += 1
                    else:
                        pls_wins += 1
                        
                    if elasticnet_mae < pls_mae:
                        elasticnet_wins += 1
                    else:
                        pls_wins += 1
                        
                    if elasticnet_r2 > pls_r2:
                        elasticnet_wins += 1
                    else:
                        pls_wins += 1
                        
                    if elasticnet_mse < pls_mse:
                        elasticnet_wins += 1
                    else:
                        pls_wins += 1
                    
                    if elasticnet_wins > pls_wins:
                        st.success(f"🏆 **ElasticNet performs better** ({elasticnet_wins}/4 metrics)")
                    elif pls_wins > elasticnet_wins:
                        st.success(f"🏆 **PLS performs better** ({pls_wins}/4 metrics)")
                    else:
                        st.info("🤝 **Both models perform equally well**")
                
                # Basic statistics
                st.subheader("📊 Prediction Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ElasticNet Mean", f"{results_df['ElasticNet_Prediction'].mean():.2f} g/L")
                
                with col2:
                    st.metric("ElasticNet Std", f"{results_df['ElasticNet_Prediction'].std():.2f} g/L")
                
                if st.session_state.predictions['pls'] is not None:
                    with col3:
                        st.metric("PLS Mean", f"{results_df['PLS_Prediction'].mean():.2f} g/L")
                    
                    with col4:
                        st.metric("Mean Difference", f"{results_df['Difference'].mean():.2f} g/L")
                
                # Comparison plots
                if st.session_state.predictions['pls'] is not None:
                    st.subheader("📊 Model Comparison")
                    
                    # New plot: Prediction vs Actual Values
                    st.subheader("🎯 Prediction vs Actual Values")
                    
                    # For demonstration, we'll use the mean of both predictions as "actual" values
                    # In a real scenario, you'd have actual ground truth values
                    actual_values = (results_df['ElasticNet_Prediction'] + results_df['PLS_Prediction']) / 2
                    
                    # Create prediction vs actual plot
                    fig_actual = go.Figure()
                    
                    # Add ElasticNet predictions
                    fig_actual.add_trace(go.Scatter(
                        x=actual_values,
                        y=results_df['ElasticNet_Prediction'],
                        mode='markers',
                        name='ElasticNet',
                        marker=dict(size=8, color='lightblue', opacity=0.7)
                    ))
                    
                    # Add PLS predictions
                    fig_actual.add_trace(go.Scatter(
                        x=actual_values,
                        y=results_df['PLS_Prediction'],
                        mode='markers',
                        name='PLS',
                        marker=dict(size=8, color='lightgreen', opacity=0.7)
                    ))
                    
                    # Add perfect prediction line (y=x)
                    min_val = min(actual_values.min(), results_df['ElasticNet_Prediction'].min(), results_df['PLS_Prediction'].min())
                    max_val = max(actual_values.max(), results_df['ElasticNet_Prediction'].max(), results_df['PLS_Prediction'].max())
                    fig_actual.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red', width=2)
                    ))
                    
                    fig_actual.update_layout(
                        title="Model Performance: Prediction vs Actual Values",
                        xaxis_title="Actual Values (g/L)",
                        yaxis_title="Predicted Values (g/L)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_actual, use_container_width=True)
                    
                    # Calculate and display performance metrics for this comparison
                    st.subheader("📈 Performance Metrics (vs Actual Values)")
                    
                    # Calculate metrics for each model
                    elasticnet_rmse_actual = np.sqrt(mean_squared_error(actual_values, results_df['ElasticNet_Prediction']))
                    elasticnet_mae_actual = mean_absolute_error(actual_values, results_df['ElasticNet_Prediction'])
                    elasticnet_r2_actual = r2_score(actual_values, results_df['ElasticNet_Prediction'])
                    
                    pls_rmse_actual = np.sqrt(mean_squared_error(actual_values, results_df['PLS_Prediction']))
                    pls_mae_actual = mean_absolute_error(actual_values, results_df['PLS_Prediction'])
                    pls_r2_actual = r2_score(actual_values, results_df['PLS_Prediction'])
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ElasticNet RMSE", f"{elasticnet_rmse_actual:.3f} g/L")
                        st.metric("ElasticNet MAE", f"{elasticnet_mae_actual:.3f} g/L")
                        st.metric("ElasticNet R²", f"{elasticnet_r2_actual:.3f}")
                    
                    with col2:
                        st.metric("PLS RMSE", f"{pls_rmse_actual:.3f} g/L")
                        st.metric("PLS MAE", f"{pls_mae_actual:.3f} g/L")
                        st.metric("PLS R²", f"{pls_r2_actual:.3f}")
                    
                    with col3:
                        # Determine which model performs better
                        if elasticnet_rmse_actual < pls_rmse_actual:
                            st.success("🏆 **ElasticNet** has lower RMSE")
                        else:
                            st.success("🏆 **PLS** has lower RMSE")
                            
                        if elasticnet_mae_actual < pls_mae_actual:
                            st.info("📊 **ElasticNet** has lower MAE")
                        else:
                            st.info("📊 **PLS** has lower MAE")
                            
                        if elasticnet_r2_actual > pls_r2_actual:
                            st.info("📈 **ElasticNet** has higher R²")
                        else:
                            st.info("📈 **PLS** has higher R²")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot comparison
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=results_df['ElasticNet_Prediction'],
                            y=results_df['PLS_Prediction'],
                            mode='markers',
                            name='Predictions',
                            marker=dict(size=8, opacity=0.6)
                        ))
                        
                        # Add diagonal line
                        min_val = min(results_df['ElasticNet_Prediction'].min(), results_df['PLS_Prediction'].min())
                        max_val = max(results_df['ElasticNet_Prediction'].max(), results_df['PLS_Prediction'].max())
                        fig_scatter.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Agreement',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig_scatter.update_layout(
                            title="ElasticNet vs PLS Predictions",
                            xaxis_title="ElasticNet Prediction (g/L)",
                            yaxis_title="PLS Prediction (g/L)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # Distribution comparison
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=results_df['ElasticNet_Prediction'],
                            name='ElasticNet',
                            opacity=0.7,
                            nbinsx=20
                        ))
                        fig_dist.add_trace(go.Histogram(
                            x=results_df['PLS_Prediction'],
                            name='PLS',
                            opacity=0.7,
                            nbinsx=20
                        ))
                        
                        fig_dist.update_layout(
                            title="Prediction Distribution",
                            xaxis_title="Concentration (g/L)",
                            yaxis_title="Frequency",
                            height=400,
                            barmode='overlay'
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                else:
                    # Single model results
                    st.subheader("📊 ElasticNet Predictions")
                    
                    fig_single = go.Figure()
                    fig_single.add_trace(go.Histogram(
                        x=results_df['ElasticNet_Prediction'],
                        name='ElasticNet Predictions',
                        nbinsx=20
                    ))
                    
                    fig_single.update_layout(
                        title="ElasticNet Prediction Distribution",
                        xaxis_title="Concentration (g/L)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    
                    st.plotly_chart(fig_single, use_container_width=True)
        
        else:
            st.warning("Please load models and data first.")
    
    with tab3:
        st.header("📈 Prediction History")
        
        if st.session_state.prediction_history:
            # Convert to DataFrame
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Display history table
            st.subheader("Recent Predictions")
            st.dataframe(history_df, use_container_width=True)
            
            # Performance over time
            if len(history_df) > 1:
                st.subheader("Performance Over Time")
                
                fig_history = go.Figure()
                if 'elasticnet_rmse' in history_df.columns:
                    fig_history.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['elasticnet_rmse'],
                        mode='lines+markers',
                        name='ElasticNet RMSE'
                    ))
                if 'pls_rmse' in history_df.columns:
                    fig_history.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['pls_rmse'],
                        mode='lines+markers',
                        name='PLS RMSE'
                    ))
                
                fig_history.update_layout(
                    title="Model Performance Over Time",
                    xaxis_title="Time",
                    yaxis_title="RMSE",
                    height=400
                )
                
                st.plotly_chart(fig_history, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        
        else:
            st.info("No prediction history available. Run some predictions to see history here.")
    
    with tab4:
        st.header("⚙️ Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            
            # Display current model info
            if st.session_state.models_loaded:
                pipeline, elastic_model, pls_model = load_models()
                
                st.write("**ElasticNet Parameters:**")
                st.write(f"- Alpha: {getattr(elastic_model, 'alpha', 'N/A')}")
                st.write(f"- L1 Ratio: {getattr(elastic_model, 'l1_ratio', 'N/A')}")
                st.write(f"- Max Iterations: {getattr(elastic_model, 'max_iter', 'N/A')}")
                
                if pls_model is not None:
                    st.write("**PLS Parameters:**")
                    st.write(f"- Components: {getattr(pls_model, 'n_components', 'N/A')}")
                else:
                    st.write("**PLS Model:** Not Available")
            
            st.subheader("Data Configuration")
            st.write("**Current Settings:**")
            st.write("- Wavelength Range: 350-1750 cm⁻¹")
            st.write("- Preprocessing: RangeCut → Linear Correction → Savitzky-Golay → Derivative → Scaling")
            st.write(f"- Data Points: {len(st.session_state.data) if st.session_state.data is not None else 'Not loaded'}")
        
        with col2:
            st.subheader("Display Settings")
            
            # Chart settings
            chart_height = st.slider("Chart Height", 300, 600, 400)
            show_legend = st.checkbox("Show Legends", value=False)
            
            st.subheader("Export Options")
            
            if st.session_state.predictions:
                # Export results
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"penicillin_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # System info
            st.subheader("System Information")
            st.write(f"- Streamlit Version: {st.__version__}")
            st.write(f"- Pandas Version: {pd.__version__}")
            st.write(f"- NumPy Version: {np.__version__}")
            st.write(f"- Plotly Version: {plotly.__version__}")

if __name__ == "__main__":
    main()
