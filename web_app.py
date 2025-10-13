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

# @st.cache_data  # Temporarily disabled to force data reload
def load_data():
    """Load the spectral data"""
    try:
        data_file = './test_data/test_samples.csv'  # Test data for deployment
        # Add cache busting - check file modification time and size
        import os
        file_mtime = os.path.getmtime(data_file) if os.path.exists(data_file) else 0
        file_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
        
        if os.path.exists(data_file):
            # Load exactly 100 samples for demonstration
            data = pd.read_csv(data_file, nrows=100)
            
            # Debug: Show data info
            target_col = 'Penicillin concentration(P:g/L)'
            if target_col in data.columns:
                st.sidebar.write(f"📊 **Data Info:**")
                st.sidebar.write(f"• Samples: {len(data)}")
                st.sidebar.write(f"• Range: {data[target_col].min():.3f} - {data[target_col].max():.3f} g/L")
                st.sidebar.write(f"• Mean: {data[target_col].mean():.3f} g/L")
                st.sidebar.write(f"• Std: {data[target_col].std():.3f} g/L")
            
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
                
                # Calculate performance metrics first (needed for plots)
                elasticnet_pred = results_df['ElasticNet_Prediction'].values
                pls_pred = results_df['PLS_Prediction'].values if st.session_state.predictions['pls'] is not None else None
                
                # Use the real target values from the test data as ground truth
                target_col = 'Penicillin concentration(P:g/L)'
                # Slice ground truth to match the number of predictions made
                ground_truth = st.session_state.data[target_col].values[:len(elasticnet_pred)]
                
                # 1. MODEL COMPARISON PLOTS (TOP)
                if st.session_state.predictions['pls'] is not None:
                    st.subheader("📊 Model Comparison")
                    
                    # Three plots in a row: Prediction vs Actual, Residuals Distribution, Concentration Distribution
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Plot 1: Prediction vs Actual Values
                        st.subheader("🎯 Prediction vs Actual")
                        
                        # Use the real target values from the test data
                        actual_values = st.session_state.data[target_col].values[:len(elasticnet_pred)]
                        
                        fig_actual = go.Figure()
                        
                        # Add ElasticNet predictions
                        fig_actual.add_trace(go.Scatter(
                            x=actual_values,
                            y=results_df['ElasticNet_Prediction'],
                            mode='markers',
                            name='ElasticNet',
                            marker=dict(size=6, color='#d62728', opacity=0.7)  # Dark red for better contrast
                        ))
                        
                        # Add PLS predictions
                        fig_actual.add_trace(go.Scatter(
                            x=actual_values,
                            y=results_df['PLS_Prediction'],
                            mode='markers',
                            name='PLS',
                            marker=dict(size=6, color='#2ca02c', opacity=0.7)  # Distinctive green
                        ))
                        
                        # Add perfect prediction line (y=x)
                        min_val = min(actual_values.min(), results_df['ElasticNet_Prediction'].min(), results_df['PLS_Prediction'].min())
                        max_val = max(actual_values.max(), results_df['ElasticNet_Prediction'].max(), results_df['PLS_Prediction'].max())
                        fig_actual.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='#1f77b4', width=2)
                        ))
                        
                        fig_actual.update_layout(
                            title="Model Performance",
                            xaxis_title="Actual (g/L)",
                            yaxis_title="Predicted (g/L)",
                            height=400,
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig_actual, use_container_width=True)
                    
                    with col2:
                        # Plot 2: Residuals Distribution
                        st.subheader("📊 Residuals Distribution")
                        
                        # Calculate residuals
                        elasticnet_residuals = actual_values - results_df['ElasticNet_Prediction']
                        pls_residuals = actual_values - results_df['PLS_Prediction']
                        
                        fig_residuals = go.Figure()
                        fig_residuals.add_trace(go.Histogram(
                            x=elasticnet_residuals,
                            name='ElasticNet',
                            opacity=0.7,
                            nbinsx=15,
                            marker_color='#d62728'  # Dark red for better contrast
                        ))
                        fig_residuals.add_trace(go.Histogram(
                            x=pls_residuals,
                            name='PLS',
                            opacity=0.7,
                            nbinsx=15,
                            marker_color='#2ca02c'  # Same green as scatter plot
                        ))
                        
                        # Add vertical line at zero
                        fig_residuals.add_vline(x=0, line_dash="dash", line_color="#1f77b4", 
                                              annotation_text="Perfect Prediction", 
                                              annotation_position="top")
                        
                        fig_residuals.update_layout(
                            title="Prediction Errors",
                            xaxis_title="Residuals (Actual - Predicted) g/L",
                            yaxis_title="Frequency",
                            height=400,
                            barmode='overlay',
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig_residuals, use_container_width=True)
                    
                    with col3:
                        # Plot 3: Concentration Distribution
                        st.subheader("📊 Concentration Distribution")
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=results_df['ElasticNet_Prediction'],
                            name='ElasticNet',
                            opacity=0.7,
                            nbinsx=15,
                            marker_color='#d62728'  # Dark red for better contrast
                        ))
                        fig_dist.add_trace(go.Histogram(
                            x=results_df['PLS_Prediction'],
                            name='PLS',
                            opacity=0.7,
                            nbinsx=15,
                            marker_color='#2ca02c'  # Same green as scatter plot
                        ))
                        
                        fig_dist.update_layout(
                            title="Prediction Distribution",
                            xaxis_title="Concentration (g/L)",
                            yaxis_title="Frequency",
                            height=400,
                            barmode='overlay',
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                # 2. MODEL PERFORMANCE COMPARISON (MIDDLE)
                st.subheader("📊 Model Performance Comparison")
                
                
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
                
                # Debug: Show data statistics to understand R² scores
                st.write("🔍 **Debug Info:**")
                st.write(f"**Ground Truth Range:** {ground_truth.min():.3f} - {ground_truth.max():.3f} g/L")
                st.write(f"**Ground Truth Mean:** {ground_truth.mean():.3f} g/L")
                st.write(f"**Ground Truth Std:** {ground_truth.std():.3f} g/L")
                st.write(f"**ElasticNet Predictions Range:** {elasticnet_pred.min():.3f} - {elasticnet_pred.max():.3f} g/L")
                if pls_pred is not None:
                    st.write(f"**PLS Predictions Range:** {pls_pred.min():.3f} - {pls_pred.max():.3f} g/L")
                
                # Calculate variance of ground truth (needed for R² interpretation)
                ground_truth_var = np.var(ground_truth)
                st.write(f"**Ground Truth Variance:** {ground_truth_var:.3f}")
                st.write(f"**ElasticNet MSE vs Variance:** {elasticnet_mse:.3f} vs {ground_truth_var:.3f}")
                if pls_pred is not None:
                    st.write(f"**PLS MSE vs Variance:** {pls_mse:.3f} vs {ground_truth_var:.3f}")
                
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
                
                
                # 3. PREDICTION TABLE (BOTTOM)
                st.subheader("📈 Prediction Results")
                st.dataframe(results_df, use_container_width=True)
        
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
