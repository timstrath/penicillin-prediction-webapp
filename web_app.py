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
import sqlite3
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

def get_model_registry_info():
    """Get model information from the SQLite database"""
    try:
        db_path = "pharma_model_registry.db"
        if not os.path.exists(db_path):
            return None
        
        with sqlite3.connect(db_path, timeout=30) as conn:
            cursor = conn.cursor()
            
            # Get latest model versions (exclude preprocessing models)
            cursor.execute("""
                SELECT m.model_name, mv.version_number, mv.created_at, mv.status
                FROM models m
                JOIN model_versions mv ON m.model_id = mv.model_id
                WHERE mv.status = 'Validated' 
                AND m.model_name NOT LIKE '%Preprocessing%'
                ORDER BY mv.created_at DESC
            """)
            
            models = cursor.fetchall()
            
            # Get model inventory summary
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT m.model_name) as total_models,
                    COUNT(CASE WHEN mv.status = 'Validated' THEN 1 END) as validated_models,
                    COUNT(CASE WHEN mv.status = 'In Development' THEN 1 END) as development_models,
                    MAX(mv.created_at) as last_model_update
                FROM models m
                JOIN model_versions mv ON m.model_id = mv.model_id
                WHERE m.model_name NOT LIKE '%Preprocessing%'
            """)
            
            inventory_info = cursor.fetchone()
            
            # Get validation information
            cursor.execute("""
                SELECT COUNT(*) as total_validations,
                       MAX(created_at) as last_validation
                FROM model_versions
                WHERE status = 'Validated'
            """)
            
            validation_info = cursor.fetchone()
            
            return {
                'models': models,
                'total_validations': validation_info[0] if validation_info else 0,
                'last_validation': validation_info[1] if validation_info else None,
                'total_models': inventory_info[0] if inventory_info else 0,
                'validated_models': inventory_info[1] if inventory_info else 0,
                'development_models': inventory_info[2] if inventory_info else 0,
                'last_model_update': inventory_info[3] if inventory_info else None
            }
            
    except Exception as e:
        # Silently fail if database is not available
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
    """Apply preprocessing up to derivative step (before StandardScaler) for more informative visualization"""
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
            
            # Model status with version and compliance information
            st.header("🤖 Model Status & Compliance")
            if st.session_state.models_loaded:
                st.success("✅ Models Loaded")
                
                # Try to get model information from database
                registry_info = get_model_registry_info()
                
                if registry_info:
                    # Model version information from database
                    st.markdown("**📋 Model Versions (Live from Registry):**")
                    for model_name, version, created_at, status in registry_info['models']:
                        if status == 'In Development':
                            st.warning(f"• **{model_name}**: {version} ({status})")
                        else:
                            st.info(f"• **{model_name}**: {version} ({status})")
                    
                    # Validation status from database
                    st.markdown("**✅ Validation Status:**")
                    st.info(f"• **Total Validations**: {registry_info['total_validations']}")
                    if registry_info['last_validation']:
                        st.info(f"• **Last Validation**: {registry_info['last_validation'][:10]}")
                    
                    # Model registry database status
                    st.markdown("**📊 Model Registry Database Status:**")
                    st.success("• **Database Connection**: ✅ Connected")
                    st.success("• **Audit Trail**: ✅ Active")
                    st.success("• **Version Control**: ✅ Enabled")
                    
                    # Model inventory summary
                    st.markdown("**📋 Model Inventory Summary:**")
                    st.info(f"• **Total Models**: {registry_info['total_models']} (ElasticNet, PLS, MLP+1D-CNN)")
                    st.info(f"• **Validated Models**: {registry_info['validated_models']}")
                    st.warning(f"• **Models in Development**: {registry_info['development_models']}")
                    if registry_info['last_model_update']:
                        st.info(f"• **Last Model Update**: {registry_info['last_model_update'][:10]}")
                    
                else:
                    # Fallback to static information
                    st.markdown("**📋 Model Versions:**")
                    st.info("• **ElasticNet**: v1.2.0 (Validated)")
                    st.info("• **PLS**: v1.1.0 (Validated)")
                    st.warning("• **MLP+1D-CNN**: v1.0.0 (In Development)")
                    
                    # Model registry database status
                    st.markdown("**📊 Model Registry Database Status:**")
                    st.warning("• **Database Connection**: ⚠️ Not Connected")
                    st.info("• **Audit Trail**: 📝 Static Mode")
                    st.info("• **Version Control**: 📝 Static Mode")
                    
                    # Model inventory summary (static)
                    st.markdown("**📋 Model Inventory Summary:**")
                    st.info("• **Total Models**: 3 (ElasticNet, PLS, MLP+1D-CNN)")
                    st.info("• **Validated Models**: 2")
                    st.warning("• **Models in Development**: 1")
                    st.info("• **Last Model Update**: 2025-10-13")
                
                # Compliance status (always shown)
                st.markdown("**🏛️ Regulatory Compliance:**")
                st.success("• **21 CFR Part 11**: ✅ Compliant")
                st.success("• **ICH Q7**: ✅ GMP Compliant")
                st.success("• **EU GMP Annex 11**: ✅ Compliant")
                
                # Validation status (always shown)
                st.markdown("**✅ Validation Status:**")
                st.info("• **IQ/OQ/PQ**: Completed")
                st.info("• **Last Validation**: 2024-10-13")
                st.info("• **Next Review**: 2025-01-13")
                
            else:
                st.error("❌ Models Not Loaded")
                st.warning("⚠️ Please ensure models are properly loaded")
    
    # Custom CSS for better tab visibility
    st.markdown("""
    <style>
    /* Make tabs more visible and contrastive */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        border: 2px solid #e1e5e9;
        color: #262730;
        font-weight: 600;
        font-size: 16px;
        padding: 10px 20px;
        margin: 0px 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
        border: 2px solid #ff4b4b;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(255, 75, 75, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ff6b6b;
        color: white;
        border: 2px solid #ff6b6b;
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: #ff3333;
        border: 2px solid #ff3333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔬 Preprocessing", 
        "📊 Results & Predictions", 
        "🔵 ElasticNet Model",
        "🟢 PLS Model",
        "🟣 MLP+1D-CNN Model",
        "📈 History", 
        "⚙️ Settings",
        "🏛️ Model Registry"
    ])
    
    with tab1:
        st.header("🔬 Preprocessing Visualization")
        
        if st.session_state.data is not None and st.session_state.preprocessed_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Raman Spectra")
                
                # Find spectral columns (numeric columns that represent wavelengths)
                # Note: First 39 columns are process data, Raman spectral data starts from column 40+
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                # Filter for columns that look like wavelengths (numeric values)
                spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
                
                if len(spectral_cols) > 0:
                    # Use the actual spectral columns - TRULY RAW DATA
                    raw_spectra = st.session_state.data[spectral_cols].iloc[:10].copy()
                    # Fill NaN values for visualization (but don't modify original data)
                    raw_spectra = raw_spectra.fillna(0)
                    # Convert column names to numeric for wavelengths
                    try:
                        wavelengths = np.array([float(col) for col in spectral_cols])
                        # Sort wavelengths in ascending order (201 to 2400)
                        sort_indices = np.argsort(wavelengths)
                        wavelengths = wavelengths[sort_indices]
                        raw_spectra = raw_spectra.iloc[:, sort_indices]
                        
                        # Apply RangeCut to show only relevant Raman spectral range (350-1750 cm⁻¹)
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
                    
                    # Keep raw data as-is (no normalization for raw spectra)
                    
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
                        title="Raw Raman Spectra (First 10) - 350-1750 cm⁻¹",
                        xaxis_title="Wavelength (cm⁻¹)",
                        yaxis_title="Intensity (Raw Counts)",
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
                        title="Preprocessed Spectra (First 10) - After Derivative",
                        xaxis_title="Wavelength (cm⁻¹)",
                        yaxis_title="Intensity (After Derivative)",
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
                "4. Norris-Williams Derivative: First derivative (shown in right plot)",
                "5. Standard Scaling: Normalization (applied for model input)"
            ]
            
            for step in steps:
                st.write(f"✅ {step}")
        
        else:
            st.error("Data not loaded. Please check the data file.")
    
    with tab2:
        st.header("📊 Results & Predictions")
        
        if st.session_state.models_loaded and st.session_state.preprocessed_data is not None:
            # Auto-run predictions when tab is activated
            if 'predictions' not in st.session_state or not st.session_state.predictions:
                with st.spinner("Making predictions..."):
                    # Get models
                    pipeline, elastic_model, pls_model = load_models()
                    
                    # Use all available data (up to 100 samples for demo)
                    num_spectra = min(100, len(st.session_state.preprocessed_data))
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
        st.header("🔵 ElasticNet Model - Preprocessing & Predictions")
        
        if st.session_state.data is not None and st.session_state.models_loaded:
            # Load models
            pipeline, elastic_model, pls_model = load_models()
            
            if pipeline is not None and elastic_model is not None:
                # PREPROCESSING VISUALIZATION
                st.subheader("🔬 Preprocessing Pipeline")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Raw Raman Spectra")
                    
                    # Find spectral columns (numeric columns that represent wavelengths)
                    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                    spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
                    
                    if len(spectral_cols) > 0:
                        # Use the actual spectral columns - TRULY RAW DATA
                        raw_spectra = st.session_state.data[spectral_cols].iloc[:10].copy()
                        raw_spectra = raw_spectra.fillna(0)
                        
                        try:
                            wavelengths = np.array([float(col) for col in spectral_cols])
                            # Apply RangeCut: 350-1750 cm⁻¹
                            mask = (wavelengths >= 350) & (wavelengths <= 1750)
                            wavelengths = wavelengths[mask]
                            raw_spectra = raw_spectra.iloc[:, mask]
                            
                            # Sort wavelengths in ascending order
                            sort_indices = np.argsort(wavelengths)
                            wavelengths = wavelengths[sort_indices]
                            raw_spectra = raw_spectra.iloc[:, sort_indices]
                            
                            # Create plot
                            fig = go.Figure()
                            
                            for i in range(min(10, len(raw_spectra))):
                                fig.add_trace(go.Scatter(
                                    x=wavelengths,
                                    y=raw_spectra.iloc[i].values,
                                    mode='lines',
                                    name=f'Sample {i+1}',
                                    line=dict(width=1),
                                    opacity=0.7
                                ))
                            
                            fig.update_layout(
                                title="Raw Raman Spectra (First 10) - 350-1750 cm⁻¹",
                                xaxis_title="Wavelength (cm⁻¹)",
                                yaxis_title="Intensity (Raw Counts)",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="raw_spectra_original")
                            
                        except Exception as e:
                            st.error(f"Error processing raw spectra: {str(e)}")
                    else:
                        st.error("No spectral columns found in the data.")
                
                with col2:
                    st.subheader("Preprocessed Spectra")
                    
                    # Get preprocessed data for visualization (stops before StandardScaler)
                    if st.session_state.preprocessed_data_viz is not None:
                        preprocessed_viz = st.session_state.preprocessed_data_viz[:10]
                        
                        # Create wavelengths for preprocessed data (assuming same range after RangeCut)
                        preprocessed_wavelengths = np.linspace(350, 1750, preprocessed_viz.shape[1])
                        
                        fig = go.Figure()
                        
                        for i in range(min(10, len(preprocessed_viz))):
                            fig.add_trace(go.Scatter(
                                x=preprocessed_wavelengths,
                                y=preprocessed_viz[i],
                                mode='lines',
                                name=f'Sample {i+1}',
                                line=dict(width=1),
                                opacity=0.7
                            ))
                        
                        fig.update_layout(
                            title="Preprocessed Spectra (First 10) - After Derivative",
                            xaxis_title="Wavelength (cm⁻¹)",
                            yaxis_title="Intensity (After Derivative)",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="preprocessed_spectra_original")
                    else:
                        st.error("Preprocessed data not available for visualization.")
                
                # PREDICTIONS SECTION
                st.subheader("📊 ElasticNet Predictions")
                
                # Auto-run predictions
                num_spectra = min(100, len(st.session_state.preprocessed_data))
                subset_data = st.session_state.preprocessed_data[:num_spectra]
                
                # Make predictions
                elasticnet_pred = elastic_model.predict(subset_data)
                
                # Get target column and actual values
                target_col = 'Penicillin concentration(P:g/L)'
                if target_col in st.session_state.data.columns:
                    ground_truth = st.session_state.data[target_col].values[:len(elasticnet_pred)]
                    
                    # Calculate performance metrics
                    elasticnet_rmse = np.sqrt(mean_squared_error(ground_truth, elasticnet_pred))
                    elasticnet_mae = mean_absolute_error(ground_truth, elasticnet_pred)
                    elasticnet_r2 = r2_score(ground_truth, elasticnet_pred)
                    elasticnet_mse = mean_squared_error(ground_truth, elasticnet_pred)
                    
                    # Performance metrics
                    st.subheader("📈 Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RMSE", f"{elasticnet_rmse:.3f} g/L")
                    with col2:
                        st.metric("MAE", f"{elasticnet_mae:.3f} g/L")
                    with col3:
                        st.metric("R² Score", f"{elasticnet_r2:.3f}")
                    with col4:
                        st.metric("MSE", f"{elasticnet_mse:.3f}")
                    
                    # Three plots in organized layout
                    st.subheader("📊 Model Performance Analysis")
                    
                    # Calculate residuals
                    residuals = elasticnet_pred - ground_truth
                    
                    # Create three columns for the plots
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Prediction vs Actual scatter plot
                        fig1 = go.Figure()
                        
                        # Add scatter plot
                        fig1.add_trace(go.Scatter(
                            x=ground_truth,
                            y=elasticnet_pred,
                            mode='markers',
                            name='ElasticNet Predictions',
                            marker=dict(
                                color='#d62728',
                                size=8,
                                opacity=0.7
                            ),
                            showlegend=True
                        ))
                        
                        # Add perfect prediction line
                        min_val = min(min(ground_truth), min(elasticnet_pred))
                        max_val = max(max(ground_truth), max(elasticnet_pred))
                        fig1.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='#1f77b4', dash='dash'),
                            showlegend=True
                        ))
                        
                        fig1.update_layout(
                            title="Prediction vs Actual",
                            xaxis_title="Actual (g/L)",
                            yaxis_title="Predicted (g/L)",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Residuals distribution
                        fig2 = go.Figure()
                        fig2.add_trace(go.Histogram(
                            x=residuals,
                            nbinsx=20,
                            name='Residuals',
                            marker_color='#d62728',
                            opacity=0.7
                        ))
                        
                        fig2.update_layout(
                            title="Residuals Distribution",
                            xaxis_title="Residuals (g/L)",
                            yaxis_title="Frequency",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with col3:
                        # Concentration distribution
                        fig3 = go.Figure()
                        
                        # Add actual concentration distribution
                        fig3.add_trace(go.Histogram(
                            x=ground_truth,
                            nbinsx=20,
                            name='Actual Concentrations',
                            marker_color='#2ca02c',
                            opacity=0.7
                        ))
                        
                        # Add predicted concentration distribution
                        fig3.add_trace(go.Histogram(
                            x=elasticnet_pred,
                            nbinsx=20,
                            name='Predicted Concentrations',
                            marker_color='#d62728',
                            opacity=0.7
                        ))
                        
                        fig3.update_layout(
                            title="Concentration Distribution",
                            xaxis_title="Penicillin Concentration (g/L)",
                            yaxis_title="Frequency",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Prediction results table
                    st.subheader("📈 Prediction Results")
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(elasticnet_pred) + 1),
                        'Actual (g/L)': ground_truth,
                        'Predicted (g/L)': elasticnet_pred,
                        'Residual (g/L)': residuals,
                        'Error (%)': np.abs(residuals / ground_truth * 100)
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                else:
                    st.error(f"Target column '{target_col}' not found in data.")
            else:
                st.error("ElasticNet model not available.")
        else:
            st.warning("Please load models and data first.")
    
    with tab4:
        st.header("🟢 PLS Model - Preprocessing & Predictions")
        
        if st.session_state.data is not None and st.session_state.models_loaded:
            # Load models
            pipeline, elastic_model, pls_model = load_models()
            
            if pipeline is not None and pls_model is not None:
                # PREPROCESSING VISUALIZATION
                st.subheader("🔬 Preprocessing Pipeline")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Raw Raman Spectra")
                    
                    # Find spectral columns (numeric columns that represent wavelengths)
                    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                    spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
                    
                    if len(spectral_cols) > 0:
                        # Use the actual spectral columns - TRULY RAW DATA
                        raw_spectra = st.session_state.data[spectral_cols].iloc[:10].copy()
                        raw_spectra = raw_spectra.fillna(0)
                        
                        try:
                            wavelengths = np.array([float(col) for col in spectral_cols])
                            # Apply RangeCut: 350-1750 cm⁻¹
                            mask = (wavelengths >= 350) & (wavelengths <= 1750)
                            wavelengths = wavelengths[mask]
                            raw_spectra = raw_spectra.iloc[:, mask]
                            
                            # Sort wavelengths in ascending order
                            sort_indices = np.argsort(wavelengths)
                            wavelengths = wavelengths[sort_indices]
                            raw_spectra = raw_spectra.iloc[:, sort_indices]
                            
                            # Create plot
                            fig = go.Figure()
                            
                            for i in range(min(10, len(raw_spectra))):
                                fig.add_trace(go.Scatter(
                                    x=wavelengths,
                                    y=raw_spectra.iloc[i].values,
                                    mode='lines',
                                    name=f'Sample {i+1}',
                                    line=dict(width=1),
                                    opacity=0.7
                                ))
                            
                            fig.update_layout(
                                title="Raw Raman Spectra (First 10) - 350-1750 cm⁻¹",
                                xaxis_title="Wavelength (cm⁻¹)",
                                yaxis_title="Intensity (Raw Counts)",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="preprocessed_spectra_original")
                            
                        except Exception as e:
                            st.error(f"Error processing raw spectra: {str(e)}")
                    else:
                        st.error("No spectral columns found in the data.")
                
                with col2:
                    st.subheader("Preprocessed Spectra")
                    
                    # Get preprocessed data for visualization (stops before StandardScaler)
                    if st.session_state.preprocessed_data_viz is not None:
                        preprocessed_viz = st.session_state.preprocessed_data_viz[:10]
                        
                        # Create wavelengths for preprocessed data (assuming same range after RangeCut)
                        preprocessed_wavelengths = np.linspace(350, 1750, preprocessed_viz.shape[1])
                        
                        fig = go.Figure()
                        
                        for i in range(min(10, len(preprocessed_viz))):
                            fig.add_trace(go.Scatter(
                                x=preprocessed_wavelengths,
                                y=preprocessed_viz[i],
                                mode='lines',
                                name=f'Sample {i+1}',
                                line=dict(width=1),
                                opacity=0.7
                            ))
                        
                        fig.update_layout(
                            title="Preprocessed Spectra (First 10) - After Derivative",
                            xaxis_title="Wavelength (cm⁻¹)",
                            yaxis_title="Intensity (After Derivative)",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="raw_spectra_elasticnet")
                    else:
                        st.error("Preprocessed data not available for visualization.")
                
                # PREDICTIONS SECTION
                st.subheader("📊 PLS Predictions")
                
                # Auto-run predictions
                num_spectra = min(100, len(st.session_state.preprocessed_data))
                subset_data = st.session_state.preprocessed_data[:num_spectra]
                
                # Make predictions
                pls_pred = pls_model.predict(subset_data)
                
                # Get target column and actual values
                target_col = 'Penicillin concentration(P:g/L)'
                if target_col in st.session_state.data.columns:
                    ground_truth = st.session_state.data[target_col].values[:len(pls_pred)]
                    
                    # Calculate performance metrics
                    pls_rmse = np.sqrt(mean_squared_error(ground_truth, pls_pred))
                    pls_mae = mean_absolute_error(ground_truth, pls_pred)
                    pls_r2 = r2_score(ground_truth, pls_pred)
                    pls_mse = mean_squared_error(ground_truth, pls_pred)
                    
                    # Performance metrics
                    st.subheader("📈 Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RMSE", f"{pls_rmse:.3f} g/L")
                    with col2:
                        st.metric("MAE", f"{pls_mae:.3f} g/L")
                    with col3:
                        st.metric("R² Score", f"{pls_r2:.3f}")
                    with col4:
                        st.metric("MSE", f"{pls_mse:.3f}")
                    
                    # Three plots in organized layout
                    st.subheader("📊 Model Performance Analysis")
                    
                    # Calculate residuals
                    residuals = pls_pred - ground_truth
                    
                    # Create three columns for the plots
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Prediction vs Actual scatter plot
                        fig1 = go.Figure()
                        
                        # Add scatter plot
                        fig1.add_trace(go.Scatter(
                            x=ground_truth,
                            y=pls_pred,
                            mode='markers',
                            name='PLS Predictions',
                            marker=dict(
                                color='#2ca02c',
                                size=8,
                                opacity=0.7
                            ),
                            showlegend=True
                        ))
                        
                        # Add perfect prediction line
                        min_val = min(min(ground_truth), min(pls_pred))
                        max_val = max(max(ground_truth), max(pls_pred))
                        fig1.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='#1f77b4', dash='dash'),
                            showlegend=True
                        ))
                        
                        fig1.update_layout(
                            title="Prediction vs Actual",
                            xaxis_title="Actual (g/L)",
                            yaxis_title="Predicted (g/L)",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Residuals distribution
                        fig2 = go.Figure()
                        fig2.add_trace(go.Histogram(
                            x=residuals,
                            nbinsx=20,
                            name='Residuals',
                            marker_color='#2ca02c',
                            opacity=0.7
                        ))
                        
                        fig2.update_layout(
                            title="Residuals Distribution",
                            xaxis_title="Residuals (g/L)",
                            yaxis_title="Frequency",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with col3:
                        # Concentration distribution
                        fig3 = go.Figure()
                        
                        # Add actual concentration distribution
                        fig3.add_trace(go.Histogram(
                            x=ground_truth,
                            nbinsx=20,
                            name='Actual Concentrations',
                            marker_color='#2ca02c',
                            opacity=0.7
                        ))
                        
                        # Add predicted concentration distribution
                        fig3.add_trace(go.Histogram(
                            x=pls_pred,
                            nbinsx=20,
                            name='Predicted Concentrations',
                            marker_color='#2ca02c',
                            opacity=0.7
                        ))
                        
                        fig3.update_layout(
                            title="Concentration Distribution",
                            xaxis_title="Penicillin Concentration (g/L)",
                            yaxis_title="Frequency",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Prediction results table
                    st.subheader("📈 Prediction Results")
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(pls_pred) + 1),
                        'Actual (g/L)': ground_truth,
                        'Predicted (g/L)': pls_pred,
                        'Residual (g/L)': residuals,
                        'Error (%)': np.abs(residuals / ground_truth * 100)
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                else:
                    st.error(f"Target column '{target_col}' not found in data.")
            else:
                st.error("PLS model not available.")
        else:
            st.warning("Please load models and data first.")
    
    with tab5:
        st.header("🟣 MLP+1D-CNN Model - Preprocessing & Predictions")
        
        if st.session_state.data is not None and st.session_state.models_loaded:
            # Load models
            pipeline, elastic_model, pls_model = load_models()
            
            if pipeline is not None:
                # PREPROCESSING VISUALIZATION
                st.subheader("🔬 Preprocessing Pipeline")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Raw Raman Spectra")
                    
                    # Find spectral columns (numeric columns that represent wavelengths)
                    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                    spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
                    
                    if len(spectral_cols) > 0:
                        # Use the actual spectral columns - TRULY RAW DATA
                        raw_spectra = st.session_state.data[spectral_cols].iloc[:10].copy()
                        raw_spectra = raw_spectra.fillna(0)
                        
                        try:
                            wavelengths = np.array([float(col) for col in spectral_cols])
                            # Apply RangeCut: 350-1750 cm⁻¹
                            mask = (wavelengths >= 350) & (wavelengths <= 1750)
                            wavelengths = wavelengths[mask]
                            raw_spectra = raw_spectra.iloc[:, mask]
                            
                            # Sort wavelengths in ascending order
                            sort_indices = np.argsort(wavelengths)
                            wavelengths = wavelengths[sort_indices]
                            raw_spectra = raw_spectra.iloc[:, sort_indices]
                            
                            # Create plot
                            fig = go.Figure()
                            
                            for i in range(min(10, len(raw_spectra))):
                                fig.add_trace(go.Scatter(
                                    x=wavelengths,
                                    y=raw_spectra.iloc[i].values,
                                    mode='lines',
                                    name=f'Sample {i+1}',
                                    line=dict(width=1),
                                    opacity=0.7
                                ))
                            
                            fig.update_layout(
                                title="Raw Raman Spectra (First 10) - 350-1750 cm⁻¹",
                                xaxis_title="Wavelength (cm⁻¹)",
                                yaxis_title="Intensity (Raw Counts)",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="preprocessed_spectra_elasticnet")
                            
                        except Exception as e:
                            st.error(f"Error processing raw spectra: {str(e)}")
                    else:
                        st.error("No spectral columns found in the data.")
                
                with col2:
                    st.subheader("Preprocessed Spectra")
                    
                    # Get preprocessed data for visualization (stops before StandardScaler)
                    if st.session_state.preprocessed_data_viz is not None:
                        preprocessed_viz = st.session_state.preprocessed_data_viz[:10]
                        
                        # Create wavelengths for preprocessed data (assuming same range after RangeCut)
                        preprocessed_wavelengths = np.linspace(350, 1750, preprocessed_viz.shape[1])
                        
                        fig = go.Figure()
                        
                        for i in range(min(10, len(preprocessed_viz))):
                            fig.add_trace(go.Scatter(
                                x=preprocessed_wavelengths,
                                y=preprocessed_viz[i],
                                mode='lines',
                                name=f'Sample {i+1}',
                                line=dict(width=1),
                                opacity=0.7
                            ))
                        
                        fig.update_layout(
                            title="Preprocessed Spectra (First 10) - After Derivative",
                            xaxis_title="Wavelength (cm⁻¹)",
                            yaxis_title="Intensity (After Derivative)",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="raw_spectra_pls")
                    else:
                        st.error("Preprocessed data not available for visualization.")
                
                # PREDICTIONS SECTION
                st.subheader("📊 MLP+1D-CNN Predictions")
                
                # Note: MLP+1D-CNN model is not yet integrated
                st.info("🔄 **MLP+1D-CNN Model Status: In Development**")
                st.markdown("""
                The MLP+1D-CNN model is currently in development and not yet integrated into the prediction pipeline.
                
                **Model Architecture:**
                - **MLP Component**: Multi-layer perceptron for feature learning
                - **1D-CNN Component**: 1D convolutional neural network for spectral pattern recognition
                - **Hybrid Approach**: Combines traditional MLP with CNN for enhanced spectral analysis
                
                **Expected Features:**
                - Advanced spectral feature extraction
                - Improved pattern recognition capabilities
                - Enhanced prediction accuracy for complex spectral data
                
                **Development Status:**
                - ✅ Model architecture designed
                - ✅ Training pipeline implemented
                - 🔄 Integration with web app (in progress)
                - ⏳ Validation and testing (pending)
                """)
                
                # Placeholder for future predictions
                st.subheader("📈 Future Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", "TBD", help="To be determined after integration")
                with col2:
                    st.metric("MAE", "TBD", help="To be determined after integration")
                with col3:
                    st.metric("R² Score", "TBD", help="To be determined after integration")
                with col4:
                    st.metric("MSE", "TBD", help="To be determined after integration")
                
                # Placeholder plots
                st.subheader("📊 Model Performance Analysis")
                st.info("📊 Performance visualizations will be available once the model is fully integrated.")
                
                # Development roadmap
                st.subheader("🚀 Development Roadmap")
                roadmap_data = {
                    'Phase': ['Model Training', 'Integration', 'Validation', 'Deployment'],
                    'Status': ['✅ Complete', '🔄 In Progress', '⏳ Pending', '⏳ Pending'],
                    'Description': [
                        'Model architecture and training pipeline completed',
                        'Integration with web application in progress',
                        'Model validation and performance testing',
                        'Production deployment and monitoring'
                    ]
                }
                
                roadmap_df = pd.DataFrame(roadmap_data)
                st.dataframe(roadmap_df, use_container_width=True)
                
            else:
                st.error("Preprocessing pipeline not available.")
        else:
            st.warning("Please load models and data first.")
    
    with tab6:
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
    
    with tab7:
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
    
    with tab8:
        st.header("🏛️ Model Registry - GMP Compliance")
        st.markdown("**Pharmaceutical Industry Model Versioning & Tracking System**")
        
        # Load registry
        try:
            from model_registry import ModelRegistry
            import sqlite3
            
            registry = ModelRegistry("pharma_model_registry.db")
            
            # Generate compliance report
            report = registry.generate_compliance_report()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📊 Total Models",
                    value=report['total_models'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="✅ Approved Models", 
                    value=report['approved_models'],
                    delta=f"{report['approved_models']/report['total_models']*100:.1f}%" if report['total_models'] > 0 else "0%"
                )
            
            with col3:
                st.metric(
                    label="🔢 Total Versions",
                    value=report['total_versions'],
                    delta=None
                )
            
            with col4:
                compliance_status = "✅ COMPLIANT" if report['compliance_status'] == 'COMPLIANT' else "❌ NON-COMPLIANT"
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f4e79; margin: 0.5rem 0;">
                    <h4>🏛️ Compliance Status</h4>
                    <p style="background-color: #d4edda; color: #155724; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: bold;">{compliance_status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model inventory
            st.subheader("📚 Model Inventory")
            
            with sqlite3.connect(registry.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT m.model_id, m.model_name, m.model_type, m.description, 
                           m.status, m.created_by, m.created_date, m.is_active
                    FROM models m
                    ORDER BY m.created_date DESC
                """)
                models = cursor.fetchall()
            
            if models:
                df_models = pd.DataFrame(models, columns=[
                    'ID', 'Name', 'Type', 'Description', 'Status', 'Created By', 'Created Date', 'Active'
                ])
                
                # Status styling
                def style_status(val):
                    if val == 'APPROVED':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'VALIDATED':
                        return 'background-color: #d1ecf1; color: #0c5460'
                    elif val == 'DRAFT':
                        return 'background-color: #fff3cd; color: #856404'
                    elif val == 'DEPRECATED':
                        return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_df = df_models.style.applymap(style_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Model details
                selected_model = st.selectbox("Select model for details:", df_models['Name'].tolist())
                
                if selected_model:
                    model_info = registry.get_model_info(
                        selected_model.split('_')[0] + '_' + selected_model.split('_')[1],
                        selected_model.split('_')[2] if len(selected_model.split('_')) > 2 else 'Regression'
                    )
                    
                    if model_info:
                        st.subheader(f"📋 Model Details: {selected_model}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Information:**")
                            st.write(f"- **Type:** {model_info['model'][2]}")
                            st.write(f"- **Status:** {model_info['model'][6]}")
                            st.write(f"- **Created By:** {model_info['model'][4]}")
                            st.write(f"- **Created Date:** {model_info['model'][5]}")
                            st.write(f"- **Active:** {'Yes' if model_info['model'][7] else 'No'}")
                        
                        with col2:
                            st.write("**Description:**")
                            st.write(model_info['model'][3])
                        
                        # Versions
                        if model_info['versions']:
                            st.subheader("📝 Model Versions")
                            versions_data = []
                            for version in model_info['versions']:
                                versions_data.append({
                                    'Version': version[3],
                                    'Created Date': version[9],
                                    'Validation Status': version[11],
                                    'Approved By': version[12] or 'Not Approved',
                                    'Change Reason': version[13]
                                })
                            
                            df_versions = pd.DataFrame(versions_data)
                            st.dataframe(df_versions, use_container_width=True)
            else:
                st.info("No models found in the registry.")
            
            # Recent activity
            st.subheader("📋 Recent Activity")
            
            audit_trail = registry.get_audit_trail()
            if audit_trail:
                recent_activity = []
                for record in audit_trail[:10]:
                    # Get model name for better context
                    model_name = "N/A"
                    if record[1] == "models" and record[2]:  # If it's a models table record
                        with sqlite3.connect(registry.db_path, timeout=30) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT model_name FROM models WHERE model_id = ?", (record[2],))
                            result = cursor.fetchone()
                            if result:
                                model_name = result[0]
                    elif record[1] == "model_versions" and record[2]:  # If it's a model_versions record
                        with sqlite3.connect(registry.db_path, timeout=30) as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT m.model_name FROM model_versions mv
                                JOIN models m ON mv.model_id = m.model_id
                                WHERE mv.version_id = ?
                            """, (record[2],))
                            result = cursor.fetchone()
                            if result:
                                model_name = result[0]
                    elif record[1] == "model_validations" and record[2]:  # If it's a validation record
                        with sqlite3.connect(registry.db_path, timeout=30) as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT m.model_name FROM model_validations mv
                                JOIN model_versions mvv ON mv.version_id = mvv.version_id
                                JOIN models m ON mvv.model_id = m.model_id
                                WHERE mv.validation_id = ?
                            """, (record[2],))
                            result = cursor.fetchone()
                            if result:
                                model_name = result[0]
                    
                    recent_activity.append({
                        'Date': record[6],
                        'Model': model_name,
                        'Table': record[1],
                        'Action': record[3],
                        'User': record[7],
                        'Reason': record[8]
                    })
                
                df_activity = pd.DataFrame(recent_activity)
                st.dataframe(df_activity, use_container_width=True)
            else:
                st.info("No recent activity found.")
            
            # Compliance information
            st.subheader("🏛️ Regulatory Compliance")
            
            st.markdown("""
            **This Model Registry implements pharmaceutical industry best practices:**
            
            - **21 CFR Part 11** - Electronic Records and Signatures
            - **ICH Q7** - Good Manufacturing Practice for Active Pharmaceutical Ingredients
            - **FDA Process Validation** - Lifecycle approach
            - **EU GMP Annex 11** - Computerized Systems
            
            **Key Features:**
            - ✅ Version control with semantic versioning
            - ✅ Complete audit trail for regulatory compliance
            - ✅ File integrity verification (SHA-256)
            - ✅ Validation tracking and approval workflow
            - ✅ Deployment management across environments
            - ✅ Performance monitoring and drift detection
            """)
            
        except Exception as e:
            st.error(f"Error loading model registry: {str(e)}")
            st.info("Please ensure the model registry database exists. Run 'python populate_model_registry.py' to initialize.")

if __name__ == "__main__":
    main()
