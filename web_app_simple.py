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
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_data
def load_data():
    """Load the test data"""
    try:
        data_file = './test_data/test_samples.csv'
        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            # Simple preprocessing - just fill NaN values
            data = data.fillna(0)
            return data
        else:
            st.error(f"Data file not found: {data_file}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_models():
    """Load the trained models"""
    try:
        models = {}
        
        # Try to load PLS model
        pls_path = './app/models/pls_penicillin.pkl'
        if os.path.exists(pls_path):
            models['pls'] = joblib.load(pls_path)
            st.success("✅ PLS model loaded successfully!")
        
        # Try to load ElasticNet model
        elastic_path = './app/models/elasticnet_penicillin.pkl'
        if os.path.exists(elastic_path):
            models['elasticnet'] = joblib.load(elastic_path)
            st.success("✅ ElasticNet model loaded successfully!")
        
        # Try to load scaler
        scaler_path = './app/models/scaler.pkl'
        if os.path.exists(scaler_path):
            models['scaler'] = joblib.load(scaler_path)
            st.success("✅ Scaler loaded successfully!")
        
        return models if models else None
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def simple_preprocessing(data):
    """Simple preprocessing without chemotools"""
    try:
        # Get process features (first 4 columns)
        process_features = ['Time (h)', 'Temperature(°C)', 'pH', 'DO(%)']
        X_process = data[process_features].values
        
        # Get spectral features (remaining columns except target)
        target_col = 'Penicillin concentration(P:g/L)'
        spectral_features = [col for col in data.columns if col not in process_features + [target_col]]
        X_spectral = data[spectral_features].values
        
        # Simple preprocessing - just standardize
        from sklearn.preprocessing import StandardScaler
        scaler_process = StandardScaler()
        scaler_spectral = StandardScaler()
        
        X_process_scaled = scaler_process.fit_transform(X_process)
        X_spectral_scaled = scaler_spectral.fit_transform(X_spectral)
        
        return X_process_scaled, X_spectral_scaled, spectral_features
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None, None

def make_prediction(models, X_process, X_spectral):
    """Make predictions using available models"""
    predictions = {}
    
    try:
        # Combine features for traditional models
        X_combined = np.concatenate([X_process, X_spectral], axis=1)
        
        # PLS prediction
        if 'pls' in models and 'scaler' in models:
            X_scaled = models['scaler'].transform(X_combined)
            predictions['PLS'] = models['pls'].predict(X_scaled)
        
        # ElasticNet prediction
        if 'elasticnet' in models and 'scaler' in models:
            X_scaled = models['scaler'].transform(X_combined)
            predictions['ElasticNet'] = models['elasticnet'].predict(X_scaled)
        
        return predictions
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return {}

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 Penicillin Concentration Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        data = load_data()
        models = load_models()
    
    if data is None:
        st.error("❌ Failed to load data. Please check the files.")
        return
    
    if models is None:
        st.error("❌ Failed to load models. Please check the files.")
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data overview
    st.sidebar.subheader("📈 Data Overview")
    st.sidebar.metric("Total Samples", len(data))
    st.sidebar.metric("Features", len(data.columns) - 1)
    
    # Target statistics
    target_col = 'Penicillin concentration(P:g/L)'
    if target_col in data.columns:
        target_stats = data[target_col].describe()
        st.sidebar.metric("Mean Concentration", f"{target_stats['mean']:.3f} g/L")
        st.sidebar.metric("Max Concentration", f"{target_stats['max']:.3f} g/L")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "🔮 Predictions", "📈 Model Performance"])
    
    with tab1:
        st.header("📊 Data Visualization")
        
        # Target distribution
        if target_col in data.columns:
            fig = px.histogram(data, x=target_col, nbins=30, 
                             title="Penicillin Concentration Distribution",
                             labels={'Penicillin concentration(P:g/L)': 'Concentration (g/L)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Process variables
        process_features = ['Time (h)', 'Temperature(°C)', 'pH', 'DO(%)']
        available_process = [col for col in process_features if col in data.columns]
        
        if available_process:
            st.subheader("Process Variables")
            fig = make_subplots(rows=2, cols=2, subplot_titles=available_process)
            
            for i, feature in enumerate(available_process):
                row = (i // 2) + 1
                col = (i % 2) + 1
                fig.add_trace(go.Histogram(x=data[feature], name=feature), row=row, col=col)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔮 Model Predictions")
        
        # Preprocess data
        X_process, X_spectral, spectral_features = simple_preprocessing(data)
        
        if X_process is not None:
            # Make predictions
            predictions = make_prediction(models, X_process, X_spectral)
            
            if predictions:
                st.success("✅ Predictions generated successfully!")
                
                # Display predictions
                for model_name, pred in predictions.items():
                    st.subheader(f"{model_name} Predictions")
                    
                    # Create prediction dataframe
                    pred_df = pd.DataFrame({
                        'Sample': range(len(pred)),
                        'Predicted Concentration (g/L)': pred.flatten()
                    })
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Prediction", f"{pred.mean():.3f} g/L")
                    with col2:
                        st.metric("Std Prediction", f"{pred.std():.3f} g/L")
                    with col3:
                        st.metric("Max Prediction", f"{pred.max():.3f} g/L")
                    
                    # Plot predictions
                    fig = px.line(pred_df, x='Sample', y='Predicted Concentration (g/L)',
                                title=f'{model_name} Predictions Over Samples')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ No predictions could be generated. Check model compatibility.")
    
    with tab3:
        st.header("📈 Model Performance")
        
        st.info("""
        **Model Performance Summary:**
        
        - **PLS Model**: Excellent performance with R² = 0.9997, RMSE = 0.1721 g/L
        - **ElasticNet Model**: Good performance with RMSE = 0.33 g/L
        - **Hybrid 1D-CNN+MLP**: Advanced deep learning model with R² = 0.9905
        
        **Best Model**: PLS (5000 samples) - Recommended for production use.
        """)
        
        # Performance comparison
        performance_data = {
            'Model': ['PLS (5000 samples)', 'ElasticNet (2000 samples)', 'Hybrid CNN+MLP (5000 samples)'],
            'RMSE (g/L)': [0.1721, 0.33, 0.9674],
            'R²': [0.9997, 'N/A', 0.9905],
            'Samples': [5000, 2000, 5000]
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # Performance chart
        fig = px.bar(perf_df, x='Model', y='RMSE (g/L)', 
                    title='Model Performance Comparison (Lower RMSE is Better)',
                    color='RMSE (g/L)', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
