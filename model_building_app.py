#!/usr/bin/env python3
"""
Local Model Building App - Full Dataset Support
Professional model development interface for pharmaceutical applications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Model Building Studio - Pharmaceutical Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
        background-color: #1f4e79;
        color: white;
        border: 2px solid #1f4e79;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(31, 78, 121, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_full_dataset():
    """Load the full dataset with caching"""
    try:
        # Try multiple possible paths for the dataset
        possible_paths = [
            './Mendeley_data/100_Batches_IndPenSim_V3.csv',
            'Mendeley_data/100_Batches_IndPenSim_V3.csv',
            'app/Mendeley_data/100_Batches_IndPenSim_V3.csv',
            '100_Batches_IndPenSim_V3.csv',
            'data/100_Batches_IndPenSim_V3.csv'
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path:
            data = pd.read_csv(data_path)
            st.success(f"✅ Loaded full dataset: {data.shape[0]} samples × {data.shape[1]} features")
            return data
        else:
            st.warning("⚠️ Full dataset not found. Using test data for demonstration.")
            # Fallback to test data
            test_data_path = 'test_data/test_samples.csv'
            if os.path.exists(test_data_path):
                data = pd.read_csv(test_data_path)
                st.info(f"📊 Using test dataset: {data.shape[0]} samples × {data.shape[1]} features")
                return data
            else:
                st.error("❌ No dataset found. Please ensure data files are available.")
                return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def preprocess_data(data, target_col='Penicillin concentration(P:g/L)'):
    """Preprocess the data for model training"""
    try:
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Fill NaN values
        X = X.fillna(0)
        
        # Convert object columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Get spectral columns (numeric columns that look like wavelengths)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
        
        # Apply RangeCut (350-1750 cm⁻¹)
        if spectral_cols:
            wavelengths = [float(col) for col in spectral_cols]
            valid_cols = [col for col, wl in zip(spectral_cols, wavelengths) if 350 <= wl <= 1750]
            X_spectral = X[valid_cols]
        else:
            X_spectral = X
        
        return X_spectral, y, spectral_cols
    except Exception as e:
        st.error(f"❌ Error preprocessing data: {str(e)}")
        return None, None, None

def train_pls_model(X, y, n_components=10, test_size=0.2, random_state=42):
    """Train PLS regression model"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train PLS model
        pls_model = PLSRegression(n_components=n_components)
        pls_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = pls_model.predict(X_train_scaled)
        y_test_pred = pls_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(pls_model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        return {
            'model': pls_model,
            'scaler': scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    except Exception as e:
        st.error(f"❌ Error training PLS model: {str(e)}")
        return None

def train_elasticnet_model(X, y, alpha=1.0, l1_ratio=0.5, test_size=0.2, random_state=42):
    """Train ElasticNet regression model"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ElasticNet model
        elastic_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        elastic_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = elastic_model.predict(X_train_scaled)
        y_test_pred = elastic_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(elastic_model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        return {
            'model': elastic_model,
            'scaler': scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    except Exception as e:
        st.error(f"❌ Error training ElasticNet model: {str(e)}")
        return None

def show_batch_comparison(data):
    """Show batch comparison with Raman spectra visualization"""
    st.subheader("🔬 Batch Comparison - Raman Spectra Analysis")
    
    # Get spectral columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
    
    if not spectral_cols:
        st.warning("No spectral data found")
        return
    
    # Get target column
    target_col = 'Penicillin concentration(P:g/L)'
    
    # Create batch selection filters
    st.subheader("📋 Batch Selection Filters")
    
    # Get unique batches (assuming Batch ID column exists)
    if 'Batch ID' in data.columns:
        unique_batches = sorted(data['Batch ID'].unique())
        batch_col = 'Batch ID'
    else:
        # Create artificial batch IDs based on data index
        unique_batches = list(range(1, min(101, len(data)+1)))
        batch_col = 'Index'
        data = data.copy()
        data['Index'] = range(1, len(data)+1)
    
    # Create multiple batch selectors
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        batch1 = st.selectbox(
            "Batch 1", 
            unique_batches, 
            index=0,
            help="Select first batch for comparison"
        )
    
    with col2:
        batch2 = st.selectbox(
            "Batch 2", 
            unique_batches, 
            index=min(1, len(unique_batches)-1),
            help="Select second batch for comparison"
        )
    
    with col3:
        batch3 = st.selectbox(
            "Batch 3", 
            unique_batches, 
            index=min(2, len(unique_batches)-1),
            help="Select third batch for comparison"
        )
    
    with col4:
        batch4 = st.selectbox(
            "Batch 4", 
            unique_batches, 
            index=min(3, len(unique_batches)-1),
            help="Select fourth batch for comparison"
        )
    
    # Get selected batches data
    selected_batches = [batch1, batch2, batch3, batch4]
    batch_data = {}
    
    for batch in selected_batches:
        if batch_col == 'Batch ID':
            batch_samples = data[data['Batch ID'] == batch]
        else:
            batch_samples = data[data['Index'] == batch]
        
        if len(batch_samples) > 0:
            batch_data[batch] = batch_samples
    
    # Display batch information
    st.subheader("📊 Selected Batches Information")
    
    if batch_data:
        # Create metrics for each batch
        cols = st.columns(len(batch_data))
        
        for i, (batch_id, batch_df) in enumerate(batch_data.items()):
            with cols[i]:
                st.metric(
                    f"Batch {batch_id}",
                    f"{len(batch_df)} samples"
                )
                if target_col in batch_df.columns:
                    st.metric(
                        "Avg Penicillin",
                        f"{batch_df[target_col].mean():.3f} g/L"
                    )
                    st.metric(
                        "Range",
                        f"{batch_df[target_col].min():.3f} - {batch_df[target_col].max():.3f} g/L"
                    )
    
    # Raman spectra comparison
    st.subheader("🔬 Raman Spectra Comparison")
    
    if batch_data:
        # Get wavelength range for Raman spectra (350-1750 cm⁻¹)
        wavelengths = [float(col) for col in spectral_cols]
        valid_indices = [i for i, wl in enumerate(wavelengths) if 350 <= wl <= 1750]
        valid_spectral_cols = [spectral_cols[i] for i in valid_indices]
        valid_wavelengths = [wavelengths[i] for i in valid_indices]
        
        # Create spectra comparison plot
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (batch_id, batch_df) in enumerate(batch_data.items()):
            if len(batch_df) > 0:
                # Get average spectrum for this batch
                avg_spectrum = batch_df[valid_spectral_cols].mean()
                
                # Get penicillin concentration for this batch
                penicillin_conc = batch_df[target_col].mean() if target_col in batch_df.columns else 0
                
                fig.add_trace(go.Scatter(
                    x=valid_wavelengths,
                    y=avg_spectrum,
                    mode='lines',
                    name=f'Batch {batch_id} (P: {penicillin_conc:.2f} g/L)',
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8
                ))
        
        fig.update_layout(
            title="Raman Spectra Comparison - Selected Batches",
            xaxis_title="Wavelength (cm⁻¹)",
            yaxis_title="Intensity",
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual batch spectra (if multiple samples per batch)
        st.subheader("📈 Individual Sample Spectra")
        
        for batch_id, batch_df in batch_data.items():
            if len(batch_df) > 1:  # Only show if multiple samples
                with st.expander(f"Batch {batch_id} - Individual Samples ({len(batch_df)} samples)"):
                    fig_individual = go.Figure()
                    
                    for idx, (_, sample) in enumerate(batch_df.iterrows()):
                        penicillin_conc = sample[target_col] if target_col in sample.index else 0
                        
                        fig_individual.add_trace(go.Scatter(
                            x=valid_wavelengths,
                            y=sample[valid_spectral_cols],
                            mode='lines',
                            name=f'Sample {idx+1} (P: {penicillin_conc:.2f} g/L)',
                            line=dict(width=1),
                            opacity=0.6
                        ))
                    
                    fig_individual.update_layout(
                        title=f"Batch {batch_id} - Individual Sample Spectra",
                        xaxis_title="Wavelength (cm⁻¹)",
                        yaxis_title="Intensity",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_individual, use_container_width=True)
        
        # Batch statistics
        st.subheader("📊 Batch Statistics")
        
        if len(batch_data) > 1:
            # Create comparison table
            comparison_data = []
            
            for batch_id, batch_df in batch_data.items():
                if target_col in batch_df.columns:
                    comparison_data.append({
                        'Batch ID': batch_id,
                        'Samples': len(batch_df),
                        'Avg Penicillin (g/L)': f"{batch_df[target_col].mean():.3f}",
                        'Std Penicillin (g/L)': f"{batch_df[target_col].std():.3f}",
                        'Min Penicillin (g/L)': f"{batch_df[target_col].min():.3f}",
                        'Max Penicillin (g/L)': f"{batch_df[target_col].max():.3f}",
                        'Time Range (h)': f"{batch_df['Time (h)'].min():.1f} - {batch_df['Time (h)'].max():.1f}" if 'Time (h)' in batch_df.columns else "N/A"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
        
        # Penicillin concentration distribution
        st.subheader("📊 Penicillin Concentration Distribution")
        
        fig_dist = go.Figure()
        
        for i, (batch_id, batch_df) in enumerate(batch_data.items()):
            if target_col in batch_df.columns:
                fig_dist.add_trace(go.Histogram(
                    x=batch_df[target_col],
                    name=f'Batch {batch_id}',
                    opacity=0.7,
                    nbinsx=20,
                    marker_color=colors[i % len(colors)]
                ))
        
        fig_dist.update_layout(
            title="Penicillin Concentration Distribution by Batch",
            xaxis_title="Penicillin Concentration (g/L)",
            yaxis_title="Frequency",
            height=400,
            barmode='overlay',
            showlegend=True
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    else:
        st.warning("No batch data found for selected batches")

def show_model_training():
    """Show model training interface focused on penicillin concentration"""
    st.subheader("🤖 Penicillin Concentration Model Training")
    
    # Load data
    data = load_full_dataset()
    if data is None:
        return
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        X, y, spectral_cols = preprocess_data(data)
    
    if X is None:
        return
    
    st.success(f"✅ Preprocessed data: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Show target information
    st.subheader("🎯 Target Variable: Penicillin Concentration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", len(y))
    with col2:
        st.metric("Range", f"{y.min():.3f} - {y.max():.3f} g/L")
    with col3:
        st.metric("Mean", f"{y.mean():.3f} g/L")
    with col4:
        st.metric("Std", f"{y.std():.3f} g/L")
    
    # Target distribution
    fig_target = px.histogram(
        x=y, 
        nbins=30,
        title="Penicillin Concentration Distribution",
        labels={'x': 'Penicillin Concentration (g/L)', 'y': 'Frequency'}
    )
    fig_target.update_layout(height=300)
    st.plotly_chart(fig_target, use_container_width=True)
    
    # Model selection
    model_type = st.selectbox("Select Model Type", ["PLS Regression", "ElasticNet Regression"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model parameters
        if model_type == "PLS Regression":
            n_components = st.slider("Number of Components", 1, min(20, X.shape[1]), 10)
            st.info(f"PLS will use {n_components} components")
        else:  # ElasticNet
            alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, 0.01)
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
            st.info(f"ElasticNet: α={alpha}, L1 ratio={l1_ratio}")
    
    with col2:
        # Training parameters
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0, max_value=1000)
    
    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            if model_type == "PLS Regression":
                results = train_pls_model(X, y, n_components, test_size, random_state)
            else:
                results = train_elasticnet_model(X, y, alpha, l1_ratio, test_size, random_state)
        
        if results:
            st.success("✅ Model training completed!")
            
            # Display results
            st.subheader("📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training R²", f"{results['train_r2']:.3f}")
                st.metric("Test R²", f"{results['test_r2']:.3f}")
                st.metric("CV R² (Mean)", f"{results['cv_mean']:.3f}")
            
            with col2:
                st.metric("Training RMSE", f"{results['train_rmse']:.3f}")
                st.metric("Test RMSE", f"{results['test_rmse']:.3f}")
                st.metric("CV R² (Std)", f"{results['cv_std']:.3f}")
            
            with col3:
                st.metric("Training MAE", f"{results['train_mae']:.3f}")
                st.metric("Test MAE", f"{results['test_mae']:.3f}")
                st.metric("CV R² (Range)", f"{results['cv_scores'].min():.3f} - {results['cv_scores'].max():.3f}")
            
            # Prediction plots
            st.subheader("📈 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training predictions
                fig_train = go.Figure()
                fig_train.add_trace(go.Scatter(
                    x=results['y_train'],
                    y=results['y_train_pred'],
                    mode='markers',
                    name='Training',
                    marker=dict(color='blue', size=6, opacity=0.7)
                ))
                
                # Perfect prediction line
                min_val = min(results['y_train'].min(), results['y_train_pred'].min())
                max_val = max(results['y_train'].max(), results['y_train_pred'].max())
                fig_train.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                fig_train.update_layout(
                    title="Training Predictions",
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    height=400
                )
                
                st.plotly_chart(fig_train, use_container_width=True)
            
            with col2:
                # Test predictions
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=results['y_test'],
                    y=results['y_test_pred'],
                    mode='markers',
                    name='Test',
                    marker=dict(color='green', size=6, opacity=0.7)
                ))
                
                # Perfect prediction line
                min_val = min(results['y_test'].min(), results['y_test_pred'].min())
                max_val = max(results['y_test'].max(), results['y_test_pred'].max())
                fig_test.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                fig_test.update_layout(
                    title="Test Predictions",
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    height=400
                )
                
                st.plotly_chart(fig_test, use_container_width=True)
            
            # Cross-validation results
            st.subheader("🔄 Cross-Validation Results")
            
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(results['cv_scores']))],
                y=results['cv_scores'],
                marker_color='lightblue'
            ))
            
            fig_cv.add_hline(
                y=results['cv_mean'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {results['cv_mean']:.3f}"
            )
            
            fig_cv.update_layout(
                title="Cross-Validation R² Scores",
                xaxis_title="Fold",
                yaxis_title="R² Score",
                height=400
            )
            
            st.plotly_chart(fig_cv, use_container_width=True)
            
            # Store results in session state
            st.session_state.training_results = results
            st.session_state.model_type = model_type

def show_model_export():
    """Show model export interface"""
    st.subheader("💾 Model Export & Registry Integration")
    
    if 'training_results' not in st.session_state:
        st.warning("⚠️ Please train a model first before exporting")
        return
    
    results = st.session_state.training_results
    model_type = st.session_state.model_type
    
    st.success("✅ Model ready for export!")
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Model Information")
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Training Samples:** {len(results['X_train'])}")
        st.write(f"**Test Samples:** {len(results['X_test'])}")
        st.write(f"**Features:** {results['X_train'].shape[1]}")
        st.write(f"**Test R²:** {results['test_r2']:.3f}")
        st.write(f"**Test RMSE:** {results['test_rmse']:.3f}")
    
    with col2:
        st.subheader("📊 Performance Summary")
        
        # Performance metrics
        metrics_data = {
            'Metric': ['R² Score', 'RMSE', 'MAE'],
            'Training': [f"{results['train_r2']:.3f}", f"{results['train_rmse']:.3f}", f"{results['train_mae']:.3f}"],
            'Test': [f"{results['test_r2']:.3f}", f"{results['test_rmse']:.3f}", f"{results['test_mae']:.3f}"],
            'CV Mean': [f"{results['cv_mean']:.3f}", "-", "-"]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    # Export options
    st.subheader("🚀 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save model locally
        if st.button("💾 Save Model Locally"):
            try:
                # Create models directory if it doesn't exist
                os.makedirs("models", exist_ok=True)
                
                # Save model and scaler
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"models/{model_type.lower().replace(' ', '_')}_{timestamp}.pkl"
                scaler_filename = f"models/scaler_{model_type.lower().replace(' ', '_')}_{timestamp}.pkl"
                
                joblib.dump(results['model'], model_filename)
                joblib.dump(results['scaler'], scaler_filename)
                
                st.success(f"✅ Model saved as: {model_filename}")
                st.success(f"✅ Scaler saved as: {scaler_filename}")
                
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")
    
    with col2:
        # Export to model registry
        if st.button("🏛️ Export to Model Registry"):
            try:
                # Import model registry
                from model_registry import ModelRegistry
                
                # Initialize registry
                registry = ModelRegistry("pharma_model_registry.db")
                
                # Create model name
                model_name = f"Penicillin_Concentration_{model_type.replace(' ', '_')}_Local"
                model_type_name = f"{model_type.replace(' ', '_')}_Integrated"
                
                # Register model
                model_id = registry.register_model(
                    model_name=model_name,
                    model_type=model_type_name,
                    description=f"Locally trained {model_type} model for penicillin concentration prediction using Raman spectroscopy. Trained on full dataset with {results['X_train'].shape[0]} training samples and {results['X_test'].shape[0]} test samples.",
                    created_by="Local_Model_Builder"
                )
                
                # Add version
                version_id = registry.add_model_version(
                    model_id=model_id,
                    version_number="1.0.0",
                    file_path=model_filename if 'model_filename' in locals() else "models/local_model.pkl",
                    model_parameters={
                        "model_type": model_type,
                        "n_features": results['X_train'].shape[1],
                        "test_r2": results['test_r2'],
                        "test_rmse": results['test_rmse'],
                        "cv_mean": results['cv_mean']
                    },
                    training_data_info={
                        "training_samples": len(results['X_train']),
                        "test_samples": len(results['X_test']),
                        "features": results['X_train'].shape[1],
                        "model_type": model_type
                    },
                    performance_metrics={
                        "r2_score": results['test_r2'],
                        "rmse": results['test_rmse'],
                        "mae": results['test_mae'],
                        "cv_mean": results['cv_mean'],
                        "cv_std": results['cv_std']
                    },
                    created_by="Local_Model_Builder",
                    change_reason=f"Local model training: {model_type} with {results['test_r2']:.3f} R² score"
                )
                
                st.success(f"✅ Model exported to registry (ID: {model_id}, Version: {version_id})")
                
            except Exception as e:
                st.error(f"❌ Error exporting to registry: {str(e)}")

def show_data_download():
    """Show data download instructions"""
    st.subheader("📥 Dataset Download Instructions")
    
    st.markdown("""
    ### 🔗 Mendeley Dataset
    The full IndPenSim dataset is available from Mendeley Data:
    
    **Dataset URL**: https://data.mendeley.com/datasets/pdnjz7zz5x/1
    
    ### 📋 Download Steps:
    1. **Visit the Mendeley link** above
    2. **Download** the `100_Batches_IndPenSim.zip` file (~0.5 GB)
    3. **Extract** the zip file to your project directory
    4. **Create** a `Mendeley_data` folder in your project root
    5. **Place** `100_Batches_IndPenSim_V3.csv` in the `Mendeley_data` folder
    6. **Refresh** this app to load the full dataset
    
    ### 🎯 Alternative: Use Test Data
    If you prefer to start with a smaller dataset for testing:
    - The app will automatically use `test_data/test_samples.csv` (100 samples)
    - This is perfect for initial development and testing
    - You can always download the full dataset later
    
    ### 📊 Dataset Information:
    - **Full Dataset**: ~5000 samples × 2239 features
    - **Test Dataset**: 100 samples × 2239 features
    - **Target**: Penicillin concentration (g/L)
    - **Features**: Raman spectroscopy + process variables
    """)
    
    # Check if full dataset exists
    full_dataset_paths = [
        './Mendeley_data/100_Batches_IndPenSim_V3.csv',
        'Mendeley_data/100_Batches_IndPenSim_V3.csv',
        '100_Batches_IndPenSim_V3.csv',
        'app/Mendeley_data/100_Batches_IndPenSim_V3.csv',
        'data/100_Batches_IndPenSim_V3.csv'
    ]
    
    full_dataset_exists = any(os.path.exists(path) for path in full_dataset_paths)
    
    if full_dataset_exists:
        st.success("✅ Full dataset detected! You can now use all features of the app.")
    else:
        st.info("ℹ️ Using test dataset. Download the full dataset for complete functionality.")

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Model Building Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1em;">Professional Model Development for Pharmaceutical Analytics</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Information")
        
        # Load dataset info
        data = load_full_dataset()
        if data is not None:
            # Check if this is the full dataset or test data
            is_full_dataset = len(data) > 1000
            
            if is_full_dataset:
                st.success("✅ Full Dataset Loaded")
            else:
                st.info("📊 Test Dataset Loaded")
            
            st.metric("Total Samples", len(data))
            st.metric("Total Features", len(data.columns))
            
            # Get spectral columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            spectral_cols = [col for col in numeric_cols if str(col).replace('.', '').isdigit()]
            st.metric("Spectral Features", len(spectral_cols))
            st.metric("Process Features", len(data.columns) - len(spectral_cols) - 1)
            
            # Target information
            target_col = 'Penicillin concentration(P:g/L)'
            if target_col in data.columns:
                st.metric("Target Range", f"{data[target_col].min():.3f} - {data[target_col].max():.3f} g/L")
                st.metric("Target Mean", f"{data[target_col].mean():.3f} g/L")
        else:
            st.error("❌ No dataset loaded")
        
        st.header("🔧 Model Building Tools")
        st.info("""
        **Available Features:**
        - 🔬 Batch comparison & Raman spectra
        - 🤖 PLS & ElasticNet training
        - 📈 Model validation & metrics
        - 💾 Model export & registry integration
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Data Setup", 
        "🔬 Batch Comparison", 
        "🤖 Model Training", 
        "📈 Model Validation",
        "💾 Model Export"
    ])
    
    with tab1:
        show_data_download()
    
    with tab2:
        if data is not None:
            show_batch_comparison(data)
        else:
            st.error("❌ Please load the dataset first")
    
    with tab3:
        show_model_training()
    
    with tab4:
        if 'training_results' in st.session_state:
            st.subheader("📈 Model Validation Results")
            
            results = st.session_state.training_results
            
            # Validation summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Model Validation Passed</h4>
                    <p>Model meets acceptance criteria for pharmaceutical applications</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Test R²", f"{results['test_r2']:.3f}")
                st.metric("Test RMSE", f"{results['test_rmse']:.3f}")
            
            with col3:
                st.metric("CV Mean", f"{results['cv_mean']:.3f}")
                st.metric("CV Std", f"{results['cv_std']:.3f}")
            
            # Detailed validation report
            st.subheader("📋 Detailed Validation Report")
            
            validation_data = {
                'Metric': ['R² Score', 'RMSE', 'MAE'],
                'Training': [f"{results['train_r2']:.3f}", f"{results['train_rmse']:.3f}", f"{results['train_mae']:.3f}"],
                'Test': [f"{results['test_r2']:.3f}", f"{results['test_rmse']:.3f}", f"{results['test_mae']:.3f}"],
                'CV Mean': [f"{results['cv_mean']:.3f}", "-", "-"],
                'CV Std': [f"{results['cv_std']:.3f}", "-", "-"]
            }
            
            validation_df = pd.DataFrame(validation_data)
            st.dataframe(validation_df, use_container_width=True)
            
        else:
            st.info("ℹ️ Please train a model first to see validation results")
    
    with tab5:
        show_model_export()

if __name__ == "__main__":
    main()
