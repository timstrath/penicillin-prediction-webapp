# Model Building Studio - Local Development App

## Overview
Professional model development interface for pharmaceutical analytics, designed for local development with full dataset support.

## Features

### 📊 Data Analysis
- **Full Dataset Support**: Handle large datasets (5000+ samples) locally
- **Batch Analysis**: Process data visualization and statistics
- **Process Variables**: Correlation analysis with target variables
- **Spectral Data**: Raman spectroscopy data visualization

### 🤖 Model Training
- **PLS Regression**: Partial Least Squares with configurable components
- **ElasticNet Regression**: Regularized linear regression with L1/L2 penalties
- **Hyperparameter Tuning**: Interactive parameter adjustment
- **Cross-Validation**: 5-fold CV with performance metrics

### 📈 Model Validation
- **Performance Metrics**: R², RMSE, MAE, Cross-validation scores
- **Prediction Plots**: Training vs Test performance visualization
- **Validation Reports**: Detailed model assessment
- **Acceptance Criteria**: Pharmaceutical industry standards

### 💾 Model Export
- **Local Storage**: Save models and scalers locally
- **Model Registry Integration**: Export to GMP-compliant registry
- **Version Control**: Track model versions and performance
- **Documentation**: Automatic model documentation

## Installation

1. **Activate Environment**:
   ```bash
   conda activate indpensim
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements_model_building.txt
   ```

3. **Run Application**:
   ```bash
   streamlit run model_building_app.py
   ```

## Usage

### 1. Data Analysis Tab
- View batch-by-batch process data
- Analyze correlations with target variables
- Visualize process trends over time
- Select specific batch ranges for analysis

### 2. Model Training Tab
- Choose between PLS or ElasticNet models
- Adjust hyperparameters interactively
- Set training/test split ratios
- Monitor training progress

### 3. Model Validation Tab
- Review comprehensive performance metrics
- Analyze training vs test performance
- Check cross-validation stability
- Validate against acceptance criteria

### 4. Model Export Tab
- Save trained models locally
- Export to model registry
- Generate model documentation
- Track model versions

## Model Types

### PLS Regression
- **Use Case**: High-dimensional spectral data
- **Advantages**: Handles multicollinearity, dimension reduction
- **Parameters**: Number of components (1-20)
- **Best For**: Raman spectroscopy, NIR data

### ElasticNet Regression
- **Use Case**: Feature selection and regularization
- **Advantages**: Combines L1 and L2 penalties
- **Parameters**: Alpha (regularization), L1 ratio
- **Best For**: Process variables, feature selection

## Performance Metrics

### Primary Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

### Validation Metrics
- **Cross-Validation**: 5-fold CV with mean and std
- **Training vs Test**: Overfitting detection
- **Prediction Plots**: Visual performance assessment

## Model Registry Integration

### GMP Compliance
- **Audit Trails**: Complete model lifecycle tracking
- **Version Control**: Model versioning and rollback
- **Documentation**: Automatic model documentation
- **Validation**: Performance validation tracking

### Export Process
1. Train model locally
2. Validate performance
3. Export to registry
4. Track deployment
5. Monitor performance

## File Structure

```
model_building_app.py          # Main application
requirements_model_building.txt # Dependencies
README_Model_Building.md       # Documentation
models/                        # Local model storage
├── pls_regression_*.pkl      # PLS models
├── elasticnet_regression_*.pkl # ElasticNet models
└── scaler_*.pkl              # Feature scalers
```

## Best Practices

### Data Preprocessing
- **RangeCut**: Apply 350-1750 cm⁻¹ for Raman data
- **Standardization**: Scale features before training
- **Missing Values**: Handle NaN values appropriately
- **Feature Selection**: Use relevant spectral ranges

### Model Training
- **Train-Test Split**: Use 80-20 split with fixed random state
- **Cross-Validation**: Always use CV for robust evaluation
- **Hyperparameter Tuning**: Test multiple parameter combinations
- **Performance Monitoring**: Track training vs test performance

### Model Validation
- **Acceptance Criteria**: R² > 0.7, RMSE < 2.0 g/L
- **Overfitting Check**: Training vs test performance gap
- **Stability Check**: Cross-validation consistency
- **Documentation**: Record all validation results

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size or use data sampling
2. **Slow Training**: Use fewer features or simpler models
3. **Poor Performance**: Check data quality and preprocessing
4. **Export Errors**: Ensure model registry is properly initialized

### Performance Optimization
- **Data Sampling**: Use representative subsets for development
- **Feature Selection**: Remove irrelevant features
- **Model Complexity**: Start with simpler models
- **Caching**: Use Streamlit caching for data loading

## Integration with Cloud App

### Workflow
1. **Local Development**: Use this app for model development
2. **Model Training**: Train and validate models locally
3. **Model Export**: Export validated models to registry
4. **Cloud Deployment**: Deploy models to cloud app
5. **Production Use**: Use cloud app for predictions

### Benefits
- **Full Dataset Access**: No size limitations locally
- **Fast Development**: Interactive parameter tuning
- **Professional Interface**: Modern web-based UI
- **GMP Compliance**: Industry-standard model management

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the model registry documentation
3. Ensure all dependencies are installed
4. Verify data file paths and formats

## License

This application is designed for pharmaceutical industry use and follows GMP guidelines for model development and validation.
