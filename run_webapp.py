#!/usr/bin/env python3
"""
Script to run the Streamlit web application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to import names
    package_imports = {
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # scikit-learn is imported as sklearn
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements_webapp.txt")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        "web_app.py",
        "app/models/preprocessing_pipeline.pkl",
        "app/models/elasticnet_penicillin.pkl",
        "test_data/test_samples.csv"  # Updated to match what web_app.py actually uses
    ]
    
    # Optional files (warn but don't fail)
    optional_files = [
        "app/models/pls_penicillin.pkl",
        "app/models/best_sqrt_hybrid_5000_samples.h5"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check optional files and warn if missing
    missing_optional = []
    for file_path in optional_files:
        if not os.path.exists(file_path):
            missing_optional.append(file_path)
    
    if missing_optional:
        print("⚠️  Optional files not found (app will work but some features may be limited):")
        for file_path in missing_optional:
            print(f"   - {file_path}")
        print("   (This is okay - the app will use available models)")
    
    return True

def main():
    """Main function to run the web app"""
    print("🚀 Starting Penicillin Concentration Prediction Web App...")
    print("📦 Using indpensim conda environment")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check required files
    if not check_files():
        print("\nPlease ensure all required files are present.")
        sys.exit(1)
    
    print("✅ All dependencies and files found!")
    print("\n🌐 Starting Streamlit web application...")
    print("📡 The app will be available at: http://localhost:8501")
    print("📚 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web application stopped.")
    except Exception as e:
        print(f"❌ Error starting web application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
