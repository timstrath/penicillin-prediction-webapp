#!/usr/bin/env python3
"""
Script to run the Penicillin Concentration Prediction API from the app directory
"""

import uvicorn
import os
import sys

def check_models():
    """Check if required model files exist"""
    # Go up one level to find models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    required_files = [
        os.path.join(models_dir, "preprocessing_pipeline.pkl"),
        os.path.join(models_dir, "elasticnet_penicillin.pkl")
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure you have trained and saved your models first.")
        print("Run your notebook to create the model files.")
        return False
    
    print("✅ All required model files found!")
    return True

def main():
    """Main function to run the API"""
    print("🚀 Starting Penicillin Concentration Prediction API...")
    print("📦 Using indpensim conda environment")
    print("📁 Running from app directory")
    
    # Check if models exist
    if not check_models():
        sys.exit(1)
    
    # Configuration
    host = "0.0.0.0"
    port = 8000
    reload = True  # Set to False for production
    
    print(f"📡 API will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the API from the app directory
        uvicorn.run(
            "main:app",  # main.py in current directory
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 API server stopped.")
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
