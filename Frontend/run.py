#!/usr/bin/env python3
"""
Launcher script for DeepLense Frontend
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_api_health(url="http://localhost:8000/health", timeout=5):
    """Check if the API is running"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        print("⚠️ Requirements file not found!")

def main():
    print("🚀 Starting DeepLense Frontend...")
    
    # Check if API is running
    if not check_api_health():
        print("⚠️ Warning: FastAPI backend is not running!")
        print("💡 To start the backend, run:")
        print("   cd ../Backend && python -m uvicorn main:app --reload")
        print()
        
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Exiting...")
            return
    else:
        print("✅ FastAPI backend is running!")
    
    # Install requirements if needed
    try:
        import streamlit
        import requests
        import PIL
        import pandas
    except ImportError:
        print("📦 Installing missing dependencies...")
        install_requirements()
    
    # Launch Streamlit
    print("🎯 Launching Streamlit app...")
    app_file = Path(__file__).parent / "app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

if __name__ == "__main__":
    main()
