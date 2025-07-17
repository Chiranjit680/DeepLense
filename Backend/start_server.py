#!/usr/bin/env python3
"""
Startup script for DeepLense Backend
Handles Python path configuration and starts the FastAPI server
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_python_path():
    """Add necessary directories to Python path"""
    # Get the directory structure
    backend_dir = Path(__file__).parent
    deeplense_dir = backend_dir.parent
    
    # Add DeepLense directory to Python path for module imports
    sys.path.insert(0, str(deeplense_dir))
    
    print(f"Backend directory: {backend_dir}")
    print(f"DeepLense directory: {deeplense_dir}")
    print(f"Python path updated: {deeplense_dir} added")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'torchvision', 
        'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed")
    return True

def start_server(host="localhost", port=8000, reload=True):
    """Start the FastAPI server with uvicorn"""
    setup_python_path()
    
    if not check_dependencies():
        return
    
    print(f"🚀 Starting DeepLense Backend Server...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"📖 API Docs: http://{host}:{port}/docs")
    print(f"🔄 Reload: {reload}")
    print("-" * 50)
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd, cwd=Path(__file__).parent)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start DeepLense Backend Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
