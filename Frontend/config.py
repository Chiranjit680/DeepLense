"""
Configuration file for DeepLense Streamlit Frontend
"""

import os

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30  # seconds

# Paths
BASE_DIR = os.path.dirname(__file__)
# Correct path: go up from Frontend to DeepLense, then up to root, then to fashion_subset
FASHION_IMAGES_PATH = os.path.join(BASE_DIR, "..", "..", "fashion_subset")

# UI Configuration
DEFAULT_NUM_RESULTS = 5
MAX_RESULTS = 10
RESULTS_PER_ROW = 3

# Supported file types
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']
SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]

# Styling
PRIMARY_COLOR = "#FF6B6B"
SUCCESS_COLOR = "#4CAF50"
ERROR_COLOR = "#f44336"

# Page configuration
PAGE_CONFIG = {
    "page_title": "DeepLense - Fashion Image Search",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
