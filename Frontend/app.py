import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
import json
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="DeepLense - Fashion Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"
# Correct path: go up from Frontend to DeepLense, then up to root, then to fashion_subset
FASHION_IMAGES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "fashion_subset")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .result-card {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .distance-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
    }
    
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def search_similar_images(uploaded_file) -> List[Dict]:
    """Send image to API and get similar images"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/search/", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()["results"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return []

def display_image_with_info(image_path: str, result: Dict, col):
    """Display image with metadata in a column"""
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            with col:
                st.image(image, use_column_width=True)
                st.markdown(f"""
                <div class="result-card">
                    <h4>Rank #{result['rank']}</h4>
                    <p><strong>File:</strong> {result['file_name']}</p>
                    <p><strong>Label:</strong> {result['label']}</p>
                    <div class="distance-badge">
                        Distance: {result['distance']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col:
                st.error(f"Image not found: {result['file_name']}")
                st.json(result)
    except Exception as e:
        with col:
            st.error(f"Error loading image: {str(e)}")
            st.json(result)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 DeepLense Fashion Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a fashion image to find similar items using deep learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Status
        st.subheader("API Status")
        if check_api_health():
            st.markdown('<div class="success-message">✅ API is running</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ API is not accessible</div>', unsafe_allow_html=True)
            st.warning("Make sure your FastAPI server is running on http://localhost:8000")
        
        # Search parameters
        st.subheader("Search Parameters")
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        # File info
        st.subheader("Supported Formats")
        st.info("📁 JPEG, JPG, PNG")
        
        # About
        st.subheader("About")
        st.info("""
        This app uses:
        - **VGG16** for feature extraction
        - **FAISS** for similarity search
        - **FastAPI** backend
        - **Streamlit** frontend
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fashion image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of clothing or fashion items"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.subheader("🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            # Image info
            st.write(f"**Size:** {image.size}")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**Format:** {image.format}")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Search Results")
            
            # Search button
            if st.button("🚀 Search Similar Images", type="primary", use_container_width=True):
                if not check_api_health():
                    st.error("❌ API is not running. Please start your FastAPI server first.")
                    st.code("cd DeepLense/Backend && python -m uvicorn main:app --reload")
                else:
                    with st.spinner("🔄 Searching for similar images..."):
                        results = search_similar_images(uploaded_file)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} similar images!")
                        
                        # Display results in grid
                        st.subheader("📋 Similar Images")
                        
                        # Create columns for results
                        cols_per_row = 3
                        for i in range(0, min(len(results), num_results), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j in range(cols_per_row):
                                if i + j < len(results) and i + j < num_results:
                                    result = results[i + j]
                                    image_path = os.path.join(FASHION_IMAGES_PATH, result['file_name'])
                                    display_image_with_info(image_path, result, cols[j])
                        
                        # Detailed results table
                        with st.expander("📊 Detailed Results"):
                            import pandas as pd
                            df = pd.DataFrame(results[:num_results])
                            st.dataframe(df, use_container_width=True)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"search_results_{uploaded_file.name}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("❌ No results found or there was an error with the search.")
        else:
            st.info("👈 Please upload an image to start searching")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            Built with ❤️ using Streamlit • FastAPI • PyTorch • FAISS<br>
            <small>DeepLense Fashion Search Engine</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
