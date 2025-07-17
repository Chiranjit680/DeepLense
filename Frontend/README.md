# DeepLense Frontend

A beautiful Streamlit frontend for the DeepLense fashion image search engine.

## Features

- 🖼️ **Image Upload**: Support for JPEG, JPG, and PNG files
- 🔍 **Visual Search**: Find similar fashion items using deep learning
- 📊 **Results Display**: Grid layout with similarity scores
- 📈 **Health Monitoring**: Real-time API status checking
- 📥 **Export Results**: Download search results as CSV
- 🎨 **Beautiful UI**: Modern, responsive design

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your FastAPI backend is running:
```bash
cd ../Backend
python -m uvicorn main:app --reload
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

## How to Use

1. **Upload Image**: Click "Choose a fashion image..." and select your image
2. **Search**: Click "Search Similar Images" button
3. **View Results**: Browse similar images with similarity scores
4. **Download**: Export results as CSV for further analysis

## API Configuration

The frontend expects the FastAPI backend to be running on `http://localhost:8000`. If you're running the backend on a different port, update the `API_BASE_URL` in `app.py`.

## Features Overview

### Main Interface
- **Left Panel**: Image upload and preview
- **Right Panel**: Search results in grid layout
- **Sidebar**: Settings and API status

### Search Results
- **Grid View**: Up to 10 results in a responsive grid
- **Metadata**: File name, label, and similarity distance
- **Detailed Table**: Expandable table with all result data

### Error Handling
- **API Health Check**: Automatic backend connectivity testing
- **Error Messages**: Clear feedback for various error conditions
- **Graceful Degradation**: Continues working even if some images are missing

## Customization

You can customize the app by modifying:
- `API_BASE_URL`: Change backend URL
- `FASHION_IMAGES_PATH`: Update path to your fashion images
- CSS styles in the `st.markdown()` sections
- Number of columns in results grid (`cols_per_row`)
