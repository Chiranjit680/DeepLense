
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from PIL import Image
import io
import torch
from torchvision import transforms
import os
import sys
import logging

# Add the parent directory to Python path to access DeepSearch modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from DeepSearch.deep_search import FeatureExtractor
from DeepSearch.faiss_index import FaissIndex

# Global variables for model and index (loaded once at startup)
feature_extractor = None
faiss_index = None
device = None
transform = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    global feature_extractor, faiss_index, device, transform
    
    try:
        logger.info("Initializing models and resources...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(model_name='vgg16', pretrained=True)
        feature_extractor.model.to(device)
        logger.info("Feature extractor loaded successfully")
        
        # Initialize FAISS index (now uses the refactored class)
        faiss_index = FaissIndex()  # Will automatically find files in workspace root
        logger.info("FAISS index loaded successfully")
        
        # Define image transform pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        logger.info("Transform pipeline initialized")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e

@app.post("/search/")
async def search(file: UploadFile = File(...)):
    """Search for similar images using uploaded image"""
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Upload JPEG or PNG only."
            )
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor.extract_features_single(image_tensor)
        
        # Search similar images
        results = faiss_index.search(features, k=5)
        
        # Clean up memory
        del image_tensor, features
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"Search completed successfully, found {len(results)} results")
        return {"results": results}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": feature_extractor is not None,
        "index_loaded": faiss_index is not None
    }

