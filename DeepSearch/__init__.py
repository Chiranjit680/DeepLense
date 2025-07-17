# DeepSearch Package
# Fashion Image Search using Deep Learning

from .deep_search import FeatureExtractor, FashionDataset, initialize_dataset
from .faiss_index import FaissIndex

__version__ = "1.0.0"
__author__ = "DeepLense Team"

__all__ = ["FeatureExtractor", "FashionDataset", "FaissIndex", "initialize_dataset"]
