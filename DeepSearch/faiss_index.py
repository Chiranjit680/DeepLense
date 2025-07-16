
import pandas as pd
import os
import torch
import faiss
from PIL import Image
import numpy as np
import pickle

# Fix for OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class FaissIndex:
    def __init__(self, embeddings_file='embeddings.pth', index_file='faiss_index.bin'):
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        self.index = None
        self.file_names = []
        self.labels = []
        self.embeddings = None
        
        # Set up file paths relative to workspace root
        self.workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.embeddings_path = os.path.join(self.workspace_root, self.embeddings_file)
        self.index_path = os.path.join(self.workspace_root, self.index_file)
        
        # Load embeddings data
        self.load_embeddings()
        
        # Try to load existing index, otherwise create new one
        if not self.load_index():
            self.create_index()
    
    def load_embeddings(self):
        """Load embeddings and metadata from the saved file"""
        print(f"Loading embeddings from: {self.embeddings_path}")
        
        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found at: {self.embeddings_path}")
        
        try:
            # Try to load PyTorch format first
            data = torch.load(self.embeddings_path, map_location='cpu')
            self.embeddings = data['embeddings']
            self.file_names = data['file_names']
            self.labels = data['labels']
            print(f"Loaded PyTorch embeddings: {self.embeddings.shape}")
            
        except:
            # Try numpy format as fallback
            try:
                data = np.load(self.embeddings_path)
                self.embeddings = data['embeddings']
                self.file_names = data['file_names'].tolist()
                self.labels = data['labels'].tolist()
                print(f"Loaded NumPy embeddings: {self.embeddings.shape}")
            except Exception as e:
                raise ValueError(f"Could not load embeddings file: {e}")
        
        # Ensure embeddings are numpy float32 for FAISS
        if isinstance(self.embeddings, torch.Tensor):
            self.embeddings = self.embeddings.numpy()
        self.embeddings = self.embeddings.astype(np.float32)
        
        print(f"Loaded {len(self.embeddings)} embeddings with {len(self.file_names)} filenames and {len(self.labels)} labels")
    
    def create_index(self):
        """Create a new FAISS index and add embeddings"""
        print("Creating new FAISS index...")
        
        # Create L2 distance index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        # Save the index and metadata
        self.save_index()
        print(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save FAISS index and metadata to files"""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata (file names and labels) separately
        metadata = {
            'file_names': self.file_names,
            'labels': self.labels
        }
        
        metadata_file = self.index_path.replace('.bin', '_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved FAISS index to {self.index_path}")
        print(f"Saved metadata to {metadata_file}")

    def load_index(self):
        """Load existing FAISS index and metadata"""
        if not os.path.exists(self.index_path):
            print(f"Index file {self.index_path} does not exist.")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            metadata_file = self.index_path.replace('.bin', '_metadata.pkl')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.file_names = metadata['file_names']
                    self.labels = metadata['labels']
            
            print(f"Loaded FAISS index from {self.index_path} with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def search(self, query_vector, k=10):
        """Search for similar embeddings"""
        if self.index is None:
            raise ValueError("Index not loaded. Please create the index first.")
        
        # Ensure query vector is the right format
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.numpy()
        query_vector = np.array(query_vector, dtype=np.float32)
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Return results with metadata
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.file_names):  # Valid index
                results.append({
                    'rank': i + 1,
                    'distance': float(distance),
                    'file_name': self.file_names[idx],
                    'label': self.labels[idx],
                    'index': int(idx)
                })
        
        return results
    
    def get_embedding_by_filename(self, filename):
        """Get embedding vector for a specific filename"""
        try:
            idx = self.file_names.index(filename)
            return self.embeddings[idx]
        except ValueError:
            raise ValueError(f"Filename {filename} not found in index")
    
    def get_metadata_by_index(self, idx):
        """Get metadata for a specific index"""
        if 0 <= idx < len(self.file_names):
            return {
                'file_name': self.file_names[idx],
                'label': self.labels[idx]
            }
        else:
            raise IndexError(f"Index {idx} out of range")

# Example usage
if __name__ == "__main__":
    # Create FAISS index from embeddings (will auto-find the file in workspace root)
    faiss_index = FaissIndex()
    
    # Alternative: specify different filenames if needed
    # faiss_index = FaissIndex('my_embeddings.pth', 'my_index.bin')
    
    # Example: Search using the first embedding as query
    if len(faiss_index.embeddings) > 0:
        query_vector = faiss_index.embeddings[0]
        results = faiss_index.search(query_vector, k=5)
        
        print("\nSearch Results:")
        for result in results:
            print(f"Rank {result['rank']}: {result['file_name']} (Label: {result['label']}, Distance: {result['distance']:.4f})")
