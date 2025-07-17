import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
class FashionDataset(Dataset):
    def __init__(self, root_dir, label_file):
        # Set base directory to the parent of DeepSearch folder
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Convert relative paths to absolute paths
        if not os.path.isabs(root_dir):
            self.root_dir = os.path.join(self.base_dir, root_dir)
        else:
            self.root_dir = root_dir
            
        if not os.path.isabs(label_file):
            label_file_path = os.path.join(self.base_dir, label_file)
        else:
            label_file_path = label_file
        
        print(f"Looking for CSV file at: {label_file_path}")
        print(f"Looking for images in: {self.root_dir}")
        
        # Read CSV with error handling for malformed lines
        try:
            # Try with on_bad_lines parameter (pandas >= 1.3.0)
            self.df = pd.read_csv(label_file_path, on_bad_lines='skip', engine='python')
        except TypeError:
            # Fallback for older pandas versions
            try:
                self.df = pd.read_csv(label_file_path, error_bad_lines=False, warn_bad_lines=True, engine='python')
            except:
                # Final fallback - read with basic parameters
                self.df = pd.read_csv(label_file_path, sep=',', quotechar='"', engine='python')
        except:
            # Another fallback: read with different parameters
            self.df = pd.read_csv(label_file_path, sep=',', quotechar='"', engine='python')
        
        self.df['filename'] = self.df['id'].astype(str) + ".jpg"
        
        # Filter to only include items that have corresponding image files
        available_images = set(os.listdir(self.root_dir))
        self.df = self.df[self.df['filename'].isin(available_images)].reset_index(drop=True)
        
        self.labels = self.df['articleType'].astype(str)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path, self.labels.iloc[idx]
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a default black image if file not found
            default_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                default_image = self.transform(default_image)
            return default_image, img_path, self.labels.iloc[idx]

# Global dataset and dataloader - only create when explicitly requested
Fashion_dataset = None
FashionDataLoader = None

def get_dataset():
    """Get or create the global dataset"""
    global Fashion_dataset, FashionDataLoader
    if Fashion_dataset is None:
        initialize_dataset()
    return Fashion_dataset, FashionDataLoader

def initialize_dataset():
    """Initialize the global dataset and dataloader"""
    global Fashion_dataset, FashionDataLoader
    try:
        # Correct paths based on the actual file structure
        # From DeepSearch folder: go up to DeepLense, then up to root, then to content/fashion_subset
        root_dir = os.path.join('..', '..', '..', 'content', 'fashion_subset')
        # From DeepSearch folder: go up to DeepLense for styles.csv
        label_file = os.path.join('..', 'styles.csv')
        
        print(f"Trying to initialize dataset with:")
        print(f"  Root dir: {os.path.abspath(root_dir)}")
        print(f"  Label file: {os.path.abspath(label_file)}")
        
        Fashion_dataset = FashionDataset(root_dir=root_dir, label_file=label_file)
        FashionDataLoader = DataLoader(Fashion_dataset, batch_size=32, shuffle=True)
        print("✅ Dataset initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize global dataset: {e}")
        print("💡 This is normal for API usage - dataset not needed for inference")

class FeatureExtractor:
    def __init__(self, model_name='vgg16', pretrained=True):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'vgg16':
            # Use the newer weights parameter to avoid deprecation warnings
            if pretrained:
                self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            else:
                self.model = models.vgg16(weights=None)
            # Remove the classifier (last layer) and adaptive pooling
            self.extractor = torch.nn.Sequential(*list(self.model.features))
            # Add adaptive pooling to ensure consistent output size
            self.extractor.add_module('adaptive_pool', torch.nn.AdaptiveAvgPool2d((7, 7)))
        elif model_name == 'resnet50':
            # Use the newer weights parameter
            if pretrained:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet50(weights=None)
            # Remove the last fully connected layer
            self.extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.extractor.to(self.device)
        self.extractor.eval()

    def extract_features(self, dataloader):
        file_names = []
        labels = []
        embeddings = []
        with torch.no_grad():
            count = 0
            for images, file_paths, batch_labels in dataloader:
                count += 1
                print(f"Processing batch {count} with {len(images)} images")
                features = self.extractor(images)
                features = features.view(features.size(0), -1)
                embeddings.append(features)
                file_names.extend(file_paths)
                labels.extend(batch_labels)
        return embeddings, file_names, labels
    
    def extract_features_single(self, image_tensor):
        """Extract features from a single image tensor"""
        with torch.no_grad():
            # Ensure image is on the correct device
            if image_tensor.device != self.device:
                image_tensor = image_tensor.to(self.device)
            
            # Extract features
            features = self.extractor(image_tensor)
            features = features.view(features.size(0), -1)
            
            # Return CPU tensor for compatibility with FAISS
            return features.cpu()

    def save_embeddings(self, embeddings, file_names, labels, output_file='embeddings.csv'):
        torch.save({
            'embeddings': embeddings,
            'file_names': file_names,
            'labels': labels
        }, output_file)

def main():
    """Main function to extract and save embeddings"""
    # Get dataset (will initialize if needed)
    dataset, dataloader = get_dataset()
    
    if dataset is None or dataloader is None:
        print("❌ Could not initialize dataset. Please check file paths.")
        return
    
    feature_extractor = FeatureExtractor(model_name='vgg16', pretrained=True)
    embeddings, file_names, labels = feature_extractor.extract_features(dataloader)
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    feature_extractor.save_embeddings(embeddings, file_names, labels, output_file='embeddings.pth')

if __name__ == "__main__":
    main()