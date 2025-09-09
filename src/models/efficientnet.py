"""
EfficientNet model loading and inference for album cover feature extraction.

Handles model initialization, preprocessing, and feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from loguru import logger


class AlbumFeatureExtractor:
    """EfficientNet-B0 based feature extractor for album covers."""
    
    def __init__(self, model_name: str = 'efficientnet-b0', device: str = 'auto'):
        """
        Initialize feature extractor.
        
        Args:
            model_name: EfficientNet model variant to use
            device: Device to run model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.model = None
        self.device = self._get_device(device)
        self.transform = self._create_transform()
        self.is_loaded = False
        
    def _get_device(self, device_preference: str) -> torch.device:
        """
        Determine the best device to use.
        
        Args:
            device_preference: User device preference
            
        Returns:
            torch.device: Selected device
        """
        if device_preference == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Using CUDA GPU for inference")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using Metal Performance Shaders (MPS) for inference")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU for inference")
        else:
            device = torch.device(device_preference)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _create_transform(self) -> transforms.Compose:
        """
        Create image preprocessing transform.
        
        Returns:
            transforms.Compose: Image preprocessing pipeline
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self) -> bool:
        """
        Load pretrained EfficientNet model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading {self.model_name} model...")
            
            # Load pretrained EfficientNet
            self.model = EfficientNet.from_pretrained(self.model_name)
            
            # Remove classifier head to get features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from album cover image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Optional[np.ndarray]: 512-dimensional feature vector or None if extraction fails
        """
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]  # BGR to RGB
            else:
                logger.error("Invalid image format")
                return None
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Apply preprocessing
            input_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                
                # Global average pooling to get fixed-size features
                features = torch.mean(features, dim=[2, 3])  # Remove spatial dimensions
                
                # Ensure we get 512-dimensional features
                if features.shape[1] != 1280:  # EfficientNet-B0 outputs 1280 features
                    # Add a linear layer to reduce to 512 dimensions
                    if not hasattr(self, 'feature_reducer'):
                        self.feature_reducer = nn.Linear(1280, 512).to(self.device)
                        self.feature_reducer.eval()
                    features = self.feature_reducer(features)
                
                # Convert to numpy
                features_np = features.cpu().numpy().flatten()
                
                # Normalize features
                features_np = features_np / np.linalg.norm(features_np)
                
                return features_np
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'feature_dim': 512
        }
        
        if self.is_loaded and self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            info['total_parameters'] = total_params
        
        return info
    
    def benchmark_inference(self, image_size: Tuple[int, int] = (224, 224)) -> dict:
        """
        Benchmark model inference speed.
        
        Args:
            image_size: Input image size for benchmarking
            
        Returns:
            dict: Benchmark results
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            # Create dummy input
            dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                self.extract_features(dummy_image)
            
            # Benchmark
            import time
            num_runs = 50
            start_time = time.time()
            
            for _ in range(num_runs):
                self.extract_features(dummy_image)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            return {
                'average_inference_time_ms': avg_time * 1000,
                'fps': 1.0 / avg_time,
                'num_runs': num_runs
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return {'error': str(e)}
