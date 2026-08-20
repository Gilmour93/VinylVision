"""
EfficientNet model loading and inference for album cover feature extraction.
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
        self.model_name = model_name
        self.model = None
        self.device = self._get_device(device)
        self.transform = self._create_transform()
        self.is_loaded = False
        
    def _get_device(self, device_preference: str) -> torch.device:
        if device_preference == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Using CUDA GPU for inference")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU for inference")
        else:
            device = torch.device(device_preference)
        return device
    
    def _create_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self) -> bool:
        try:
            logger.info(f"Loading {self.model_name} model...")
            self.model = EfficientNet.from_pretrained(self.model_name)
            
            # Estrae direttamente il feature vector da 1280 dimensioni senza layer casuali
            self.model._fc = nn.Identity()
            self.model.eval()
            self.model.to(self.device)
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def extract_features(self, image: np.ndarray, is_bgr: bool = True) -> Optional[np.ndarray]:
        """
        Extract features from album cover image.
        """
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Converte in RGB solo se l'input è dichiarato BGR (es. OpenCV)
            if is_bgr:
                image_rgb = image[:, :, ::-1]
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
                features_np = features.cpu().numpy().flatten().astype(np.float32)
                
                # Normalizzazione L2 dell'embedding
                norm = np.linalg.norm(features_np)
                if norm > 0:
                    features_np = features_np / norm
                
                return features_np
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def get_model_info(self) -> dict:
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'feature_dim': 1280
        }