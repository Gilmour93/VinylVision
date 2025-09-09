"""
Computer vision pipeline for album cover recognition.

Handles feature extraction, album detection, and image processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class AlbumDetector:
    """Detects album covers in camera frames."""
    
    def __init__(self, min_area: int = 10000):
        """
        Initialize album detector.
        
        Args:
            min_area: Minimum area for album detection (pixels)
        """
        self.min_area = min_area
        
    def detect_albums(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect rectangular album covers in frame.
        
        Args:
            frame: Input camera frame
            
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            albums = []
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (albums are roughly square)
                    aspect_ratio = w / h
                    if 0.7 <= aspect_ratio <= 1.4:
                        albums.append((x, y, w, h))
            
            return albums
            
        except Exception as e:
            logger.error(f"Error detecting albums: {e}")
            return []
    
    def extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract region of interest from frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Optional[np.ndarray]: Extracted ROI or None if invalid
        """
        try:
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                return None
                
            roi = frame[y:y+h, x:x+w]
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            return None


class FeatureExtractor:
    """Deep learning feature extraction using EfficientNet."""
    
    def __init__(self):
        """Initialize feature extractor."""
        from ..models.efficientnet import AlbumFeatureExtractor
        self.model = AlbumFeatureExtractor()
        self.is_loaded = False
        
    def initialize(self) -> bool:
        """
        Initialize and load the EfficientNet model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            success = self.model.load_model()
            self.is_loaded = success
            if success:
                logger.info("Feature extraction model loaded successfully")
            else:
                logger.error("Failed to load feature extraction model")
            return success
        except Exception as e:
            logger.error(f"Error initializing feature extractor: {e}")
            return False
        
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from album cover image.
        
        Args:
            image: Album cover image (RGB format)
            
        Returns:
            Optional[np.ndarray]: 512-dimensional feature vector or None if extraction fails
        """
        if not self.is_loaded:
            logger.warning("Feature extractor not initialized. Call initialize() first.")
            return None
            
        try:
            # Extract features using EfficientNet
            features = self.model.extract_features(image)
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if self.is_loaded:
            return self.model.get_model_info()
        else:
            return {'is_loaded': False, 'error': 'Model not initialized'}
