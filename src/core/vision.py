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
    
    def __init__(self, min_area: int = 2000):
        """
        Initialize album detector.
        
        Args:
            min_area: Minimum area for album detection (pixels)
        """
        self.min_area = min_area
        
    def detect_albums(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect rectangular album covers in frame using multiple detection strategies.
        
        Args:
            frame: Input camera frame
            
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h)
        """
        try:
            albums = []
            h, w = frame.shape[:2]
            
            # Strategy 1: Center region detection (most reliable for user-held albums)
            center_x, center_y = w // 2, h // 2
            sizes = [min(w, h) // 3, min(w, h) // 4, min(w, h) // 5]
            
            for size in sizes:
                x = max(0, center_x - size // 2)
                y = max(0, center_y - size // 2)
                w_box = min(size, w - x)
                h_box = min(size, h - y)
                
                if w_box * h_box >= self.min_area:
                    albums.append((x, y, w_box, h_box))
            
            # Strategy 2: Color variance detection
            albums.extend(self._detect_by_color_variance(frame))
            
            # Strategy 3: Edge density detection  
            albums.extend(self._detect_by_edge_density(frame))
            
            # Strategy 4: Traditional contour detection (as backup)
            albums.extend(self._detect_by_contours(frame))
            
            # Remove duplicates and overlapping detections
            albums = self._remove_overlapping_detections(albums)
            
            # Sort by area (largest first) and return top candidates
            albums_with_area = [(x, y, w, h, w*h) for x, y, w, h in albums]
            albums_with_area.sort(key=lambda x: x[4], reverse=True)
            
            return [(x, y, w, h) for x, y, w, h, _ in albums_with_area[:5]]
            
        except Exception as e:
            logger.error(f"Error detecting albums: {e}")
            return []
    
    def _detect_by_color_variance(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions with high color variance."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            h, w = frame.shape[:2]
            boxes = []
            
            step = 60
            size = 200
            
            for y in range(0, h - size, step):
                for x in range(0, w - size, step):
                    roi = lab[y:y+size, x:x+size]
                    std_dev = np.std(roi)
                    
                    if std_dev > 25:  # Threshold for color variance
                        boxes.append((x, y, size, size))
            
            return boxes[:3]
        except:
            return []
    
    def _detect_by_edge_density(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions with high edge density."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 80)
            h, w = frame.shape[:2]
            boxes = []
            
            step = 50
            size = 180
            
            for y in range(0, h - size, step):
                for x in range(0, w - size, step):
                    roi = edges[y:y+size, x:x+size]
                    edge_count = np.sum(roi > 0)
                    
                    if edge_count > 400:  # Threshold for edge density
                        boxes.append((x, y, size, size))
            
            return boxes[:4]
        except:
            return []
    
    def _detect_by_contours(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Traditional contour-based detection (backup method)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 20, 80)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            albums = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.3 <= aspect_ratio <= 3.0:  # Very flexible aspect ratio
                    albums.append((x, y, w, h))
            
            return albums
        except:
            return []
    
    def _remove_overlapping_detections(self, albums: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping detections."""
        if not albums:
            return []
        
        unique_albums = []
        for x1, y1, w1, h1 in albums:
            overlaps = False
            for x2, y2, w2, h2 in unique_albums:
                # Check overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                min_area = min(w1 * h1, w2 * h2)
                if overlap_area > 0.3 * min_area:  # 30% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                unique_albums.append((x1, y1, w1, h1))
        
        return unique_albums
    
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
