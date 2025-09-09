"""
Camera capture and processing module for VinylVision.

Handles video capture, frame processing, and camera management.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class CameraManager:
    """Manages camera capture and frame processing."""
    
    def __init__(self, camera_id: int = 0, target_fps: int = 3):
        """
        Initialize camera manager.
        
        Args:
            camera_id: Camera device ID (default: 0)
            target_fps: Target frames per second for processing (default: 3)
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        
    def initialize(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_active = True
            logger.info(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success status and frame data
        """
        if not self.is_active or self.cap is None:
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                # Basic preprocessing
                frame = self._preprocess_frame(frame)
            return ret, frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply basic preprocessing to frame.
        
        Args:
            frame: Raw camera frame
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        # Resize if needed
        height, width = frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280/width, 720/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Apply noise reduction
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        return frame
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_active = False
            logger.info("Camera released")
