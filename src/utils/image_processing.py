"""
Image preprocessing utilities for VinylVision.

Common image processing functions for album cover preparation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from loguru import logger


def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int],
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        np.ndarray: Resized image
    """
    try:
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image if needed
            if new_w != target_w or new_h != target_h:
                # Create black background
                padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                
                # Calculate padding offsets
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                # Place resized image in center
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return padded
            else:
                return resized
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        np.ndarray: Normalized image
    """
    try:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        return image / 255.0
        
    except Exception as e:
        logger.error(f"Error normalizing image: {e}")
        return image


def apply_noise_reduction(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Apply noise reduction to image.
    
    Args:
        image: Input image
        method: Noise reduction method ('bilateral', 'gaussian', 'median')
        
    Returns:
        np.ndarray: Denoised image
    """
    try:
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            return image
            
    except Exception as e:
        logger.error(f"Error applying noise reduction: {e}")
        return image


def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        np.ndarray: Enhanced image
    """
    try:
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing contrast: {e}")
        return image

def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordina 4 punti in sequenza: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left ha la somma x+y minima
    rect[2] = pts[np.argmax(s)]  # bottom-right ha la somma x+y massima

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right ha la diff y-x minima
    rect[3] = pts[np.argmax(diff)]  # bottom-left ha la diff y-x massima
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray, target_dim: int = 500) -> Optional[np.ndarray]:
    """
    Raddrizza prospetticamente un quadrilatero trasformandolo in un quadrato perfetto (1:1).
    Risolve il problema della telecamera angolata o posizionata più in basso.
    """
    try:
        rect = order_points(pts)
        dst = np.array([
            [0, 0],
            [target_dim - 1, 0],
            [target_dim - 1, target_dim - 1],
            [0, target_dim - 1]
        ], dtype="float32")

        # Calcola la matrice di trasformazione prospettica e applicala
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (target_dim, target_dim))
        return warped
    except Exception as e:
        logger.error(f"Errore warp prospettico: {e}")
        return None

def find_vinyl_quadrilateral(frame: np.ndarray, min_area_ratio: float = 0.05) -> Optional[np.ndarray]:
    """
    Rileva il quadrilatero del vinile anche con prospettiva dal basso.
    """
    try:
        h, w = frame.shape[:2]
        total_area = h * w
        
        # Preprocessing: scala di grigi e blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Rilevamento bordi combinato (Canny + Morfologia)
        edged = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < total_area * min_area_ratio:
                continue
                
            peri = cv2.arcLength(c, True)
            # Tolleranza del poligono al 4% del perimetro
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                return pts
                
        return None
    except Exception as e:
        logger.error(f"Errore auto-detection: {e}")
        return None


def correct_perspective(image: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    """
    Correct perspective distortion using four corner points.
    
    Args:
        image: Input image
        corners: Four corner points in clockwise order
        
    Returns:
        Optional[np.ndarray]: Perspective-corrected image or None if correction fails
    """
    try:
        if corners.shape != (4, 2):
            logger.error("Corners must be 4 points with 2 coordinates each")
            return None
        
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = corners.astype(np.float32)
        
        # Calculate width and height of corrected image
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))
        
        # Define destination points for rectangle
        dst_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # Apply perspective correction
        corrected = cv2.warpPerspective(image, transform_matrix, (width, height))
        
        return corrected
        
    except Exception as e:
        logger.error(f"Error correcting perspective: {e}")
        return None


def crop_to_square(image: np.ndarray, center: bool = True) -> np.ndarray:
    """
    Crop image to square aspect ratio.
    
    Args:
        image: Input image
        center: Whether to crop from center or top-left
        
    Returns:
        np.ndarray: Square-cropped image
    """
    try:
        h, w = image.shape[:2]
        size = min(h, w)
        
        if center:
            y_start = (h - size) // 2
            x_start = (w - size) // 2
        else:
            y_start = 0
            x_start = 0
        
        return image[y_start:y_start+size, x_start:x_start+size]
        
    except Exception as e:
        logger.error(f"Error cropping to square: {e}")
        return image


def adjust_lighting(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Adjust image lighting using gamma correction.
    
    Args:
        image: Input image
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        
    Returns:
        np.ndarray: Gamma-corrected image
    """
    try:
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction
        return cv2.LUT(image, table)
        
    except Exception as e:
        logger.error(f"Error adjusting lighting: {e}")
        return image


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Detect if image is blurry using Laplacian variance.
    
    Args:
        image: Input image
        threshold: Blur threshold (lower values indicate more blur)
        
    Returns:
        bool: True if image is blurry, False otherwise
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var < threshold
        
    except Exception as e:
        logger.error(f"Error detecting blur: {e}")
        return False


def preprocess_for_model(image: np.ndarray, 
                        target_size: Tuple[int, int] = (224, 224),
                        enhance: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for model input.
    
    Args:
        image: Input image
        target_size: Target size for model input
        enhance: Whether to apply enhancement
        
    Returns:
        np.ndarray: Preprocessed image ready for model
    """
    try:
        # Start with original image
        processed = image.copy()
        
        # Apply enhancements if requested
        if enhance:
            # Reduce noise
            processed = apply_noise_reduction(processed, method='bilateral')
            
            # Enhance contrast slightly
            processed = enhance_contrast(processed, alpha=1.1, beta=5)
        
        # Resize to target size
        processed = resize_image(processed, target_size, maintain_aspect_ratio=True)
        
        return processed
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        return image
