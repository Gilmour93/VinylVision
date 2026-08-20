"""
Custom UI widgets for VinylVision application.

Enhanced widgets for camera display, overlay rendering, and user interaction.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import time
import io


class CameraDisplay(tk.Label):
    """Enhanced camera display widget with overlay support."""
    
    def __init__(self, parent, width=640, height=480, **kwargs):
        """
        Initialize camera display widget.
        
        Args:
            parent: Parent widget
            width: Display width in pixels
            height: Display height in pixels
        """
        super().__init__(parent, **kwargs)
        self.display_size = (width, height)
        self.overlay_enabled = True
        self.detection_boxes = []
        self.fps_display = True
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        
        # Configure display
        self.configure(
            bg="black",
            text="Camera Initializing...",
            fg="white",
            font=("Arial", 14),
            width=width,
            height=height
        )
    
    def update_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None):
        """
        Update display with new frame and detection results.
        
        Args:
            frame: OpenCV frame (BGR format)
            detections: List of detection results with bounding boxes
        """
        try:
            # Update FPS counter
            self._update_fps()
            
            # Convert BGR to RGB
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add overlays if enabled
            if self.overlay_enabled:
                display_frame = self._add_overlays(display_frame, detections)
            
            # Convert to PIL Image and resize
            image = Image.fromarray(display_frame)
            image = image.resize(self.display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and update display
            photo = ImageTk.PhotoImage(image)
            self.configure(image=photo, text="")
            self.image = photo  # Keep reference to prevent garbage collection
            
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _add_overlays(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Add overlay graphics to frame.
        
        Args:
            frame: RGB frame
            detections: Detection results
            
        Returns:
            Frame with overlays added
        """
        # Convert to PIL for better drawing capabilities
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        
        # Add detection boxes
        if detections:
            for detection in detections:
                self._draw_detection_box(draw, detection, image.size)
        
        # Add FPS display
        if self.fps_display:
            self._draw_fps(draw, image.size)
        
        # Add center crosshair for aiming
        self._draw_crosshair(draw, image.size)
        
        return np.array(image)
    
    def _draw_detection_box(self, draw: ImageDraw.Draw, detection: Dict[str, Any], image_size: Tuple[int, int]):
        """Draw detection bounding box with confidence."""
        bbox = detection.get('bbox')
        confidence = detection.get('confidence', 0.0)
        
        if not bbox:
            return
        
        # Scale bbox to display size
        x1, y1, x2, y2 = bbox
        scale_x = image_size[0] / detection.get('original_width', image_size[0])
        scale_y = image_size[1] / detection.get('original_height', image_size[1])
        
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        
        # Choose color based on confidence
        if confidence > 0.8:
            color = "green"
        elif confidence > 0.6:
            color = "yellow"
        else:
            color = "red"
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add confidence label
        label = f"Album {confidence:.1%}"
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Calculate text size and position
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Draw background rectangle for text
        text_bg = [x1, y1 - text_height - 5, x1 + text_width + 10, y1]
        draw.rectangle(text_bg, fill=color)
        
        # Draw text
        draw.text((x1 + 5, y1 - text_height - 2), label, fill="white", font=font)
    
    def _draw_fps(self, draw: ImageDraw.Draw, image_size: Tuple[int, int]):
        """Draw FPS counter."""
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Position in top-right corner
        bbox = draw.textbbox((0, 0), fps_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = image_size[0] - text_width - 10
        y = 10
        
        # Draw background
        draw.rectangle([x - 5, y - 2, x + text_width + 5, y + text_height + 2], 
                      fill="black", outline="white")
        
        # Draw text
        draw.text((x, y), fps_text, fill="white", font=font)
    
    def _draw_crosshair(self, draw: ImageDraw.Draw, image_size: Tuple[int, int]):
        """Draw center crosshair for album positioning."""
        center_x = image_size[0] // 2
        center_y = image_size[1] // 2
        size = 20
        
        # Draw crosshair lines
        draw.line([center_x - size, center_y, center_x + size, center_y], 
                 fill="white", width=2)
        draw.line([center_x, center_y - size, center_x, center_y + size], 
                 fill="white", width=2)
        
        # Draw center circle
        draw.ellipse([center_x - 3, center_y - 3, center_x + 3, center_y + 3], 
                    outline="white", width=2)
    
    def set_overlay_enabled(self, enabled: bool):
        """Enable or disable overlay graphics."""
        self.overlay_enabled = enabled
    
    def set_fps_display(self, enabled: bool):
        """Enable or disable FPS display."""
        self.fps_display = enabled


class ConfidenceMeter(ttk.Frame):
    """Animated confidence meter widget."""
    
    def __init__(self, parent, **kwargs):
        """Initialize confidence meter."""
        super().__init__(parent, **kwargs)
        
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.threshold_var = tk.DoubleVar(value=0.8)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the confidence meter UI."""
        # Label
        ttk.Label(self, text="Confidence:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self, 
            length=200, 
            mode='determinate',
            variable=self.confidence_var
        )
        self.progress.pack(fill="x", pady=2)
        
        # Threshold indicator frame
        threshold_frame = ttk.Frame(self)
        threshold_frame.pack(fill="x", pady=2)
        
        # Confidence value label
        self.value_label = ttk.Label(threshold_frame, text="0.0%", font=("Arial", 9))
        self.value_label.pack(side="left")
        
        # Threshold label
        self.threshold_label = ttk.Label(threshold_frame, text="Threshold: 80%", font=("Arial", 9))
        self.threshold_label.pack(side="right")
        
        # Status indicator
        self.status_label = ttk.Label(self, text="●", font=("Arial", 14), foreground="gray")
        self.status_label.pack(pady=2)
    
    def update_confidence(self, confidence: float):
        """Update confidence display."""
        self.confidence_var.set(confidence * 100)
        self.value_label.config(text=f"{confidence:.1%}")
        
        # Update status indicator
        threshold = self.threshold_var.get()
        if confidence >= threshold:
            self.status_label.config(text="●", foreground="green")
        elif confidence >= threshold * 0.7:
            self.status_label.config(text="●", foreground="yellow")
        else:
            self.status_label.config(text="●", foreground="red")
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold."""
        self.threshold_var.set(threshold)
        self.threshold_label.config(text=f"Threshold: {threshold:.0%}")


class AlbumCoverDisplay(tk.Label):
    """Album cover image display widget."""
    
    def __init__(self, parent, size=(150, 150), **kwargs):
        """
        Initialize album cover display.
        
        Args:
            parent: Parent widget
            size: Display size (width, height)
        """
        super().__init__(parent, **kwargs)
        self.size = size
        self.placeholder_image = None
        
        self.configure(
            width=size[0],
            height=size[1],
            bg="lightgray",
            text="No Album",
            compound="center",
            font=("Arial", 10)
        )
        
        # Create placeholder image
        self._create_placeholder()
    
    def _create_placeholder(self):
        """Create placeholder image."""
        placeholder = Image.new('RGB', self.size, color='lightgray')
        draw = ImageDraw.Draw(placeholder)
        
        # Draw vinyl record outline
        center_x, center_y = self.size[0] // 2, self.size[1] // 2
        radius = min(self.size) // 3
        
        # Outer circle
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                    outline='gray', width=2)
        
        # Inner circle (hole)
        inner_radius = radius // 6
        draw.ellipse([center_x - inner_radius, center_y - inner_radius,
                     center_x + inner_radius, center_y + inner_radius],
                    outline='gray', width=2)
        
        self.placeholder_image = ImageTk.PhotoImage(placeholder)
        self.configure(image=self.placeholder_image)
    
    def update_cover(self, image_data: bytes = None, url: str = None):
        """
        Update album cover display.
        
        Args:
            image_data: Raw image data
            url: Image URL (for future implementation)
        """
        try:
            if image_data:
                # Load image from data
                image = Image.open(io.BytesIO(image_data))
                image = image.resize(self.size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.configure(image=photo, text="")
                self.image = photo  # Keep reference
            else:
                # Show placeholder
                self.configure(image=self.placeholder_image, text="No Album")
                
        except Exception as e:
            print(f"Error updating album cover: {e}")
            self.configure(image=self.placeholder_image, text="Error Loading")


class StatusIndicator(ttk.Frame):
    """System status indicator with multiple states."""
    
    def __init__(self, parent, **kwargs):
        """Initialize status indicator."""
        super().__init__(parent, **kwargs)
        
        self.states = {
            'disconnected': {'color': 'red', 'text': 'Disconnected'},
            'connecting': {'color': 'yellow', 'text': 'Connecting...'},
            'ready': {'color': 'green', 'text': 'Ready'},
            'processing': {'color': 'blue', 'text': 'Processing'},
            'error': {'color': 'red', 'text': 'Error'}
        }
        
        self.current_state = 'disconnected'
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup status indicator UI."""
        self.indicator = ttk.Label(self, text="●", font=("Arial", 12))
        self.indicator.pack(side="left", padx=(0, 5))
        
        self.status_label = ttk.Label(self, text="Disconnected", font=("Arial", 9))
        self.status_label.pack(side="left")
        
        self.set_state('disconnected')
    
    def set_state(self, state: str, custom_text: str = None):
        """
        Set indicator state.
        
        Args:
            state: State name from predefined states
            custom_text: Optional custom status text
        """
        if state not in self.states:
            return
        
        self.current_state = state
        state_info = self.states[state]
        
        self.indicator.config(foreground=state_info['color'])
        self.status_label.config(text=custom_text or state_info['text'])


class PerformanceMonitor(ttk.LabelFrame):
    """Real-time performance monitoring widget."""
    
    def __init__(self, parent, **kwargs):
        """Initialize performance monitor."""
        super().__init__(parent, text="Performance", padding="5", **kwargs)
        
        self.metrics = {
            'fps': 0.0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'queue_size': 0
        }
        
        self._setup_ui()
        self._start_monitoring()
    
    def _setup_ui(self):
        """Setup performance monitor UI."""
        # FPS
        ttk.Label(self, text="FPS:", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
        self.fps_label = ttk.Label(self, text="0.0", font=("Arial", 9, "bold"))
        self.fps_label.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        # Processing time
        ttk.Label(self, text="Process:", font=("Arial", 9)).grid(row=1, column=0, sticky="w")
        self.process_label = ttk.Label(self, text="0ms", font=("Arial", 9, "bold"))
        self.process_label.grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        # Memory usage
        ttk.Label(self, text="Memory:", font=("Arial", 9)).grid(row=2, column=0, sticky="w")
        self.memory_label = ttk.Label(self, text="0MB", font=("Arial", 9, "bold"))
        self.memory_label.grid(row=2, column=1, sticky="w", padx=(5, 0))
        
        # Queue size
        ttk.Label(self, text="Queue:", font=("Arial", 9)).grid(row=3, column=0, sticky="w")
        self.queue_label = ttk.Label(self, text="0", font=("Arial", 9, "bold"))
        self.queue_label.grid(row=3, column=1, sticky="w", padx=(5, 0))
    
    def _start_monitoring(self):
        """Start performance monitoring."""
        self._update_display()
        self.after(1000, self._start_monitoring)  # Update every second
    
    def _update_display(self):
        """Update performance display."""
        try:
            import psutil
            
            # Update memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics['memory_usage'] = memory_mb
            
        except ImportError:
            pass
        
        # Update display labels
        self.fps_label.config(text=f"{self.metrics['fps']:.1f}")
        self.process_label.config(text=f"{self.metrics['processing_time']:.0f}ms")
        self.memory_label.config(text=f"{self.metrics['memory_usage']:.0f}MB")
        self.queue_label.config(text=str(self.metrics['queue_size']))
    
    def update_metrics(self, **kwargs):
        """Update performance metrics."""
        self.metrics.update(kwargs)


# Export all widgets
__all__ = [
    'CameraDisplay',
    'ConfidenceMeter', 
    'AlbumCoverDisplay',
    'StatusIndicator',
    'PerformanceMonitor'
]
