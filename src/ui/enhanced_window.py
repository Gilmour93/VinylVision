"""
Enhanced VinylVision Main Window

Improved UI with custom widgets, better performance monitoring,
and enhanced user experience.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
import json
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from loguru import logger

try:
    # Try relative imports first (when used as package)
    from .widgets import CameraDisplay, ConfidenceMeter, AlbumCoverDisplay, StatusIndicator, PerformanceMonitor
    from ..core.camera import CameraManager
    from ..core.vision import AlbumDetector
    from ..core.album_pipeline import AlbumDataPipeline
    from ..models.efficientnet import AlbumFeatureExtractor
    from ..utils.config import load_config
except ImportError:
    # Fallback to absolute imports (when used directly)
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent))
    
    from ui.widgets import CameraDisplay, ConfidenceMeter, AlbumCoverDisplay, StatusIndicator, PerformanceMonitor
    from core.camera import CameraManager
    from core.vision import AlbumDetector
    from core.album_pipeline import AlbumDataPipeline
    from models.efficientnet import AlbumFeatureExtractor
    from utils.config import load_config


@dataclass
class RecognitionResult:
    """Container for album recognition results."""
    artist: str
    title: str
    year: str
    confidence: float
    cover_url: Optional[str] = None
    genre: Optional[str] = None
    label: Optional[str] = None
    discogs_id: Optional[str] = None
    processing_time: float = 0.0


class EnhancedVinylVisionWindow:
    """Enhanced main window with custom widgets and improved UX."""
    
    def __init__(self):
        """Initialize the enhanced main window."""
        self.root = tk.Tk()
        self.root.title("VinylVision - Real-time Album Recognition")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Application state
        self.running = False
        self.paused = False
        
        # Backend components
        self.camera_manager: Optional[CameraManager] = None
        self.album_detector: Optional[AlbumDetector] = None
        self.album_pipeline: Optional[AlbumDataPipeline] = None
        self.feature_extractor: Optional[AlbumFeatureExtractor] = None
        
        # Threading
        self.video_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Performance tracking
        self.performance_stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'start_time': None
        }
        
        # Settings
        self.settings = {
            'confidence_threshold': 0.8,
            'frame_rate': 2.0,
            'camera_source': 0,
            'detection_enabled': True,
            'overlay_enabled': True,
            'auto_save_results': False
        }
        
        self._initialize_components()
        self._setup_ui()
        self._load_configuration()
    
    def _initialize_components(self):
        """Initialize UI components."""
        # Custom widgets will be created in _setup_ui
        self.camera_display: Optional[CameraDisplay] = None
        self.confidence_meter: Optional[ConfidenceMeter] = None
        self.album_cover: Optional[AlbumCoverDisplay] = None
        self.status_indicator: Optional[StatusIndicator] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Result tracking
        self.current_result: Optional[RecognitionResult] = None
        self.result_history: List[RecognitionResult] = []
    
    def _setup_ui(self):
        """Setup the enhanced user interface."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        self._create_toolbar(main_frame)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Camera and controls
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self._create_camera_panel(left_panel)
        self._create_control_panel(left_panel)
        
        # Right panel - Results and settings
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        
        self._create_results_panel(right_panel)
        self._create_settings_panel(right_panel)
        self._create_performance_panel(right_panel)
        
        # Bottom status bar
        self._create_status_bar(main_frame)
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind('<F11>', lambda e: self._toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
    
    def _create_toolbar(self, parent):
        """Create application toolbar."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Main controls
        self.start_button = ttk.Button(
            toolbar, text="▶ Start", 
            command=self._start_capture,
            style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_button = ttk.Button(
            toolbar, text="⏸ Pause",
            command=self._pause_capture,
            state="disabled"
        )
        self.pause_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(
            toolbar, text="⏹ Stop",
            command=self._stop_capture,
            state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Status indicator
        self.status_indicator = StatusIndicator(toolbar)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 15))
        
        # Spacer
        ttk.Frame(toolbar).pack(side=tk.LEFT, expand=True)
        
        # Settings and help
        ttk.Button(
            toolbar, text="⚙ Settings",
            command=self._show_advanced_settings
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            toolbar, text="? Help",
            command=self._show_help
        ).pack(side=tk.RIGHT, padx=(5, 0))
    
    def _create_camera_panel(self, parent):
        """Create camera display panel."""
        camera_frame = ttk.LabelFrame(parent, text="Camera Feed", padding="10")
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced camera display
        self.camera_display = CameraDisplay(
            camera_frame, 
            width=800, 
            height=600,
            bg="black"
        )
        self.camera_display.pack(expand=True)
    
    def _create_control_panel(self, parent):
        """Create camera control panel."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Quick settings
        quick_frame = ttk.Frame(control_frame)
        quick_frame.pack(fill=tk.X)
        
        # Detection toggle
        self.detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            quick_frame, text="Enable Detection",
            variable=self.detection_var,
            command=self._toggle_detection
        ).pack(side=tk.LEFT)
        
        # Overlay toggle
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            quick_frame, text="Show Overlay",
            variable=self.overlay_var,
            command=self._toggle_overlay
        ).pack(side=tk.LEFT, padx=(20, 0))
        
        # Frame rate control
        ttk.Label(quick_frame, text="FPS:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.fps_var = tk.DoubleVar(value=2.0)
        fps_scale = ttk.Scale(
            quick_frame, from_=0.5, to=5.0,
            variable=self.fps_var,
            command=self._on_fps_change,
            length=100
        )
        fps_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        self.fps_label = ttk.Label(quick_frame, text="2.0")
        self.fps_label.pack(side=tk.LEFT)
    
    def _create_results_panel(self, parent):
        """Create results display panel."""
        results_frame = ttk.LabelFrame(parent, text="Recognition Results", padding="10")
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Album cover display
        self.album_cover = AlbumCoverDisplay(results_frame, size=(120, 120))
        self.album_cover.pack(pady=(0, 10))
        
        # Confidence meter
        self.confidence_meter = ConfidenceMeter(results_frame)
        self.confidence_meter.pack(fill=tk.X, pady=(0, 10))
        
        # Metadata display
        metadata_frame = ttk.Frame(results_frame)
        metadata_frame.pack(fill=tk.X)
        
        # Artist and title
        self.artist_var = tk.StringVar(value="—")
        self.title_var = tk.StringVar(value="—")
        self.year_var = tk.StringVar(value="—")
        
        ttk.Label(metadata_frame, text="Artist:", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
        ttk.Label(metadata_frame, textvariable=self.artist_var, font=("Arial", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(metadata_frame, text="Title:", font=("Arial", 9)).grid(row=1, column=0, sticky="w")
        ttk.Label(metadata_frame, textvariable=self.title_var, font=("Arial", 9)).grid(
            row=1, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(metadata_frame, text="Year:", font=("Arial", 9)).grid(row=2, column=0, sticky="w")
        ttk.Label(metadata_frame, textvariable=self.year_var, font=("Arial", 9)).grid(
            row=2, column=1, sticky="w", padx=(5, 0))
        
        metadata_frame.grid_columnconfigure(1, weight=1)
        
        # Action buttons
        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            action_frame, text="Save Result",
            command=self._save_current_result,
            width=12
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            action_frame, text="View Details",
            command=self._view_result_details,
            width=12
        ).pack(side=tk.RIGHT)
    
    def _create_settings_panel(self, parent):
        """Create settings panel."""
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:", font=("Arial", 9)).pack(anchor="w")
        
        confidence_frame = ttk.Frame(settings_frame)
        confidence_frame.pack(fill=tk.X, pady=(2, 10))
        
        self.confidence_var = tk.DoubleVar(value=0.8)
        confidence_scale = ttk.Scale(
            confidence_frame, from_=0.5, to=1.0,
            variable=self.confidence_var,
            command=self._on_confidence_change
        )
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.confidence_label = ttk.Label(confidence_frame, text="80%", width=5)
        self.confidence_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Camera selection
        ttk.Label(settings_frame, text="Camera Source:", font=("Arial", 9)).pack(anchor="w")
        
        camera_frame = ttk.Frame(settings_frame)
        camera_frame.pack(fill=tk.X, pady=(2, 0))
        
        self.camera_var = tk.IntVar(value=0)
        camera_combo = ttk.Combobox(
            camera_frame, 
            textvariable=self.camera_var,
            values=[0, 1, 2],
            state="readonly",
            width=10
        )
        camera_combo.pack(side=tk.LEFT)
        
        ttk.Button(
            camera_frame, text="Refresh",
            command=self._refresh_cameras,
            width=8
        ).pack(side=tk.RIGHT)
    
    def _create_performance_panel(self, parent):
        """Create performance monitoring panel."""
        self.performance_monitor = PerformanceMonitor(parent)
        self.performance_monitor.pack(fill=tk.X, pady=(0, 10))
    
    def _create_status_bar(self, parent):
        """Create status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        
        # Connection indicator
        self.connection_label = ttk.Label(status_frame, text="●", foreground="red")
        self.connection_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(status_frame, text="Discogs:", font=("Arial", 8)).pack(side=tk.RIGHT)
    
    def _load_configuration(self):
        """Load application configuration."""
        try:
            config = load_config()
            
            # Initialize backend components
            self.camera_manager = CameraManager()
            self.album_detector = AlbumDetector()
            self.feature_extractor = AlbumFeatureExtractor()
            
            # Initialize album pipeline
            if config.discogs.consumer_key and config.discogs.consumer_secret:
                self.album_pipeline = AlbumDataPipeline(
                    config.discogs.consumer_key,
                    config.discogs.consumer_secret
                )
                self.connection_label.config(foreground="green")
                self.status_indicator.set_state('ready')
            else:
                self.connection_label.config(foreground="red")
                self.status_indicator.set_state('error', 'No Discogs credentials')
            
            self.status_var.set("Configuration loaded successfully")
            logger.info("Enhanced VinylVision configuration loaded")
            
        except Exception as e:
            error_msg = f"Configuration error: {str(e)}"
            self.status_var.set(error_msg)
            self.status_indicator.set_state('error', 'Config failed')
            logger.error(error_msg)
    
    # Event handlers
    def _start_capture(self):
        """Start video capture and processing."""
        if self.running:
            return
        
        try:
            self.status_indicator.set_state('connecting')
            
            # Initialize camera
            camera_source = self.camera_var.get()
            self.camera_manager.camera_id = camera_source
            if not self.camera_manager.initialize():
                raise Exception(f"Failed to initialize camera {camera_source}")
            
            self.running = True
            self.paused = False
            self.performance_stats['start_time'] = time.time()
            
            # Update UI
            self.start_button.config(state="disabled")
            self.pause_button.config(state="normal")
            self.stop_button.config(state="normal")
            
            # Start threads
            self.video_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
            self.video_thread.start()
            
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start UI updates
            self._update_ui()
            
            self.status_var.set("Camera active - Detection enabled")
            self.status_indicator.set_state('ready')
            logger.info("Enhanced capture started")
            
        except Exception as e:
            error_msg = f"Failed to start camera: {str(e)}"
            self.status_var.set(error_msg)
            self.status_indicator.set_state('error')
            logger.error(error_msg)
            messagebox.showerror("Camera Error", error_msg)
    
    def _pause_capture(self):
        """Pause/resume capture."""
        self.paused = not self.paused
        
        if self.paused:
            self.pause_button.config(text="▶ Resume")
            self.status_var.set("Capture paused")
            self.status_indicator.set_state('ready', 'Paused')
        else:
            self.pause_button.config(text="⏸ Pause")
            self.status_var.set("Capture resumed")
            self.status_indicator.set_state('ready')
    
    def _stop_capture(self):
        """Stop video capture."""
        self.running = False
        self.paused = False
        
        # Wait for threads
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Release resources
        if self.camera_manager:
            self.camera_manager.release()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update UI
        self.start_button.config(state="normal")
        self.pause_button.config(state="disabled", text="⏸ Pause")
        self.stop_button.config(state="disabled")
        
        self.camera_display.configure(image="", text="Camera stopped")
        self.status_var.set("Camera stopped")
        self.status_indicator.set_state('disconnected')
        
        logger.info("Enhanced capture stopped")
    
    def _video_capture_loop(self):
        """Enhanced video capture loop."""
        frame_time = 1.0 / self.fps_var.get()
        
        while self.running:
            if not self.paused:
                try:
                    ret, frame = self.camera_manager.read_frame()
                    if ret and frame is not None:
                        frame_data = {
                            'frame': frame,
                            'timestamp': time.time()
                        }
                        
                        try:
                            self.frame_queue.put_nowait(frame_data)
                        except queue.Full:
                            # Drop oldest frame
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame_data)
                            except queue.Empty:
                                pass
                
                except Exception as e:
                    logger.error(f"Video capture error: {e}")
            
            time.sleep(frame_time)
    
    def _processing_loop(self):
        """Enhanced processing loop."""
        while self.running:
            if not self.paused and self.detection_var.get():
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                    start_time = time.time()
                    
                    result = self._process_frame(frame_data['frame'])
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.performance_stats['frames_processed'] += 1
                    self.performance_stats['total_processing_time'] += processing_time
                    
                    # Update performance monitor
                    if self.performance_monitor:
                        fps = self.performance_stats['frames_processed'] / max(1, time.time() - self.performance_stats['start_time'])
                        self.performance_monitor.update_metrics(
                            fps=fps,
                            processing_time=processing_time,
                            queue_size=self.frame_queue.qsize()
                        )
                    
                    if result:
                        result.processing_time = processing_time
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait(result)
                            except queue.Empty:
                                pass
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Processing error: {e}")
            else:
                time.sleep(0.1)
    
    def _process_frame(self, frame) -> Optional[RecognitionResult]:
        """Enhanced frame processing with better error handling."""
        try:
            # Detect albums
            album_bboxes = self.album_detector.detect_albums(frame)
            if not album_bboxes:
                return None
            
            # Get largest detection (first one returned)
            bbox = album_bboxes[0]  # (x, y, w, h)
            album_roi = self.album_detector.extract_roi(frame, bbox)
            
            if album_roi is None:
                return None
            
            # Extract features (ensure model is loaded)
            if not hasattr(self.feature_extractor, 'model') or self.feature_extractor.model is None:
                self.feature_extractor.load_model()
            
            features = self.feature_extractor.extract_features(album_roi)
            
            # Search database
            if self.album_pipeline and features is not None:
                search_results = self.album_pipeline.search_similar_albums(features)
                
                if search_results:
                    best_match = search_results[0]
                    confidence = best_match.get('confidence', 0)
                    
                    if confidence >= self.confidence_var.get():
                        metadata = best_match.get('metadata', {})
                        
                        return RecognitionResult(
                            artist=metadata.get('artist', 'Unknown'),
                            title=metadata.get('title', 'Unknown'),
                            year=str(metadata.get('year', 'Unknown')),
                            confidence=confidence,
                            genre=metadata.get('genre'),
                            label=metadata.get('label'),
                            discogs_id=metadata.get('discogs_id')
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced processing error: {e}")
            return None
    
    def _update_ui(self):
        """Enhanced UI update loop."""
        if not self.running:
            return
        
        try:
            # Update camera display
            if not self.frame_queue.empty():
                latest_frame = None
                # Get the most recent frame
                while not self.frame_queue.empty():
                    try:
                        latest_frame = self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if latest_frame:
                    # Detect albums for overlay
                    detections = []
                    if self.overlay_var.get() and self.album_detector:
                        try:
                            album_bboxes = self.album_detector.detect_albums(latest_frame['frame'])
                            # Convert tuples to format expected by camera display
                            for bbox in album_bboxes:
                                x, y, w, h = bbox
                                detections.append({
                                    'bbox': (x, y, x + w, y + h),  # Convert to (x1, y1, x2, y2)
                                    'confidence': 0.8,  # Default confidence for display
                                    'original_width': latest_frame['frame'].shape[1],
                                    'original_height': latest_frame['frame'].shape[0]
                                })
                        except:
                            pass
                    
                    self.camera_display.update_frame(latest_frame['frame'], detections)
            
            # Update results
            try:
                result = self.result_queue.get_nowait()
                self._update_result_display(result)
            except queue.Empty:
                pass
        
        except Exception as e:
            logger.error(f"UI update error: {e}")
        
        # Schedule next update
        if self.running:
            self.root.after(33, self._update_ui)  # ~30 FPS UI updates
    
    def _update_result_display(self, result: RecognitionResult):
        """Update result display widgets."""
        self.current_result = result
        self.result_history.append(result)
        
        # Keep only recent results
        if len(self.result_history) > 50:
            self.result_history = self.result_history[-50:]
        
        # Update metadata
        self.artist_var.set(result.artist)
        self.title_var.set(result.title)
        self.year_var.set(result.year)
        
        # Update confidence meter
        self.confidence_meter.update_confidence(result.confidence)
        
        # TODO: Update album cover when cover_url is available
        # self.album_cover.update_cover(image_data)
        
        logger.info(f"Result updated: {result.artist} - {result.title} ({result.confidence:.1%})")
    
    # Settings event handlers
    def _toggle_detection(self):
        """Toggle detection on/off."""
        enabled = self.detection_var.get()
        self.settings['detection_enabled'] = enabled
        status = "enabled" if enabled else "disabled"
        self.status_var.set(f"Detection {status}")
    
    def _toggle_overlay(self):
        """Toggle overlay display."""
        enabled = self.overlay_var.get()
        self.settings['overlay_enabled'] = enabled
        if self.camera_display:
            self.camera_display.set_overlay_enabled(enabled)
    
    def _on_fps_change(self, value):
        """Handle FPS slider change."""
        fps = float(value)
        self.fps_label.config(text=f"{fps:.1f}")
        self.settings['frame_rate'] = fps
    
    def _on_confidence_change(self, value):
        """Handle confidence threshold change."""
        conf = float(value)
        self.confidence_label.config(text=f"{conf:.0%}")
        self.settings['confidence_threshold'] = conf
        if self.confidence_meter:
            self.confidence_meter.set_threshold(conf)
    
    def _refresh_cameras(self):
        """Refresh available cameras."""
        # TODO: Implement camera detection
        self.status_var.set("Camera list refreshed")
    
    # Menu and dialog handlers
    def _show_advanced_settings(self):
        """Show advanced settings dialog."""
        # TODO: Implement advanced settings dialog
        messagebox.showinfo("Settings", "Advanced settings dialog coming soon!")
    
    def _show_help(self):
        """Show help dialog."""
        help_text = """VinylVision - Quick Help

Controls:
• F11: Toggle fullscreen
• Esc: Exit fullscreen

Tips:
• Ensure good lighting for best results
• Hold album steady in camera view
• Adjust confidence threshold for accuracy
• Use pause to save battery during breaks

For detailed documentation, visit the project repository."""
        
        messagebox.showinfo("Help", help_text)
    
    def _save_current_result(self):
        """Save current recognition result."""
        if not self.current_result:
            messagebox.showwarning("Save Result", "No result to save")
            return
        
        # TODO: Implement result saving
        self.status_var.set("Result saved")
    
    def _view_result_details(self):
        """View detailed result information."""
        if not self.current_result:
            messagebox.showwarning("Result Details", "No result to view")
            return
        
        # TODO: Implement detailed result viewer
        details = f"""Album Details:

Artist: {self.current_result.artist}
Title: {self.current_result.title}
Year: {self.current_result.year}
Confidence: {self.current_result.confidence:.1%}
Processing Time: {self.current_result.processing_time:.1f}ms

Genre: {self.current_result.genre or 'Unknown'}
Label: {self.current_result.label or 'Unknown'}
Discogs ID: {self.current_result.discogs_id or 'Unknown'}"""
        
        messagebox.showinfo("Album Details", details)
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)
    
    def _on_closing(self):
        """Handle window closing."""
        if self.running:
            self._stop_capture()
        
        # Save settings
        try:
            settings_path = "user_data/settings.json"
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")
        
        self.root.destroy()
        logger.info("Enhanced VinylVision application closed")
    
    def run(self):
        """Start the enhanced application."""
        logger.info("Starting Enhanced VinylVision application")
        self.root.mainloop()


if __name__ == "__main__":
    app = EnhancedVinylVisionWindow()
    app.run()
