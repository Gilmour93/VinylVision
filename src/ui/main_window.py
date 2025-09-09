"""
VinylVision Main UI Window

Real-time album recognition interface with camera feed, results display,
and user controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from loguru import logger
import json
import os

try:
    # Try relative imports first (when used as package)
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


class SettingsPanel:
    """User settings and configuration panel."""
    
    def __init__(self, parent: tk.Widget, settings_callback):
        self.parent = parent
        self.settings_callback = settings_callback
        self.settings = self._load_default_settings()
        
    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default application settings."""
        return {
            'confidence_threshold': 0.8,
            'frame_rate': 2.0,
            'camera_source': 0,
            'detection_enabled': True,
            'overlay_enabled': True,
            'auto_save_results': False
        }
    
    def create_panel(self) -> tk.Frame:
        """Create the settings panel UI."""
        frame = ttk.LabelFrame(self.parent, text="Settings", padding="10")
        
        # Confidence threshold
        ttk.Label(frame, text="Confidence Threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.confidence_var = tk.DoubleVar(value=self.settings['confidence_threshold'])
        confidence_scale = ttk.Scale(
            frame, from_=0.5, to=1.0, 
            variable=self.confidence_var,
            command=self._on_confidence_change
        )
        confidence_scale.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=2)
        self.confidence_label = ttk.Label(frame, text=f"{self.settings['confidence_threshold']:.2f}")
        self.confidence_label.grid(row=0, column=2, padx=(5, 0), pady=2)
        
        # Frame rate
        ttk.Label(frame, text="Frame Rate (FPS):").grid(row=1, column=0, sticky="w", pady=2)
        self.framerate_var = tk.DoubleVar(value=self.settings['frame_rate'])
        framerate_scale = ttk.Scale(
            frame, from_=0.5, to=5.0,
            variable=self.framerate_var,
            command=self._on_framerate_change
        )
        framerate_scale.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=2)
        self.framerate_label = ttk.Label(frame, text=f"{self.settings['frame_rate']:.1f}")
        self.framerate_label.grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # Toggle controls
        self.detection_var = tk.BooleanVar(value=self.settings['detection_enabled'])
        ttk.Checkbutton(
            frame, text="Enable Detection",
            variable=self.detection_var,
            command=self._on_detection_toggle
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        self.overlay_var = tk.BooleanVar(value=self.settings['overlay_enabled'])
        ttk.Checkbutton(
            frame, text="Show Overlay",
            variable=self.overlay_var,
            command=self._on_overlay_toggle
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        
        # Camera selection
        ttk.Label(frame, text="Camera:").grid(row=4, column=0, sticky="w", pady=5)
        self.camera_var = tk.IntVar(value=self.settings['camera_source'])
        camera_spin = ttk.Spinbox(
            frame, from_=0, to=3, width=5,
            textvariable=self.camera_var,
            command=self._on_camera_change
        )
        camera_spin.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=5)
        
        frame.grid_columnconfigure(1, weight=1)
        return frame
    
    def _on_confidence_change(self, value):
        """Handle confidence threshold change."""
        conf = float(value)
        self.confidence_label.config(text=f"{conf:.2f}")
        self.settings['confidence_threshold'] = conf
        self.settings_callback('confidence_threshold', conf)
    
    def _on_framerate_change(self, value):
        """Handle frame rate change."""
        fps = float(value)
        self.framerate_label.config(text=f"{fps:.1f}")
        self.settings['frame_rate'] = fps
        self.settings_callback('frame_rate', fps)
    
    def _on_detection_toggle(self):
        """Handle detection enable/disable."""
        enabled = self.detection_var.get()
        self.settings['detection_enabled'] = enabled
        self.settings_callback('detection_enabled', enabled)
    
    def _on_overlay_toggle(self):
        """Handle overlay enable/disable."""
        enabled = self.overlay_var.get()
        self.settings['overlay_enabled'] = enabled
        self.settings_callback('overlay_enabled', enabled)
    
    def _on_camera_change(self):
        """Handle camera source change."""
        source = self.camera_var.get()
        self.settings['camera_source'] = source
        self.settings_callback('camera_source', source)


class ResultsPanel:
    """Album recognition results display panel."""
    
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.current_result: Optional[RecognitionResult] = None
        self.history: List[RecognitionResult] = []
        
    def create_panel(self) -> tk.Frame:
        """Create the results panel UI."""
        frame = ttk.LabelFrame(self.parent, text="Recognition Results", padding="10")
        
        # Main result display
        self.result_frame = ttk.Frame(frame)
        self.result_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Album cover placeholder
        self.cover_label = ttk.Label(self.result_frame, text="No album detected", 
                                   background="lightgray", width=20, anchor="center")
        self.cover_label.grid(row=0, column=0, rowspan=4, padx=(0, 15), pady=5)
        
        # Metadata display
        self.artist_var = tk.StringVar(value="Artist: —")
        self.title_var = tk.StringVar(value="Title: —")
        self.year_var = tk.StringVar(value="Year: —")
        self.confidence_var = tk.StringVar(value="Confidence: —")
        
        ttk.Label(self.result_frame, textvariable=self.artist_var, font=("Arial", 10, "bold")).grid(
            row=0, column=1, sticky="w", pady=2)
        ttk.Label(self.result_frame, textvariable=self.title_var, font=("Arial", 10)).grid(
            row=1, column=1, sticky="w", pady=2)
        ttk.Label(self.result_frame, textvariable=self.year_var, font=("Arial", 9)).grid(
            row=2, column=1, sticky="w", pady=2)
        ttk.Label(self.result_frame, textvariable=self.confidence_var, font=("Arial", 9)).grid(
            row=3, column=1, sticky="w", pady=2)
        
        # Confidence meter
        self.confidence_progress = ttk.Progressbar(
            frame, length=200, mode='determinate')
        self.confidence_progress.grid(row=1, column=0, sticky="ew", pady=5)
        
        # History list
        history_frame = ttk.LabelFrame(frame, text="Recent Detections", padding="5")
        history_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        # Scrollable history listbox
        self.history_listbox = tk.Listbox(history_frame, height=6, font=("Arial", 9))
        history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", 
                                        command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.grid(row=0, column=0, sticky="ew")
        history_scrollbar.grid(row=0, column=1, sticky="ns")
        
        history_frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        return frame
    
    def update_result(self, result: Optional[RecognitionResult]):
        """Update the main result display."""
        self.current_result = result
        
        if result:
            self.artist_var.set(f"Artist: {result.artist}")
            self.title_var.set(f"Title: {result.title}")
            self.year_var.set(f"Year: {result.year}")
            self.confidence_var.set(f"Confidence: {result.confidence:.1%}")
            
            # Update confidence meter
            self.confidence_progress['value'] = result.confidence * 100
            
            # Add to history
            self._add_to_history(result)
            
            # TODO: Load and display album cover image
            self.cover_label.config(text=f"{result.artist}\n{result.title}")
            
        else:
            self.artist_var.set("Artist: —")
            self.title_var.set("Title: —")
            self.year_var.set("Year: —")
            self.confidence_var.set("Confidence: —")
            self.confidence_progress['value'] = 0
            self.cover_label.config(text="No album detected")
    
    def _add_to_history(self, result: RecognitionResult):
        """Add result to history list."""
        self.history.append(result)
        
        # Keep only recent 20 items
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        # Update listbox
        entry = f"{result.artist} - {result.title} ({result.confidence:.1%})"
        self.history_listbox.insert(0, entry)
        
        # Remove old items from listbox
        if self.history_listbox.size() > 20:
            self.history_listbox.delete(20, tk.END)


class VinylVisionMainWindow:
    """Main application window for VinylVision."""
    
    def __init__(self):
        """Initialize the main window."""
        self.root = tk.Tk()
        self.root.title("VinylVision - Real-time Album Recognition")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Application state
        self.running = False
        self.camera_manager: Optional[CameraManager] = None
        self.album_detector: Optional[AlbumDetector] = None
        self.album_pipeline: Optional[AlbumDataPipeline] = None
        self.feature_extractor: Optional[AlbumFeatureExtractor] = None
        
        # Threading
        self.video_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        
        # UI components
        self.video_label: Optional[tk.Label] = None
        self.results_panel: Optional[ResultsPanel] = None
        self.settings_panel: Optional[SettingsPanel] = None
        
        # Settings
        self.current_settings = {
            'confidence_threshold': 0.8,
            'frame_rate': 2.0,
            'camera_source': 0,
            'detection_enabled': True,
            'overlay_enabled': True
        }
        
        self._setup_ui()
        self._load_configuration()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container with paned windows
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)
        
        # Camera feed
        camera_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="10")
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(camera_frame, bg="black", text="Camera Initializing...", 
                                  fg="white", font=("Arial", 14))
        self.video_label.pack(expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="Start Camera", command=self._start_capture)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Camera", command=self._stop_capture, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="Save Settings", command=self._save_settings).pack(side=tk.RIGHT)
        
        # Right panel - Results and Settings
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Results panel
        self.results_panel = ResultsPanel(right_frame)
        results_widget = self.results_panel.create_panel()
        results_widget.pack(fill=tk.X, pady=(0, 10))
        
        # Settings panel
        self.settings_panel = SettingsPanel(right_frame, self._on_setting_change)
        settings_widget = self.settings_panel.create_panel()
        settings_widget.pack(fill=tk.X)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Menu bar
        self._create_menu()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Settings...", command=self._load_settings_file)
        file_menu.add_command(label="Save Settings...", command=self._save_settings_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Full Screen", command=self._toggle_fullscreen)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _load_configuration(self):
        """Load application configuration."""
        try:
            config = load_config()
            
            # Initialize backend components
            self.camera_manager = CameraManager()
            self.album_detector = AlbumDetector()
            self.feature_extractor = AlbumFeatureExtractor()
            
            # Initialize album pipeline with Discogs credentials
            if config.discogs.consumer_key and config.discogs.consumer_secret:
                self.album_pipeline = AlbumDataPipeline(
                    config.discogs.consumer_key,
                    config.discogs.consumer_secret
                )
            
            self.status_var.set("Configuration loaded successfully")
            logger.info("VinylVision configuration loaded")
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            self.status_var.set(error_msg)
            logger.error(error_msg)
            messagebox.showerror("Configuration Error", error_msg)
    
    def _on_setting_change(self, setting: str, value: Any):
        """Handle setting changes from the settings panel."""
        self.current_settings[setting] = value
        logger.info(f"Setting changed: {setting} = {value}")
        
        # Apply specific setting changes
        if setting == 'camera_source' and self.camera_manager:
            if self.running:
                self._stop_capture()
                self.camera_manager.release()
                time.sleep(0.5)  # Give time for cleanup
                self._start_capture()
        
        elif setting == 'frame_rate':
            # Frame rate will be applied in video capture loop
            pass
    
    def _start_capture(self):
        """Start video capture and processing."""
        if self.running:
            return
            
        try:
            # Initialize camera
            if not self.camera_manager:
                raise Exception("Camera manager not initialized")
            
            camera_source = self.current_settings['camera_source']
            if not self.camera_manager.initialize_camera(camera_source):
                raise Exception(f"Failed to initialize camera {camera_source}")
            
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Start video capture thread
            self.video_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
            self.video_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start UI update loop
            self._update_ui()
            
            self.status_var.set("Camera active - Album detection enabled")
            logger.info("Video capture started")
            
        except Exception as e:
            error_msg = f"Failed to start camera: {str(e)}"
            self.status_var.set(error_msg)
            logger.error(error_msg)
            messagebox.showerror("Camera Error", error_msg)
            
    def _stop_capture(self):
        """Stop video capture and processing."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop threads
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera_manager:
            self.camera_manager.release()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.video_label.config(image="", text="Camera stopped")
        
        self.status_var.set("Camera stopped")
        logger.info("Video capture stopped")
    
    def _video_capture_loop(self):
        """Video capture loop running in separate thread."""
        frame_time = 1.0 / self.current_settings['frame_rate']
        
        while self.running:
            try:
                # Capture frame
                frame = self.camera_manager.capture_frame()
                if frame is not None:
                    # Add timestamp
                    frame_data = {
                        'frame': frame,
                        'timestamp': time.time()
                    }
                    
                    # Add to queue (drop old frames if queue is full)
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                
                time.sleep(frame_time)
                
            except Exception as e:
                logger.error(f"Video capture error: {e}")
                if self.running:
                    time.sleep(0.1)  # Brief pause before retry
    
    def _processing_loop(self):
        """Frame processing loop running in separate thread."""
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1.0)
                frame = frame_data['frame']
                
                # Process frame if detection is enabled
                if self.current_settings['detection_enabled']:
                    result = self._process_frame(frame)
                    if result:
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            # Remove old result and add new one
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait(result)
                            except queue.Empty:
                                pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> Optional[RecognitionResult]:
        """Process a single frame for album recognition."""
        try:
            # Detect albums in frame
            albums = self.album_detector.detect_albums(frame)
            
            if not albums:
                return None
            
            # Process the best album detection
            best_album = max(albums, key=lambda x: x.get('confidence', 0))
            album_roi = best_album['roi']
            
            # Extract features
            features = self.feature_extractor.extract_features(album_roi)
            
            # Search in database
            if self.album_pipeline:
                search_results = self.album_pipeline.search_album(features)
                
                if search_results:
                    best_match = search_results[0]
                    confidence = best_match.get('confidence', 0)
                    
                    # Check confidence threshold
                    if confidence >= self.current_settings['confidence_threshold']:
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
            logger.error(f"Frame processing error: {e}")
            return None
    
    def _update_ui(self):
        """Update UI elements from main thread."""
        if not self.running:
            return
        
        try:
            # Update video display
            if not self.frame_queue.empty():
                frame_data = self.frame_queue.queue[-1]  # Get latest frame
                frame = frame_data['frame']
                
                # Convert frame for display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add overlay if enabled
                if self.current_settings['overlay_enabled'] and self.album_detector:
                    albums = self.album_detector.detect_albums(frame)
                    for album in albums:
                        roi = album['roi']
                        # Draw bounding box (simplified overlay)
                        # TODO: Implement proper overlay rendering
                
                # Convert to PhotoImage
                image = Image.fromarray(display_frame)
                # Resize to fit display
                display_size = (640, 480)
                image = image.resize(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo  # Keep reference
            
            # Update results
            try:
                result = self.result_queue.get_nowait()
                self.results_panel.update_result(result)
            except queue.Empty:
                pass
                
        except Exception as e:
            logger.error(f"UI update error: {e}")
        
        # Schedule next update
        if self.running:
            self.root.after(50, self._update_ui)  # 20 FPS UI update
    
    def _save_settings(self):
        """Save current settings to file."""
        try:
            settings_path = "user_data/settings.json"
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            
            with open(settings_path, 'w') as f:
                json.dump(self.current_settings, f, indent=2)
            
            self.status_var.set("Settings saved")
            logger.info("Settings saved successfully")
            
        except Exception as e:
            error_msg = f"Failed to save settings: {str(e)}"
            self.status_var.set(error_msg)
            logger.error(error_msg)
    
    def _load_settings_file(self):
        """Load settings from file dialog."""
        try:
            filename = filedialog.askopenfilename(
                title="Load Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                
                self.current_settings.update(settings)
                # TODO: Update UI elements with loaded settings
                self.status_var.set("Settings loaded")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings: {str(e)}")
    
    def _save_settings_file(self):
        """Save settings using file dialog."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.current_settings, f, indent=2)
                
                self.status_var.set("Settings saved")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings: {str(e)}")
    
    def _export_results(self):
        """Export recognition results."""
        if not self.results_panel.history:
            messagebox.showinfo("Export", "No results to export")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # TODO: Implement results export
                self.status_var.set("Results exported")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """VinylVision v1.0

Real-time vinyl record album cover recognition powered by computer vision.

Features:
• EfficientNet-B0 based recognition
• ChromaDB vector database
• Discogs API integration
• Real-time camera processing

© 2024 VinylVision Project"""
        
        messagebox.showinfo("About VinylVision", about_text)
    
    def _on_closing(self):
        """Handle window closing."""
        if self.running:
            self._stop_capture()
        
        self._save_settings()
        self.root.destroy()
        logger.info("VinylVision application closed")
    
    def run(self):
        """Start the application."""
        logger.info("Starting VinylVision application")
        self.root.mainloop()


if __name__ == "__main__":
    app = VinylVisionMainWindow()
    app.run()
