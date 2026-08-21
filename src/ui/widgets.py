"""
Custom UI widgets for VinylVision application.

Enhanced widgets for camera display, overlay rendering, audio visualization, and user interaction.
Fully integrated with centralized theme engine (ui.theme).
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import time
import io
from ui.theme import Colors, Fonts


class CameraDisplay(tk.Label):
    """Enhanced camera display widget with live corner drag-and-drop calibration."""
    
    def __init__(self, parent, width=640, height=480, on_corners_changed=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.target_width = width
        self.target_height = height
        self.display_size = (width, height)
        self.overlay_enabled = True
        self.fps_display = True
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        
        # Live Calibration Handling
        self.calibration_mode = False
        self.corners: List[List[float]] = []  # [x, y] coordinates in original frame space
        self.selected_corner_idx = -1
        self.scale = 1.0
        self.frame_dims = (height, width)  # (h, w)
        self.on_corners_changed = on_corners_changed

        # Mouse event bindings for corner dragging
        self.bind("<Button-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        self.configure(
            bg=Colors.BG_ROOT,
            text="Camera Initializing...",
            fg=Colors.TEXT_PRIMARY,
            font=Fonts.SETTINGS_TITLE,
            cursor="arrow"
        )
    
    def set_calibration_mode(self, enabled: bool):
        """Enable or disable interactive calibration mode."""
        self.calibration_mode = enabled
        self.configure(cursor="crosshair" if enabled else "arrow")

    def set_corners(self, corners: Optional[Any]):
        """Set the 4 initial calibration corner coordinates."""
        if corners is not None and len(corners) == 4:
            self.corners = [[float(p[0]), float(p[1])] for p in corners]
        else:
            self.corners = []

    def _on_mouse_down(self, event):
        if not self.calibration_mode or len(self.corners) != 4 or self.scale <= 0:
            return
        
        # Convert UI click coordinates to original camera pixel space
        click_x = event.x / self.scale
        click_y = event.y / self.scale
        
        # Tolerance radius to grab handle (25px on screen)
        hit_radius = 25 / self.scale
        
        self.selected_corner_idx = -1
        for idx, (cx, cy) in enumerate(self.corners):
            if np.hypot(cx - click_x, cy - click_y) <= hit_radius:
                self.selected_corner_idx = idx
                break

    def _on_mouse_drag(self, event):
        if not self.calibration_mode or self.selected_corner_idx == -1 or self.scale <= 0:
            return
            
        orig_h, orig_w = self.frame_dims
        new_x = float(np.clip(event.x / self.scale, 0, orig_w - 1))
        new_y = float(np.clip(event.y / self.scale, 0, orig_h - 1))
        
        self.corners[self.selected_corner_idx] = [new_x, new_y]
        
        if self.on_corners_changed:
            self.on_corners_changed(np.array(self.corners, dtype="float32"))

    def _on_mouse_up(self, event):
        self.selected_corner_idx = -1

    def update_frame(self, frame: np.ndarray, detected_corners: Optional[np.ndarray] = None):
        if frame is None:
            return
            
        try:
            self._update_fps()
            h, w = frame.shape[:2]
            self.frame_dims = (h, w)
            
            self.scale = min(self.target_width / w, self.target_height / h)
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            
            # If corner list is empty, initialize default quadrilateral
            if len(self.corners) != 4:
                if detected_corners is not None and len(detected_corners) == 4:
                    self.corners = [[float(p[0]), float(p[1])] for p in detected_corners]
                else:
                    pad_w, pad_h = w * 0.15, h * 0.15
                    self.corners = [
                        [pad_w, pad_h],
                        [w - pad_w, pad_h],
                        [w - pad_w, h - pad_h],
                        [pad_w, h - pad_h]
                    ]
                if self.on_corners_changed:
                    self.on_corners_changed(np.array(self.corners, dtype="float32"))

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_frame)
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(image)
            
            if self.overlay_enabled:
                font = Fonts.get_pil_font(Fonts.DETAIL_LABEL)
                scaled_pts = [(int(p[0] * self.scale), int(p[1] * self.scale)) for p in self.corners]
                
                if self.calibration_mode:
                    # In calibration mode: accent polygon and interactive control handles
                    draw.polygon(scaled_pts, outline=Colors.ACCENT_PRIMARY, width=3)
                    labels = ["TL", "TR", "BR", "BL"]
                    handle_colors = [Colors.ACCENT_DANGER, Colors.ACCENT_WARNING, Colors.ACCENT_PRIMARY, Colors.ACCENT_DISCOGS]
                    r = 8
                    for (pt, col, lbl) in zip(scaled_pts, handle_colors, labels):
                        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=col, outline=Colors.TEXT_PRIMARY, width=2)
                        draw.text((pt[0] + 12, pt[1] - 8), lbl, fill=Colors.TEXT_PRIMARY, font=font)
                    
                    draw.text((15, 15), "LIVE CALIBRATION - Drag the 4 corner pins to align the stand", fill=Colors.ACCENT_WARNING, font=font)
                else:
                    # Normal mode: subtle accent polygon around calibrated area
                    draw.polygon(scaled_pts, outline=Colors.ACCENT_SUCCESS, width=2)
                    draw.text((scaled_pts[0][0] + 5, max(5, scaled_pts[0][1] - 18)), "VINYL AREA (WARP 1:1)", fill=Colors.ACCENT_SUCCESS, font=font)
                
                if self.fps_display:
                    self._draw_fps(draw, (new_w, new_h))
            
            photo = ImageTk.PhotoImage(image=image)
            self.configure(image=photo, text="")
            self.image = photo
            
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def _update_fps(self):
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _add_overlays(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Add overlay graphics to frame."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        
        if detections:
            for detection in detections:
                self._draw_detection_box(draw, detection, image.size)
        
        if self.fps_display:
            self._draw_fps(draw, image.size)
        
        self._draw_crosshair(draw, image.size)
        return np.array(image)
    
    def _draw_detection_box(self, draw: ImageDraw.Draw, detection: Dict[str, Any], image_size: Tuple[int, int]):
        bbox = detection.get('bbox')
        confidence = detection.get('confidence', 0.0)
        
        if not bbox:
            return
        
        x1, y1, x2, y2 = bbox
        scale_x = image_size[0] / detection.get('original_width', image_size[0])
        scale_y = image_size[1] / detection.get('original_height', image_size[1])
        
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        
        color = Colors.ACCENT_SUCCESS if confidence > 0.8 else (Colors.ACCENT_WARNING if confidence > 0.6 else Colors.ACCENT_DANGER)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    def _draw_fps(self, draw: ImageDraw.Draw, image_size: Tuple[int, int]):
        fps_text = f"FPS: {self.current_fps:.1f}"
        font = Fonts.get_pil_font(Fonts.DETAIL_LABEL)
        draw.text((image_size[0] - 70, 10), fps_text, fill=Colors.TEXT_PRIMARY, font=font)
    
    def _draw_crosshair(self, draw: ImageDraw.Draw, image_size: Tuple[int, int]):
        center_x = image_size[0] // 2
        center_y = image_size[1] // 2
        size = 20
        draw.line([center_x - size, center_y, center_x + size, center_y], fill=Colors.TEXT_PRIMARY, width=2)
        draw.line([center_x, center_y - size, center_x, center_y + size], fill=Colors.TEXT_PRIMARY, width=2)

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled
    
    def set_fps_display(self, enabled: bool):
        self.fps_display = enabled


class ConfidenceMeter(ttk.Frame):
    """Animated confidence meter widget displaying current match likelihood and threshold."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.threshold_var = tk.DoubleVar(value=0.6)
        self._setup_ui()
    
    def _setup_ui(self):
        header_frame = tk.Frame(self, bg=Colors.BG_CARD)
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="Likelihood:", font=Fonts.SETTINGS_LABEL, fg=Colors.TEXT_PRIMARY, bg=Colors.BG_CARD).pack(side="left")
        self.status_label = tk.Label(header_frame, text="●", font=Fonts.SETTINGS_TITLE, fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.status_label.pack(side="right")
        
        self.progress = ttk.Progressbar(
            self, 
            length=200, 
            mode='determinate', 
            variable=self.confidence_var
        )
        self.progress.pack(fill="x", pady=4)
        
        info_frame = tk.Frame(self, bg=Colors.BG_CARD)
        info_frame.pack(fill="x", pady=2)
        
        self.value_label = tk.Label(info_frame, text="Likelihood: 0.0%", font=Fonts.DETAIL_VALUE, fg=Colors.TEXT_PRIMARY, bg=Colors.BG_CARD)
        self.value_label.pack(side="left")
        
        self.threshold_label = tk.Label(info_frame, text="Threshold: 60%", font=Fonts.DETAIL_LABEL, fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.threshold_label.pack(side="right")
    
    def update_confidence(self, confidence: float):
        """Updates the current likelihood value."""
        self.confidence_var.set(confidence * 100)
        self.value_label.config(text=f"Likelihood: {confidence:.1%}")
        
        threshold = self.threshold_var.get()
        if confidence >= threshold:
            self.status_label.config(text="●", fg=Colors.ACCENT_SUCCESS)
        elif confidence >= threshold * 0.75:
            self.status_label.config(text="●", fg=Colors.ACCENT_WARNING)
        else:
            self.status_label.config(text="●", fg=Colors.ACCENT_DANGER)
    
    def set_threshold(self, threshold: float):
        self.threshold_var.set(threshold)
        self.threshold_label.config(text=f"Threshold: {threshold:.0%}")


class AlbumCoverDisplay(tk.Label):
    """Album cover image display widget."""
    
    def __init__(self, parent, size=(150, 150), **kwargs):
        super().__init__(parent, **kwargs)
        self.size = size
        self.placeholder_image = None
        self.configure(
            bg=Colors.BG_CANVAS_EMPTY,
            text="No Album",
            fg=Colors.TEXT_MUTED,
            compound="center",
            font=Fonts.DETAIL_LABEL
        )
        self._create_placeholder()
    
    def _create_placeholder(self):
        placeholder = Image.new('RGB', self.size, color=Colors.BG_CANVAS_EMPTY)
        draw = ImageDraw.Draw(placeholder)
        center_x, center_y = self.size[0] // 2, self.size[1] // 2
        radius = min(self.size) // 3
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], outline=Colors.BORDER_FOCUS, width=2)
        inner_radius = radius // 6
        draw.ellipse([center_x - inner_radius, center_y - inner_radius, center_x + inner_radius, center_y + inner_radius], outline=Colors.BORDER_FOCUS, width=2)
        self.placeholder_image = ImageTk.PhotoImage(placeholder)
        self.configure(image=self.placeholder_image)
    
    def update_cover(self, image_data: bytes = None, url: str = None):
        try:
            if image_data:
                image = Image.open(io.BytesIO(image_data))
                image = image.resize(self.size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.configure(image=photo, text="")
                self.image = photo
            else:
                self.configure(image=self.placeholder_image, text="No Album", fg=Colors.TEXT_MUTED)
        except Exception as e:
            print(f"Error updating album cover: {e}")
            self.configure(image=self.placeholder_image, text="Error Loading", fg=Colors.ACCENT_DANGER)


class StatusIndicator(ttk.Frame):
    """System status indicator with multiple states."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.states = {
            'disconnected': {'color': Colors.ACCENT_DANGER, 'text': 'Disconnected'},
            'connecting': {'color': Colors.ACCENT_WARNING, 'text': 'Connecting...'},
            'ready': {'color': Colors.ACCENT_SUCCESS, 'text': 'Ready'},
            'processing': {'color': Colors.ACCENT_PRIMARY, 'text': 'Processing'},
            'error': {'color': Colors.ACCENT_DANGER, 'text': 'Error'}
        }
        self.current_state = 'disconnected'
        self._setup_ui()
    
    def _setup_ui(self):
        self.indicator = tk.Label(self, text="●", font=Fonts.LOGO, bg=Colors.BG_CARD)
        self.indicator.pack(side="left", padx=(0, 5))
        self.status_label = tk.Label(self, text="Disconnected", font=Fonts.DETAIL_LABEL, fg=Colors.TEXT_PRIMARY, bg=Colors.BG_CARD)
        self.status_label.pack(side="left")
        self.set_state('disconnected')
    
    def set_state(self, state: str, custom_text: str = None):
        if state not in self.states:
            return
        self.current_state = state
        state_info = self.states[state]
        self.indicator.config(fg=state_info['color'])
        self.status_label.config(text=custom_text or state_info['text'])


class PerformanceMonitor(ttk.LabelFrame):
    """Real-time performance monitoring widget."""
    
    def __init__(self, parent, **kwargs):
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
        ttk.Label(self, text="FPS:", font=Fonts.DETAIL_LABEL).grid(row=0, column=0, sticky="w")
        self.fps_label = ttk.Label(self, text="0.0", font=Fonts.DETAIL_VALUE)
        self.fps_label.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(self, text="Process:", font=Fonts.DETAIL_LABEL).grid(row=1, column=0, sticky="w")
        self.process_label = ttk.Label(self, text="0ms", font=Fonts.DETAIL_VALUE)
        self.process_label.grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(self, text="Memory:", font=Fonts.DETAIL_LABEL).grid(row=2, column=0, sticky="w")
        self.memory_label = ttk.Label(self, text="0MB", font=Fonts.DETAIL_VALUE)
        self.memory_label.grid(row=2, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(self, text="Queue:", font=Fonts.DETAIL_LABEL).grid(row=3, column=0, sticky="w")
        self.queue_label = ttk.Label(self, text="0", font=Fonts.DETAIL_VALUE)
        self.queue_label.grid(row=3, column=1, sticky="w", padx=(5, 0))
    
    def _start_monitoring(self):
        self._update_display()
        self.after(1000, self._start_monitoring)
    
    def _update_display(self):
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics['memory_usage'] = memory_mb
        except ImportError:
            pass
        
        self.fps_label.config(text=f"{self.metrics['fps']:.1f}")
        self.process_label.config(text=f"{self.metrics['processing_time']:.0f}ms")
        self.memory_label.config(text=f"{self.metrics['memory_usage']:.0f}MB")
        self.queue_label.config(text=str(self.metrics['queue_size']))
    
    def update_metrics(self, **kwargs):
        self.metrics.update(kwargs)


class AudioSpectrumVisualizer(tk.Canvas):
    """
    Renders real-time FFT spectrum bars filling 100% of available canvas width.
    Handles dynamic responsive width and NumPy/list arrays safely.
    """
    def __init__(self, parent, width: int = 180, height: int = 40, num_bars: int = 24, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=Colors.BG_CARD,
            highlightthickness=0,
            **kwargs
        )
        self.w = width
        self.h = height
        self.num_bars = num_bars
        self.bar_values = [0.0] * num_bars
        self.target_values = [0.0] * num_bars
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        """Captures actual runtime width when packed with fill=tk.X."""
        if event.width > 10:
            self.w = event.width
            self.h = event.height
            self._redraw()

    def render_spectrum(self, spectrum_data: Any = None):
        """Standard update entry point expected by kiosk_window."""
        self.update_levels(spectrum_data)

    def update_levels(self, levels: Any = None):
        """Sets new target FFT levels safely and triggers redraw."""
        if levels is None or len(levels) == 0:
            self.target_values = [0.0] * self.num_bars
        else:
            total_len = len(levels)
            step = max(1, total_len // self.num_bars)
            self.target_values = [
                min(1.0, max(0.0, float(levels[i * step]) if i * step < total_len else 0.0))
                for i in range(self.num_bars)
            ]
        self._redraw()

    def _redraw(self):
        self.delete("all")
        if self.num_bars == 0:
            return

        # Rileva la larghezza effettiva per coprire tutto lo spazio
        canvas_width = self.winfo_width() if self.winfo_width() > 10 else self.w
        canvas_height = self.winfo_height() if self.winfo_height() > 10 else self.h

        spacing = 2
        total_spacing = spacing * (self.num_bars - 1)
        bar_width = max(1.0, (canvas_width - total_spacing) / float(self.num_bars))

        for i, val in enumerate(self.target_values):
            self.bar_values[i] += (val - self.bar_values[i]) * 0.4
            bar_h = max(2, int(self.bar_values[i] * (canvas_height - 4)))
            
            x0 = int(i * (bar_width + spacing))
            # L'ultima barra si estende esattamente fino al pixel finale del canvas
            x1 = canvas_width if i == self.num_bars - 1 else int(x0 + bar_width)
            y0 = canvas_height - bar_h
            y1 = canvas_height

            fill_color = Colors.SPECTRUM_PEAK if self.bar_values[i] > 0.85 else Colors.SPECTRUM_BAR
            self.create_rectangle(x0, y0, x1, y1, fill=fill_color, width=0)


class LyricsDisplay(ttk.Frame):
    """7-line synchronized lyrics display and Spotify-style Progress Bar using theme tokens."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.configure(padding=10)
        
        # 1. Track header
        self.track_header = tk.Label(
            self, 
            text="SYNCHRONIZED LYRICS", 
            font=Fonts.SECTION_HEADER, 
            fg=Colors.ACCENT_SUCCESS,  # Highlight Hi-Fi / Spotify Green
            bg=Colors.BG_CARD
        )
        self.track_header.pack(anchor="w", pady=(0, 4))

        # 2. Lyrics lines container (expands to center vertically)
        self.lines_container = tk.Frame(self, bg=Colors.BG_CARD)
        self.lines_container.pack(fill=tk.BOTH, expand=True)

        # 7 Lines with gradual fading toward borders based on theme hierarchy
        # Line -3 (distant past)
        self.lbl_p3 = tk.Label(
            self.lines_container, 
            text="", 
            font=Fonts.LYRICS_LINE_DIST, 
            fg=Colors.LYRICS_DISTANT, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_p3.pack(fill=tk.BOTH, expand=True, pady=1)

        # Line -2 (far past)
        self.lbl_p2 = tk.Label(
            self.lines_container, 
            text="", 
            font=Fonts.LYRICS_LINE_FAR, 
            fg=Colors.LYRICS_FAR, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_p2.pack(fill=tk.BOTH, expand=True, pady=1)

        # Line -1 (near past)
        self.lbl_p1 = tk.Label(
            self.lines_container, 
            text="", 
            font=Fonts.LYRICS_LINE_NEAR, 
            fg=Colors.LYRICS_NEAR, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_p1.pack(fill=tk.BOTH, expand=True, pady=2)

        # Line 0 (ACTIVE / NOW SINGING - Full Focus)
        self.lbl_curr = tk.Label(
            self.lines_container, 
            text="Waiting for track...", 
            font=Fonts.LYRICS_LINE_ACTIVE, 
            fg=Colors.LYRICS_ACTIVE, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_curr.pack(fill=tk.BOTH, expand=True, pady=3)

        # Line +1 (next line)
        self.lbl_n1 = tk.Label(
            self.lines_container, 
            text="", 
            font=Fonts.LYRICS_LINE_NEAR, 
            fg=Colors.LYRICS_NEAR, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_n1.pack(fill=tk.BOTH, expand=True, pady=2)

        # Line +2 (upcoming)
        self.lbl_n2 = tk.Label(
            self.lines_container, 
            text="", 
            font=Fonts.LYRICS_LINE_FAR, 
            fg=Colors.LYRICS_FAR, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_n2.pack(fill=tk.BOTH, expand=True, pady=1)

        # Line +3 (distant upcoming)
        self.lbl_n3 = tk.Label(
            self.lines_container, 
            text="", 
            font=Fonts.LYRICS_LINE_DIST, 
            fg=Colors.LYRICS_DISTANT, 
            bg=Colors.BG_CARD, 
            anchor="w", 
            wraplength=580, 
            justify="left"
        )
        self.lbl_n3.pack(fill=tk.BOTH, expand=True, pady=1)

        # 3. Spotify Progress Bar & Timers
        self.progress_container = tk.Frame(self, bg=Colors.BG_CARD)
        self.progress_container.pack(fill=tk.X, pady=(6, 0))

        self.time_cur_lbl = tk.Label(
            self.progress_container, 
            text="0:00", 
            font=Fonts.DETAIL_LABEL, 
            fg=Colors.TEXT_MUTED, 
            bg=Colors.BG_CARD, 
            width=5, 
            anchor="w"
        )
        self.time_cur_lbl.pack(side=tk.LEFT)

        self.prog_canvas = tk.Canvas(
            self.progress_container, 
            bg=Colors.BORDER_FOCUS, 
            height=5, 
            highlightthickness=0
        )
        self.prog_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.prog_bar = self.prog_canvas.create_rectangle(
            0, 0, 0, 5, 
            fill=Colors.ACCENT_SUCCESS, 
            width=0
        )

        self.time_tot_lbl = tk.Label(
            self.progress_container, 
            text="--:--", 
            font=Fonts.DETAIL_LABEL, 
            fg=Colors.TEXT_MUTED, 
            bg=Colors.BG_CARD, 
            width=5, 
            anchor="e"
        )
        self.time_tot_lbl.pack(side=tk.RIGHT)

    @staticmethod
    def _format_time(seconds: float) -> str:
        s = max(0, int(seconds))
        m = s // 60
        sec = s % 60
        return f"{m}:{sec:02d}"

    def update_progress(self, current_sec: float, total_sec: float):
        """Updates progress bar and playback timestamps."""
        self.time_cur_lbl.config(text=self._format_time(current_sec))
        
        if total_sec > 0:
            self.time_tot_lbl.config(text=self._format_time(total_sec))
            w = self.prog_canvas.winfo_width()
            if w > 1:
                ratio = min(1.0, max(0.0, current_sec / total_sec))
                self.prog_canvas.coords(self.prog_bar, 0, 0, int(w * ratio), 5)
        else:
            self.time_tot_lbl.config(text="--:--")
            self.prog_canvas.coords(self.prog_bar, 0, 0, 0, 5)

    def update_lyrics_7lines(self, track_name: str, p3: str, p2: str, p1: str, curr: str, n1: str, n2: str, n3: str):
        """Updates all 7 lyrics lines simultaneously."""
        if track_name:
            self.track_header.config(text=f"LYRICS • {track_name.upper()}")
            
        self.lbl_p3.config(text=p3 or "")
        self.lbl_p2.config(text=p2 or "")
        self.lbl_p1.config(text=p1 or "")
        self.lbl_curr.config(text=curr or ("..." if track_name else "Waiting for track..."))
        self.lbl_n1.config(text=n1 or "")
        self.lbl_n2.config(text=n2 or "")
        self.lbl_n3.config(text=n3 or "")


__all__ = [
    'CameraDisplay',
    'ConfidenceMeter', 
    'AlbumCoverDisplay',
    'StatusIndicator',
    'PerformanceMonitor',
    'AudioSpectrumVisualizer',
    'LyricsDisplay'
]