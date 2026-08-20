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
        
        # Gestione Calibrazione Live
        self.calibration_mode = False
        self.corners: List[List[float]] = []  # Coordinate [x, y] nello spazio originale del frame
        self.selected_corner_idx = -1
        self.scale = 1.0
        self.frame_dims = (height, width)  # (h, w)
        self.on_corners_changed = on_corners_changed

        # Bind eventi mouse per il trascinamento dei punti
        self.bind("<Button-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        self.configure(
            bg="black",
            text="Camera Initializing...",
            fg="white",
            font=("Arial", 14),
            cursor="arrow"
        )
    
    def set_calibration_mode(self, enabled: bool):
        """Attiva o disattiva la modalità calibrazione interattiva."""
        self.calibration_mode = enabled
        self.configure(cursor="crosshair" if enabled else "arrow")

    def set_corners(self, corners: Optional[Any]):
        """Imposta i 4 vertici di calibrazione iniziali."""
        if corners is not None and len(corners) == 4:
            self.corners = [[float(p[0]), float(p[1])] for p in corners]
        else:
            self.corners = []

    def _on_mouse_down(self, event):
        if not self.calibration_mode or len(self.corners) != 4 or self.scale <= 0:
            return
        
        # Converte coordinate click dalla UI allo spazio pixel originale della camera
        click_x = event.x / self.scale
        click_y = event.y / self.scale
        
        # Raggio di tolleranza per agganciare il punto con il cursore (25px su schermo)
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
            
            # Se la lista dei punti è vuota, crea un quadrilatero di default
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
                font = ImageFont.load_default()
                scaled_pts = [(int(p[0] * self.scale), int(p[1] * self.scale)) for p in self.corners]
                
                if self.calibration_mode:
                    # In modalità calibrazione: poligono magenta e cerchi di controllo interattivi
                    draw.polygon(scaled_pts, outline="#FF007F", width=3)
                    labels = ["TL", "TR", "BR", "BL"]
                    handle_colors = ["#FF3333", "#FFFF33", "#3388FF", "#FF33FF"]
                    r = 8
                    for (pt, col, lbl) in zip(scaled_pts, handle_colors, labels):
                        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=col, outline="white", width=2)
                        draw.text((pt[0] + 12, pt[1] - 8), lbl, fill="white", font=font)
                    
                    draw.text((15, 15), "CALIBRAZIONE LIVE - Trascina i 4 cerchi sugli angoli del supporto", fill="#FFCC00", font=font)
                else:
                    # Visualizzazione normale: poligono verde sull'area calibrata
                    draw.polygon(scaled_pts, outline="#00FF66", width=2)
                    draw.text((scaled_pts[0][0] + 5, max(5, scaled_pts[0][1] - 18)), "AREA VINILE (WARP 1:1)", fill="#00FF66", font=font)
                
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
        
        color = "green" if confidence > 0.8 else ("yellow" if confidence > 0.6 else "red")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    def _draw_fps(self, draw: ImageDraw.Draw, image_size: Tuple[int, int]):
        fps_text = f"FPS: {self.current_fps:.1f}"
        font = ImageFont.load_default()
        draw.text((image_size[0] - 70, 10), fps_text, fill="white", font=font)
    
    def _draw_crosshair(self, draw: ImageDraw.Draw, image_size: Tuple[int, int]):
        center_x = image_size[0] // 2
        center_y = image_size[1] // 2
        size = 20
        draw.line([center_x - size, center_y, center_x + size, center_y], fill="white", width=2)
        draw.line([center_x, center_y - size, center_x, center_y + size], fill="white", width=2)

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled
    
    def set_fps_display(self, enabled: bool):
        self.fps_display = enabled


class ConfidenceMeter(ttk.Frame):
    """Animated confidence meter widget displaying current match likelihood and threshold."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.threshold_var = tk.DoubleVar(value=0.6)
        self._setup_ui()
    
    def _setup_ui(self):
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x")
        
        ttk.Label(header_frame, text="Likelihood:", font=("Arial", 10, "bold")).pack(side="left")
        self.status_label = ttk.Label(header_frame, text="●", font=("Arial", 14), foreground="gray")
        self.status_label.pack(side="right")
        
        self.progress = ttk.Progressbar(
            self, 
            length=200, 
            mode='determinate',
            variable=self.confidence_var
        )
        self.progress.pack(fill="x", pady=4)
        
        info_frame = ttk.Frame(self)
        info_frame.pack(fill="x", pady=2)
        
        # Etichetta ben visibile per la likelihood attuale
        self.value_label = ttk.Label(info_frame, text="Likelihood: 0.0%", font=("Arial", 9, "bold"))
        self.value_label.pack(side="left")
        
        self.threshold_label = ttk.Label(info_frame, text="Threshold: 60%", font=("Arial", 9))
        self.threshold_label.pack(side="right")
    
    def update_confidence(self, confidence: float):
        """Aggiorna il valore di Likelihood attuale."""
        self.confidence_var.set(confidence * 100)
        self.value_label.config(text=f"Likelihood: {confidence:.1%}")
        
        threshold = self.threshold_var.get()
        if confidence >= threshold:
            self.status_label.config(text="●", foreground="green")
        elif confidence >= threshold * 0.75:
            self.status_label.config(text="●", foreground="yellow")
        else:
            self.status_label.config(text="●", foreground="red")
    
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
            bg="lightgray",
            text="No Album",
            compound="center",
            font=("Arial", 10)
        )
        self._create_placeholder()
    
    def _create_placeholder(self):
        placeholder = Image.new('RGB', self.size, color='lightgray')
        draw = ImageDraw.Draw(placeholder)
        center_x, center_y = self.size[0] // 2, self.size[1] // 2
        radius = min(self.size) // 3
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], outline='gray', width=2)
        inner_radius = radius // 6
        draw.ellipse([center_x - inner_radius, center_y - inner_radius, center_x + inner_radius, center_y + inner_radius], outline='gray', width=2)
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
                self.configure(image=self.placeholder_image, text="No Album")
        except Exception as e:
            print(f"Error updating album cover: {e}")
            self.configure(image=self.placeholder_image, text="Error Loading")


class StatusIndicator(ttk.Frame):
    """System status indicator with multiple states."""
    
    def __init__(self, parent, **kwargs):
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
        self.indicator = ttk.Label(self, text="●", font=("Arial", 12))
        self.indicator.pack(side="left", padx=(0, 5))
        self.status_label = ttk.Label(self, text="Disconnected", font=("Arial", 9))
        self.status_label.pack(side="left")
        self.set_state('disconnected')
    
    def set_state(self, state: str, custom_text: str = None):
        if state not in self.states:
            return
        self.current_state = state
        state_info = self.states[state]
        self.indicator.config(foreground=state_info['color'])
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
        ttk.Label(self, text="FPS:", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
        self.fps_label = ttk.Label(self, text="0.0", font=("Arial", 9, "bold"))
        self.fps_label.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(self, text="Process:", font=("Arial", 9)).grid(row=1, column=0, sticky="w")
        self.process_label = ttk.Label(self, text="0ms", font=("Arial", 9, "bold"))
        self.process_label.grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(self, text="Memory:", font=("Arial", 9)).grid(row=2, column=0, sticky="w")
        self.memory_label = ttk.Label(self, text="0MB", font=("Arial", 9, "bold"))
        self.memory_label.grid(row=2, column=1, sticky="w", padx=(5, 0))
        
        ttk.Label(self, text="Queue:", font=("Arial", 9)).grid(row=3, column=0, sticky="w")
        self.queue_label = ttk.Label(self, text="0", font=("Arial", 9, "bold"))
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
    """Visualizzatore spettro sonoro FFT in stile barre equalizzatore retro."""
    def __init__(self, parent, width=300, height=80, **kwargs):
        super().__init__(parent, width=width, height=height, bg="#111115", highlightthickness=0, **kwargs)
        self.w = width
        self.h = height

    def render_spectrum(self, fft_bands: np.ndarray):
        self.delete("all")
        n_bars = len(fft_bands)
        if n_bars == 0:
            return
            
        bar_width = (self.w - (n_bars * 2)) / n_bars
        
        for i, val in enumerate(fft_bands):
            x0 = i * (bar_width + 2) + 2
            x1 = x0 + bar_width
            bar_height = max(3, int(val * (self.h - 10)))
            y0 = self.h - bar_height
            y1 = self.h
            
            # Gradiente cromatico: Verde -> Giallo -> Rosso sui picchi
            color = "#00FF66" if val < 0.6 else ("#FFFF00" if val < 0.85 else "#FF0055")
            self.create_rectangle(x0, y0, x1, y1, fill=color, outline="")


class LyricsDisplay(ttk.Frame):
    """Visualizzatore testi a 7 righe e Progress Bar stile Spotify con centratura verticale."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.configure(padding=10)
        
        # 1. Header traccia
        self.track_header = tk.Label(
            self, 
            text="TESTI SINCRONIZZATI", 
            font=("Helvetica", 9, "bold"), 
            fg="#1DB954",  # Spotify Green
            bg="#1A1A1E"
        )
        self.track_header.pack(anchor="w", pady=(0, 4))

        # 2. Contenitore testi: occupa tutto lo spazio disponibile per centrare verticalmente
        self.lines_container = tk.Frame(self, bg="#1A1A1E")
        self.lines_container.pack(fill=tk.BOTH, expand=True)

        # 7 Righe con dissolvenza progressiva verso i bordi
        # Riga -3 (molto sfumata)
        self.lbl_p3 = tk.Label(self.lines_container, text="", font=("Helvetica", 9), fg="#303038", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_p3.pack(fill=tk.BOTH, expand=True, pady=1)

        # Riga -2 (sfumata)
        self.lbl_p2 = tk.Label(self.lines_container, text="", font=("Helvetica", 11), fg="#484855", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_p2.pack(fill=tk.BOTH, expand=True, pady=1)

        # Riga -1 (quasi attiva)
        self.lbl_p1 = tk.Label(self.lines_container, text="", font=("Helvetica", 13), fg="#757585", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_p1.pack(fill=tk.BOTH, expand=True, pady=2)

        # Riga 0 (ATTIVA / NOW SINGING)
        self.lbl_curr = tk.Label(self.lines_container, text="In attesa del brano...", font=("Helvetica", 16, "bold"), fg="#FFFFFF", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_curr.pack(fill=tk.BOTH, expand=True, pady=3)

        # Riga +1 (prossima riga)
        self.lbl_n1 = tk.Label(self.lines_container, text="", font=("Helvetica", 13), fg="#757585", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_n1.pack(fill=tk.BOTH, expand=True, pady=2)

        # Riga +2 (successiva)
        self.lbl_n2 = tk.Label(self.lines_container, text="", font=("Helvetica", 11), fg="#484855", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_n2.pack(fill=tk.BOTH, expand=True, pady=1)

        # Riga +3 (molto sfumata)
        self.lbl_n3 = tk.Label(self.lines_container, text="", font=("Helvetica", 9), fg="#303038", bg="#1A1A1E", anchor="w", wraplength=440, justify="left")
        self.lbl_n3.pack(fill=tk.BOTH, expand=True, pady=1)

        # 3. Spotify Progress Bar & Timers
        self.progress_container = tk.Frame(self, bg="#1A1A1E")
        self.progress_container.pack(fill=tk.X, pady=(6, 0))

        self.time_cur_lbl = tk.Label(self.progress_container, text="0:00", font=("Helvetica", 9), fg="#A0A0A8", bg="#1A1A1E", width=5, anchor="w")
        self.time_cur_lbl.pack(side=tk.LEFT)

        self.prog_canvas = tk.Canvas(self.progress_container, bg="#33333A", height=5, highlightthickness=0)
        self.prog_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.prog_bar = self.prog_canvas.create_rectangle(0, 0, 0, 5, fill="#1DB954", width=0)

        self.time_tot_lbl = tk.Label(self.progress_container, text="--:--", font=("Helvetica", 9), fg="#A0A0A8", bg="#1A1A1E", width=5, anchor="e")
        self.time_tot_lbl.pack(side=tk.RIGHT)

    @staticmethod
    def _format_time(seconds: float) -> str:
        s = max(0, int(seconds))
        m = s // 60
        sec = s % 60
        return f"{m}:{sec:02d}"

    def update_progress(self, current_sec: float, total_sec: float):
        """Aggiorna la barra e i timestamp di riproduzione."""
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
        """Aggiorna le 7 righe contemporaneamente."""
        if track_name:
            self.track_header.config(text=f"TESTI • {track_name.upper()}")
            
        self.lbl_p3.config(text=p3 or "")
        self.lbl_p2.config(text=p2 or "")
        self.lbl_p1.config(text=p1 or "")
        self.lbl_curr.config(text=curr or ("..." if track_name else "In attesa del brano..."))
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