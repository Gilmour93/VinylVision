"""
UI module for VinylVision.
"""

from .kiosk_window import KioskVinylVisionWindow
from .widgets import LyricsDisplay, AudioSpectrumVisualizer

__all__ = [
    "KioskVinylVisionWindow",
    "LyricsDisplay",
    "AudioSpectrumVisualizer"
]