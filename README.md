# VinylVision 🎵💿

**Smart Vinyl Companion: Real-Time Cover Recognition, Shazam Audio Sync, Live Lyrics & Discogs Marketplace Intelligence**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UI](https://img.shields.io/badge/Interface-Kiosk%20Hi--Fi-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Fork-orange.svg)]()

VinylVision transforms your record listening experience into an interactive, smart Hi-Fi hub. By combining **Computer Vision (EfficientNet)**, an **interactive 4-point perspective warp calibration**, acoustic fingerprinting with **Shazam**, real-time **synchronized 7-line lyrics** (Spotify-style), and live **Discogs Marketplace valuations**, VinylVision is the ultimate companion for vinyl enthusiasts, audiophiles, and collectors.

---

## ✨ Features

### 👁️ Computer Vision & Album Detection
- **Instant Album Identification**: Recognizes vinyl jackets in milliseconds using **EfficientNet-B0** feature extraction combined with vector similarity search.
- **Interactive 4-Point Perspective Calibration**: Dedicated calibration screen allowing you to drag 4 corner pins to map the exact vinyl stand/turntable area, correcting camera tilt and perspective distortions.
- **Auto-Persistence**: Geometric homography matrix and warp coordinates are saved locally (`calibration.npy`) across sessions.

### 🎙️ Audio Fingerprinting & Live Track Sync
- **Shazam Audio Engine**: Captures ambient audio via microphone and fingerprints the playing track.
- **Accurate Offset & Latency Tracking**: Computes recording and network latency offsets so that playback tracking matches the needle position even when recognized midway through a song.
- **Real-Time Duration Parsing**: Extracts exact track lengths from audio metadata and instant lookup fallbacks.

### 📝 Spotify-Style 7-Line Lyrics & Progress Bar
- **Dynamic 7-Line Display**: Smooth, vertically centered scrolling view featuring 3 previous lines, 1 highlighted active singing line, and 3 upcoming lines with progressive fading.
- **Live Progress Bar**: Spotify-styled green playback bar with real-time minute/second counters (`mm:ss`) synchronized with the vinyl needle.
- **Integrated FFT Audio Spectrum**: Embedded real-time audio visualizer displaying frequency spectrum bands.

### 🏷️ Discogs Database & Marketplace Intelligence
- **Comprehensive Metadata**: Album title, artist, release year, record label, catalog number, and genres.
- **Marketplace Valuations**: Real-time authenticated queries fetching active copies for sale (`num_for_sale`) alongside **Minimum, Median, and Maximum** market prices in EUR/USD/GBP.
- **Collection Synchronizer**: CLI tools to bulk sync your entire Discogs collection and pre-compute embeddings offline.

---

## 🛠️ Technology Stack

- **Computer Vision & Image Processing**: OpenCV (`cv2`), PIL/Pillow
- **Deep Learning / Embeddings**: PyTorch, EfficientNet-B0, ChromaDB / Vector Search
- **Audio Recognition & Lyrics**: `shazamio`, `sounddevice`, `numpy`, LRCLIB / iTunes Metadata
- **Metadata & Marketplace**: Discogs REST API v2.0
- **Graphical Interface**: Custom Tkinter / TTK Dark Theme with Canvas hardware acceleration
- **Platform Support**: Windows, macOS, Linux (Raspberry Pi compatible)

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 to 3.11
- USB Webcam / Camera module
- Microphone / Audio input device
- FFmpeg installed and available in system PATH (required for audio conversion)
  - Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  - Linux / Raspberry Pi: `sudo apt update && sudo apt install -y ffmpeg`
  - macOS: `brew install ffmpeg`
- Discogs Developer Account (Consumer Key & Secret)

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/Gilmour93/VinylVision.git](https://github.com/Gilmour93/VinylVision.git)
cd VinylVision

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize configuration file
Copy the example configuration file:
```bash
cp config/config.example.py config/config.py
```

### 3. Discogs API Setup

1. Create a [Discogs developer account](https://www.discogs.com/developers/)
2. Register your application to get Consumer Key and Secret
3. Add credentials to `config/config.py`, modifying:
```
DISCOGS_KEY = "YOUR_DISCOGS_CONSUMER_KEY"
DISCOGS_SECRET = "YOUR_DISCOGS_CONSUMER_SECRET"
```

## 🚀 Usage

1. **Launch Application**: Run `python src/main.py`

2. **Camera Stand Calibration** (First Time Setup)

   1. In the app, click ⚙ Settings (or press the C key).
   2. The live camera feed will show 4 colored draggable pins.
   3. Click Salva Calibrazione to persist the geometric matrix.
   4. Click Dashboard to return to the live Now Playing view.

3. **Collection Sync & Album Management**
   * Sync your personal Discogs record collection into the local database:
      ```bash
      # Sync entire collection from Discogs
      python src/sync_collection.py <YOUR_DISCOGS_USERNAME>

      # Sync only the first N albums (e.g., first 10)
      python src/sync_collection.py <YOUR_DISCOGS_USERNAME> <ALBUM_LIMIT>

      # Add a single album manually
      python src/add_album.py --id <DISCOGS_RELEASE_ID>
      ```
   * Add a single album manually: If you only want to index a specific record without syncing the whole collection, use src/add_album.py with its Discogs Release ID (found in the album's Discogs web URL, e.g., discogs.com/release/249504):
      ```bash
      # Ingest a single release via Discogs ID
      python src/add_album.py --id <DISCOGS_RELEASE_ID>
      ```
   **Note**: The visual cover recognition works exclusively with albums indexed in your local database. Make sure to sync your collection or add individual releases before scanning!

### Tips for Best Results
- Ensure good lighting conditions
- Avoid reflections and glare
- Keep album cover unobstructed

## 🏗️ Project Structure

```
VinylVision/
├── config/
│   ├── config.example.py        # Clean template configuration
│   └── config.py                # Local configuration & API tokens (ignored in git)
├── data/
│   ├── covers/                  # Local album artwork cache (ignored in git)
│   └── embeddings/              # ChromaDB vector index & embeddings (ignored in git)
├── src/
│   ├── main.py                  # Kiosk entry point
│   ├── core/
│   │   ├── album_pipeline.py    # Recognition coordinator
│   │   ├── audio_engine.py      # Shazam recognition, offset sync & 7-line lyrics
│   │   ├── camera.py            # Low-latency camera capture thread
│   │   ├── database.py          # Vector store operations
│   │   ├── discogs_client.py    # Discogs API client & marketplace stats
│   │   └── vision.py            # EfficientNet inference & embedding matching
│   ├── models/
│   │   └── efficientnet.py      # Feature extraction architecture
│   ├── ui/
│   │   ├── kiosk_window.py      # Dual-card Now Playing dashboard & settings view
│   │   └── widgets.py           # 7-line lyrics display, Spotify progress bar & FFT visualizer
│   ├── utils/
│   │   ├── config.py            # Config loader
│   │   └── image_processing.py  # Perspective warp, transforms and image resizing
│   ├── add_album.py             # Single album ingestion script
│   └── sync_collection.py       # Full collection batch syncer
├── .gitignore                   # Ignored local caches, credentials and environments
├── LICENSE                      # MIT License
├── requirements.txt             # Python packages
└── README.md                    # Project documentation
```

## 🤝 Attribution & Acknowledgments

This project is an enhanced fork of the original work by [pmoneynz/VinylVision](https://github.com/pmoneynz/VinylVision).
- Discogs: For their extensive vinyl release database and marketplace API.
- Shazam / Shazamio: For audio fingerprinting and track offset matching.
- PyTorch & EfficientNet: For real-time visual feature extraction.


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
