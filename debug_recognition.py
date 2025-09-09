#!/usr/bin/env python3
"""
Debug Recognition Script for VinylVision

Provides live console output to debug recognition pipeline issues.
Shows detailed information about each step of the process.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from core.vision import AlbumDetector
from core.database import VectorDatabase
from models.efficientnet import AlbumFeatureExtractor
from utils.image_processing import preprocess_for_model


class DebugRecognitionSystem:
    """Debug version of recognition system with verbose logging."""
    
    def __init__(self):
        self.detector = AlbumDetector(min_area=5000)
        self.extractor = AlbumFeatureExtractor()
        self.database = VectorDatabase()
        self.frame_count = 0
        
    def initialize(self):
        """Initialize all components with debug output."""
        print("🔧 INITIALIZING VINYLVISION DEBUG SYSTEM")
        print("=" * 60)
        
        # Initialize camera
        print("📷 Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Camera failed to open")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Test camera read
        ret, frame = self.cap.read()
        if ret and frame is not None:
            print(f"✅ Camera working: {frame.shape}")
        else:
            print("⚠️ Camera opened but no frame captured")
            return False
        
        # Initialize model
        print("🤖 Loading EfficientNet model...")
        if self.extractor.load_model():
            print("✅ Model loaded successfully")
        else:
            print("❌ Model failed to load")
            return False
        
        # Initialize database
        print("💾 Connecting to vector database...")
        if self.database.initialize():
            # Get database stats
            try:
                collection = self.database.collection
                count = collection.count()
                print(f"✅ Database connected: {count} embeddings available")
                
                # Show some database entries
                if count > 0:
                    sample_results = collection.peek(limit=3)
                    if sample_results and 'metadatas' in sample_results:
                        print("📊 Sample database entries:")
                        for i, metadata in enumerate(sample_results['metadatas'][:3]):
                            if metadata:
                                artist = metadata.get('artist', 'Unknown')
                                title = metadata.get('title', 'Unknown')
                                print(f"   {i+1}. {artist} - {title}")
                
            except Exception as e:
                print(f"⚠️ Database connected but stats unavailable: {e}")
        else:
            print("❌ Database failed to initialize")
            return False
        
        print("✅ All systems initialized successfully!")
        print()
        return True
    
    def process_frame_debug(self, frame):
        """Process a single frame with detailed debug output."""
        self.frame_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        print(f"\n🎬 FRAME {self.frame_count} @ {timestamp}")
        print("-" * 50)
        
        # Step 1: Album Detection
        print("🔍 Step 1: Album Detection")
        start_time = time.time()
        
        albums = self.detector.detect_albums(frame)
        detection_time = (time.time() - start_time) * 1000
        
        print(f"   ⏱️ Detection time: {detection_time:.1f}ms")
        print(f"   📦 Albums detected: {len(albums)}")
        
        if not albums:
            print("   ❌ No albums detected in frame")
            print("   💡 Try:")
            print("      - Better lighting")
            print("      - Hold album more centered")
            print("      - Ensure album is rectangular and visible")
            return None
        
        # Show detection details
        for i, album in enumerate(albums):
            x, y, w, h = album
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            print(f"   📋 Album {i+1}: pos=({x},{y}) size={w}x{h} area={area} ratio={aspect_ratio:.2f}")
        
        # Step 2: ROI Extraction
        print("✂️ Step 2: ROI Extraction")
        start_time = time.time()
        
        best_album = albums[0]  # Use first/largest detection
        roi = self.detector.extract_roi(frame, best_album)
        extraction_time = (time.time() - start_time) * 1000
        
        print(f"   ⏱️ Extraction time: {extraction_time:.1f}ms")
        
        if roi is None:
            print("   ❌ ROI extraction failed")
            return None
        
        print(f"   ✅ ROI extracted: {roi.shape}")
        
        # Step 3: Preprocessing
        print("🖼️ Step 3: Image Preprocessing")
        start_time = time.time()
        
        try:
            preprocessed = preprocess_for_model(roi, target_size=(224, 224))
            preprocessing_time = (time.time() - start_time) * 1000
            
            print(f"   ⏱️ Preprocessing time: {preprocessing_time:.1f}ms")
            print(f"   ✅ Preprocessed to: {preprocessed.shape}")
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            return None
        
        # Step 4: Feature Extraction
        print("🧠 Step 4: Feature Extraction")
        start_time = time.time()
        
        try:
            features = self.extractor.extract_features(preprocessed)
            feature_time = (time.time() - start_time) * 1000
            
            print(f"   ⏱️ Feature extraction time: {feature_time:.1f}ms")
            
            if features is None:
                print("   ❌ Feature extraction failed")
                return None
            
            print(f"   ✅ Features extracted: shape={features.shape} norm={np.linalg.norm(features):.4f}")
        except Exception as e:
            print(f"   ❌ Feature extraction error: {e}")
            return None
        
        # Step 5: Database Search
        print("🔎 Step 5: Database Search")
        start_time = time.time()
        
        try:
            search_results = self.database.search_similar(features, n_results=5)
            search_time = (time.time() - start_time) * 1000
            
            print(f"   ⏱️ Search time: {search_time:.1f}ms")
            print(f"   📊 Results found: {len(search_results) if search_results else 0}")
            
            if not search_results:
                print("   ❌ No similar albums found in database")
                print("   💡 This could mean:")
                print("      - Album not in database")
                print("      - Image quality too poor for matching")
                print("      - Feature extraction not representative")
                return None
            
            # Show top results
            print("   🏆 Top matches:")
            for i, result in enumerate(search_results[:3]):
                confidence = result.get('confidence', 0)
                metadata = result.get('metadata', {})
                artist = metadata.get('artist', 'Unknown')
                title = metadata.get('title', 'Unknown')
                print(f"      {i+1}. {confidence:.3f} - {artist} - {title}")
            
            # Check if confidence is above threshold
            top_confidence = search_results[0].get('confidence', 0)
            threshold = 0.8  # Default threshold
            
            if top_confidence < threshold:
                print(f"   ⚠️ Top confidence {top_confidence:.3f} below threshold {threshold}")
                print("   💡 Recognition uncertain - try better image quality")
            else:
                print(f"   ✅ Strong match found! Confidence: {top_confidence:.3f}")
            
            return search_results[0]
            
        except Exception as e:
            print(f"   ❌ Database search error: {e}")
            return None
    
    def run_debug_loop(self):
        """Run the debug recognition loop."""
        if not self.initialize():
            print("❌ Initialization failed")
            return
        
        print("\n🚀 STARTING DEBUG RECOGNITION LOOP")
        print("=" * 60)
        print("📋 Instructions:")
        print("   - Hold album cover in front of camera")
        print("   - Keep it well-lit and centered")
        print("   - Press 'q' to quit, 'SPACE' to pause/resume")
        print("   - Processing every 2 seconds to avoid spam")
        print()
        
        paused = False
        last_process_time = 0
        process_interval = 2.0  # Process every 2 seconds
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Show frame
                cv2.imshow('VinylVision Debug - Press Q to quit, SPACE to pause', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 User quit")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n⏸️ {'PAUSED' if paused else 'RESUMED'}")
                
                # Process frame at intervals (not paused)
                current_time = time.time()
                if not paused and (current_time - last_process_time) >= process_interval:
                    result = self.process_frame_debug(frame)
                    last_process_time = current_time
                    
                    if result:
                        print("🎉 RECOGNITION SUCCESSFUL!")
                    else:
                        print("❌ No recognition this frame")
                    
                    print(f"\n⏭️ Next processing in {process_interval} seconds...")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🧹 Cleanup completed")


def main():
    """Run the debug recognition system."""
    print("🐛 VinylVision Debug Recognition System")
    print("=" * 60)
    print("This will show detailed console output for each recognition step")
    print()
    
    debug_system = DebugRecognitionSystem()
    debug_system.run_debug_loop()


if __name__ == "__main__":
    main()
