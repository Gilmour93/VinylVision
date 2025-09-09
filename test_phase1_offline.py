#!/usr/bin/env python3
"""
Offline test script for Phase 1 Video Capture System.

This script tests the core components without requiring camera access:
- Album detection algorithms
- Feature extraction with EfficientNet
- Image preprocessing utilities
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from core.vision import AlbumDetector
from models.efficientnet import AlbumFeatureExtractor
from utils.image_processing import preprocess_for_model, resize_image, apply_noise_reduction


def test_album_detection():
    """Test album detection on synthetic images."""
    print("🔍 Testing Album Detection...")
    
    detector = AlbumDetector(min_area=5000)
    
    # Test 1: Single clear album
    test_image1 = np.ones((720, 1280, 3), dtype=np.uint8) * 50
    cv2.rectangle(test_image1, (300, 200), (500, 400), (255, 255, 255), -1)
    cv2.rectangle(test_image1, (310, 210), (490, 390), (100, 100, 100), -1)
    
    albums1 = detector.detect_albums(test_image1)
    
    # Test 2: Multiple albums
    test_image2 = np.ones((720, 1280, 3), dtype=np.uint8) * 30
    cv2.rectangle(test_image2, (200, 150), (350, 300), (200, 200, 200), -1)
    cv2.rectangle(test_image2, (500, 200), (650, 350), (180, 180, 180), -1)
    cv2.rectangle(test_image2, (800, 100), (950, 250), (220, 220, 220), -1)
    
    albums2 = detector.detect_albums(test_image2)
    
    # Test 3: Noisy image
    test_image3 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(test_image3, (400, 250), (600, 450), (255, 255, 255), -1)
    
    albums3 = detector.detect_albums(test_image3)
    
    print(f"   Test 1 (single album): Found {len(albums1)} album(s)")
    print(f"   Test 2 (multiple albums): Found {len(albums2)} album(s)")
    print(f"   Test 3 (noisy background): Found {len(albums3)} album(s)")
    
    # Test ROI extraction
    if albums1:
        roi = detector.extract_roi(test_image1, albums1[0])
        if roi is not None:
            print(f"   ROI extraction successful: {roi.shape}")
        else:
            print("   ❌ ROI extraction failed")
            return False
    
    if len(albums1) >= 1 and len(albums2) >= 2:
        print("✅ Album detection successful")
        return True
    else:
        print("❌ Album detection failed")
        return False


def test_feature_extraction():
    """Test feature extraction with EfficientNet."""
    print("\n🤖 Testing Feature Extraction...")
    
    try:
        extractor = AlbumFeatureExtractor()
        
        # Initialize the model
        print("   Loading EfficientNet model...")
        if not extractor.load_model():
            print("❌ Model loading failed")
            return False
        
        print("✅ Model loaded successfully")
        
        # Test with different image sizes
        test_cases = [
            ("Random RGB image", np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)),
            ("Solid color image", np.full((200, 200, 3), 128, dtype=np.uint8)),
            ("High contrast image", np.random.choice([0, 255], (250, 250, 3)).astype(np.uint8))
        ]
        
        extraction_times = []
        
        for test_name, test_image in test_cases:
            # Extract features multiple times for timing
            start_time = time.time()
            features = extractor.extract_features(test_image)
            extraction_time = time.time() - start_time
            extraction_times.append(extraction_time)
            
            if features is not None:
                print(f"   ✅ {test_name}: Shape {features.shape}, Norm {np.linalg.norm(features):.4f}, Time {extraction_time*1000:.1f}ms")
            else:
                print(f"   ❌ {test_name}: Feature extraction failed")
                return False
        
        # Get model info
        model_info = extractor.get_model_info()
        print(f"   Model: {model_info.get('model_name', 'Unknown')}")
        print(f"   Device: {model_info.get('device', 'Unknown')}")
        print(f"   Feature dimension: {model_info.get('feature_dim', 'Unknown')}")
        
        avg_time = np.mean(extraction_times) * 1000
        print(f"   Average extraction time: {avg_time:.1f}ms")
        
        if avg_time < 500:  # Target: <500ms
            print("✅ Feature extraction meets performance target")
            return True
        else:
            print("⚠️ Feature extraction exceeds target time but functional")
            return True
            
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False


def test_image_preprocessing():
    """Test image preprocessing utilities."""
    print("\n🖼️ Testing Image Preprocessing...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Test resize with aspect ratio preservation
        resized = resize_image(test_image, (224, 224), maintain_aspect_ratio=True)
        if resized.shape[:2] == (224, 224):
            print("   ✅ Resize with aspect ratio preservation")
        else:
            print("   ❌ Resize failed")
            return False
        
        # Test noise reduction
        denoised = apply_noise_reduction(test_image, method='bilateral')
        if denoised.shape == test_image.shape:
            print("   ✅ Noise reduction (bilateral)")
        else:
            print("   ❌ Noise reduction failed")
            return False
        
        # Test complete preprocessing pipeline
        preprocessed = preprocess_for_model(test_image, target_size=(224, 224), enhance=True)
        if preprocessed.shape[:2] == (224, 224):
            print("   ✅ Complete preprocessing pipeline")
            print(f"   Input shape: {test_image.shape} -> Output shape: {preprocessed.shape}")
        else:
            print("   ❌ Preprocessing pipeline failed")
            return False
        
        print("✅ Image preprocessing successful")
        return True
        
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")
        return False


def test_complete_pipeline():
    """Test the complete vision pipeline without camera."""
    print("\n🔄 Testing Complete Pipeline...")
    
    try:
        # Initialize components
        detector = AlbumDetector(min_area=5000)
        extractor = AlbumFeatureExtractor()
        
        if not extractor.load_model():
            print("❌ Model loading failed")
            return False
        
        # Create synthetic camera frame with album
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 40
        
        # Add realistic album cover
        album_x, album_y = 400, 200
        album_w, album_h = 200, 200
        
        # Simulate album cover with some detail
        album_roi = frame[album_y:album_y+album_h, album_x:album_x+album_w]
        album_roi[:] = np.random.randint(50, 200, album_roi.shape, dtype=np.uint8)
        
        # Add border
        cv2.rectangle(frame, (album_x, album_y), (album_x+album_w, album_y+album_h), (255, 255, 255), 2)
        
        # Time the complete pipeline
        pipeline_times = []
        
        for i in range(5):
            start_time = time.time()
            
            # Step 1: Detect albums
            albums = detector.detect_albums(frame)
            
            if albums:
                # Step 2: Extract ROI
                roi = detector.extract_roi(frame, albums[0])
                
                if roi is not None:
                    # Step 3: Preprocess
                    preprocessed = preprocess_for_model(roi, target_size=(224, 224))
                    
                    # Step 4: Extract features
                    features = extractor.extract_features(preprocessed)
                    
                    if features is not None:
                        pipeline_time = time.time() - start_time
                        pipeline_times.append(pipeline_time)
                        print(f"   Pipeline iteration {i+1}: {pipeline_time*1000:.1f}ms")
                    else:
                        print(f"   ❌ Feature extraction failed in iteration {i+1}")
                        return False
                else:
                    print(f"   ❌ ROI extraction failed in iteration {i+1}")
                    return False
            else:
                print(f"   ❌ Album detection failed in iteration {i+1}")
                return False
        
        if pipeline_times:
            avg_time = np.mean(pipeline_times) * 1000
            min_time = np.min(pipeline_times) * 1000
            max_time = np.max(pipeline_times) * 1000
            
            print(f"   Average pipeline time: {avg_time:.1f}ms")
            print(f"   Min/Max time: {min_time:.1f}ms / {max_time:.1f}ms")
            print(f"   Target <500ms: {'✅' if avg_time < 500 else '❌'}")
            
            print("✅ Complete pipeline successful")
            return True
        else:
            print("❌ No successful pipeline iterations")
            return False
            
    except Exception as e:
        print(f"❌ Complete pipeline error: {e}")
        return False


def main():
    """Run all offline Phase 1 tests."""
    print("🚀 VinylVision Phase 1 Offline Testing")
    print("=" * 50)
    
    results = {}
    
    # Test album detection
    results['detection'] = test_album_detection()
    
    # Test feature extraction
    results['extraction'] = test_feature_extraction()
    
    # Test image preprocessing
    results['preprocessing'] = test_image_preprocessing()
    
    # Test complete pipeline
    results['pipeline'] = test_complete_pipeline()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize():<15}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All Phase 1 offline tests passed!")
        print("📷 Camera testing requires hardware access")
        print("🚀 Core vision pipeline is ready for Phase 2")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
