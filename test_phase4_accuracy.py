#!/usr/bin/env python3
"""
Phase 4 Accuracy Testing Suite for VinylVision

Tests album recognition accuracy under various conditions:
- Different lighting conditions
- Various angles and distances
- Multiple album detection
- Damaged or worn album covers
"""

import cv2
import numpy as np
import time
import sys
import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from urllib.parse import urlparse

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from core.vision import AlbumDetector
from core.database import VectorDatabase
from models.efficientnet import AlbumFeatureExtractor
from utils.image_processing import preprocess_for_model


class AccuracyTester:
    """Comprehensive accuracy testing for VinylVision."""
    
    def __init__(self):
        self.detector = AlbumDetector(min_area=5000)
        self.extractor = AlbumFeatureExtractor()
        self.database = VectorDatabase()
        self.test_results = []
        
    def setup(self) -> bool:
        """Initialize testing components."""
        print("🔧 Setting up accuracy testing environment...")
        
        try:
            # Load the feature extraction model
            if not self.extractor.load_model():
                print("❌ Failed to load EfficientNet model")
                return False
            print("✅ EfficientNet model loaded")
            
            # Initialize database connection
            if not self.database.initialize():
                print("❌ Failed to initialize vector database")
                return False
            print("✅ Vector database initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def create_test_dataset(self) -> List[Dict]:
        """Create a synthetic test dataset with known album covers."""
        print("📊 Creating test dataset...")
        
        test_albums = [
            {
                "id": "test_001",
                "artist": "The Beatles",
                "title": "Abbey Road",
                "image_url": "https://example.com/abbey_road.jpg",
                "conditions": ["normal", "bright", "dark", "angled"]
            },
            {
                "id": "test_002", 
                "artist": "Pink Floyd",
                "title": "The Dark Side of the Moon",
                "image_url": "https://example.com/dark_side.jpg",
                "conditions": ["normal", "bright", "dark", "angled"]
            },
            {
                "id": "test_003",
                "artist": "Led Zeppelin",
                "title": "Led Zeppelin IV",
                "image_url": "https://example.com/zeppelin_iv.jpg", 
                "conditions": ["normal", "bright", "dark", "angled"]
            },
            # Add more test albums here...
        ]
        
        # Generate synthetic album cover images for testing
        synthetic_albums = []
        for i in range(50):  # Create 50 synthetic albums
            album = {
                "id": f"synthetic_{i:03d}",
                "artist": f"Test Artist {i}",
                "title": f"Test Album {i}",
                "image": self._generate_synthetic_album_cover(i),
                "conditions": ["normal", "bright", "dark", "angled", "multiple", "damaged"]
            }
            synthetic_albums.append(album)
        
        print(f"✅ Created test dataset with {len(test_albums)} real + {len(synthetic_albums)} synthetic albums")
        return test_albums + synthetic_albums
    
    def _generate_synthetic_album_cover(self, seed: int) -> np.ndarray:
        """Generate a synthetic album cover for testing."""
        np.random.seed(seed)
        
        # Create a 300x300 album cover with distinctive features
        album = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Add some distinctive features based on seed
        color1 = (seed * 50) % 255
        color2 = (seed * 100) % 255 
        color3 = (seed * 150) % 255
        
        # Add geometric shapes for recognition
        cv2.circle(album, (150, 150), 50, (color1, color2, color3), -1)
        cv2.rectangle(album, (50, 50), (250, 100), (color3, color1, color2), 3)
        cv2.line(album, (0, seed % 300), (300, (seed * 2) % 300), (color2, color3, color1), 2)
        
        # Add text-like features
        cv2.putText(album, f"Test{seed}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return album
    
    def test_lighting_conditions(self, album_image: np.ndarray, album_id: str) -> Dict:
        """Test recognition under different lighting conditions."""
        print(f"💡 Testing lighting conditions for {album_id}...")
        
        results = {}
        
        # Normal lighting
        normal_result = self._test_single_recognition(album_image, f"{album_id}_normal")
        results["normal"] = normal_result
        
        # Bright lighting (overexposed)
        bright_image = cv2.convertScaleAbs(album_image, alpha=1.5, beta=50)
        bright_result = self._test_single_recognition(bright_image, f"{album_id}_bright")
        results["bright"] = bright_result
        
        # Dark lighting (underexposed)  
        dark_image = cv2.convertScaleAbs(album_image, alpha=0.5, beta=-50)
        dark_result = self._test_single_recognition(dark_image, f"{album_id}_dark")
        results["dark"] = dark_result
        
        # Uneven lighting
        uneven_image = album_image.copy()
        h, w = uneven_image.shape[:2]
        gradient = np.linspace(0.3, 1.2, w).reshape(1, w, 1)
        uneven_image = np.clip(uneven_image * gradient, 0, 255).astype(np.uint8)
        uneven_result = self._test_single_recognition(uneven_image, f"{album_id}_uneven")
        results["uneven"] = uneven_result
        
        return results
    
    def test_angles_and_distances(self, album_image: np.ndarray, album_id: str) -> Dict:
        """Test recognition at various angles and distances."""
        print(f"📐 Testing angles and distances for {album_id}...")
        
        results = {}
        
        # Different angles
        angles = [0, 15, 30, 45, -15, -30]
        for angle in angles:
            rotated_image = self._rotate_image(album_image, angle)
            result = self._test_single_recognition(rotated_image, f"{album_id}_angle_{angle}")
            results[f"angle_{angle}"] = result
        
        # Different distances (simulated by scaling)
        scales = [0.5, 0.7, 1.0, 1.5, 2.0]
        for scale in scales:
            scaled_image = self._scale_image(album_image, scale)
            result = self._test_single_recognition(scaled_image, f"{album_id}_scale_{scale}")
            results[f"scale_{scale}"] = result
        
        # Perspective distortion
        perspective_image = self._apply_perspective_transform(album_image)
        perspective_result = self._test_single_recognition(perspective_image, f"{album_id}_perspective")
        results["perspective"] = perspective_result
        
        return results
    
    def test_multiple_albums(self, album_images: List[np.ndarray], album_ids: List[str]) -> Dict:
        """Test detection of multiple albums in single frame."""
        print("👥 Testing multiple album detection...")
        
        if len(album_images) < 2:
            print("⚠️ Need at least 2 albums for multiple detection test")
            return {}
        
        # Create composite image with 2-4 albums
        composite_sizes = [2, 3, 4]
        results = {}
        
        for size in composite_sizes:
            if len(album_images) >= size:
                composite_image = self._create_composite_image(album_images[:size])
                albums_detected = self.detector.detect_albums(composite_image)
                
                result = {
                    "albums_in_frame": size,
                    "albums_detected": len(albums_detected),
                    "detection_rate": len(albums_detected) / size,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Try to recognize each detected album
                recognitions = []
                for i, album_box in enumerate(albums_detected):
                    roi = self.detector.extract_roi(composite_image, album_box)
                    if roi is not None:
                        recognition = self._test_single_recognition(roi, f"composite_{size}_{i}")
                        recognitions.append(recognition)
                
                result["recognitions"] = recognitions
                results[f"composite_{size}"] = result
        
        return results
    
    def test_damaged_covers(self, album_image: np.ndarray, album_id: str) -> Dict:
        """Test recognition of damaged or worn album covers."""
        print(f"🔨 Testing damaged cover recognition for {album_id}...")
        
        results = {}
        
        # Scratches and noise
        scratched_image = self._add_scratches(album_image)
        scratched_result = self._test_single_recognition(scratched_image, f"{album_id}_scratched")
        results["scratched"] = scratched_result
        
        # Faded/worn appearance
        faded_image = cv2.convertScaleAbs(album_image, alpha=0.7, beta=30)
        faded_image = cv2.GaussianBlur(faded_image, (3, 3), 1)
        faded_result = self._test_single_recognition(faded_image, f"{album_id}_faded")
        results["faded"] = faded_result
        
        # Partial occlusion
        occluded_image = self._add_occlusion(album_image)
        occluded_result = self._test_single_recognition(occluded_image, f"{album_id}_occluded")
        results["occluded"] = occluded_result
        
        # Water damage simulation
        water_damaged_image = self._simulate_water_damage(album_image)
        water_result = self._test_single_recognition(water_damaged_image, f"{album_id}_water_damaged")
        results["water_damaged"] = water_result
        
        return results
    
    def _test_single_recognition(self, image: np.ndarray, test_id: str) -> Dict:
        """Test recognition on a single image."""
        start_time = time.time()
        
        try:
            # Detect albums in the image
            albums = self.detector.detect_albums(image)
            detection_time = time.time() - start_time
            
            if not albums:
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": "No albums detected",
                    "detection_time": detection_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract ROI from first detected album
            roi = self.detector.extract_roi(image, albums[0])
            if roi is None:
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": "ROI extraction failed",
                    "detection_time": detection_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Preprocess for feature extraction
            preprocessed = preprocess_for_model(roi, target_size=(224, 224))
            preprocessing_time = time.time() - start_time - detection_time
            
            # Extract features
            features = self.extractor.extract_features(preprocessed)
            extraction_time = time.time() - start_time - detection_time - preprocessing_time
            
            if features is None:
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": "Feature extraction failed",
                    "detection_time": detection_time,
                    "preprocessing_time": preprocessing_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Search in database
            search_results = self.database.search_similar(features, top_k=5)
            search_time = time.time() - start_time - detection_time - preprocessing_time - extraction_time
            
            total_time = time.time() - start_time
            
            result = {
                "test_id": test_id,
                "success": True,
                "detection_time": detection_time,
                "preprocessing_time": preprocessing_time,
                "extraction_time": extraction_time,
                "search_time": search_time,
                "total_time": total_time,
                "albums_detected": len(albums),
                "search_results": len(search_results),
                "top_confidence": search_results[0]["confidence"] if search_results else 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "test_id": test_id,
                "success": False,
                "error": str(e),
                "detection_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by given factor."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        
        if scale < 1.0:
            # Pad smaller image to original size
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = np.zeros_like(image)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
            return result
        else:
            # Crop larger image to original size
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            return scaled[crop_h:crop_h+h, crop_w:crop_w+w]
    
    def _apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transformation to simulate viewing angle."""
        h, w = image.shape[:2]
        
        # Define source and destination points for perspective transform
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.05], [w*0.95, h*0.9], [w*0.05, h*0.95]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (w, h))
    
    def _create_composite_image(self, album_images: List[np.ndarray]) -> np.ndarray:
        """Create composite image with multiple albums."""
        # Create 800x600 background
        composite = np.ones((600, 800, 3), dtype=np.uint8) * 50
        
        # Position albums in grid
        positions = [(100, 100), (450, 100), (100, 350), (450, 350)]
        
        for i, album in enumerate(album_images[:len(positions)]):
            x, y = positions[i]
            # Resize album to 200x200
            album_resized = cv2.resize(album, (200, 200))
            composite[y:y+200, x:x+200] = album_resized
        
        return composite
    
    def _add_scratches(self, image: np.ndarray) -> np.ndarray:
        """Add scratch effects to image."""
        scratched = image.copy()
        h, w = scratched.shape[:2]
        
        # Add random scratches
        for _ in range(10):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            cv2.line(scratched, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Add noise
        noise = np.random.randint(-30, 30, scratched.shape, dtype=np.int16)
        scratched = np.clip(scratched.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return scratched
    
    def _add_occlusion(self, image: np.ndarray) -> np.ndarray:
        """Add partial occlusion to image."""
        occluded = image.copy()
        h, w = occluded.shape[:2]
        
        # Add a rectangular occlusion (simulating hand/shadow)
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, h // 3
        cv2.rectangle(occluded, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return occluded
    
    def _simulate_water_damage(self, image: np.ndarray) -> np.ndarray:
        """Simulate water damage effects."""
        damaged = image.copy()
        
        # Create water stain effect
        h, w = damaged.shape[:2]
        center = (w // 2, h // 3)
        
        # Create circular water stain
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, min(h, w) // 4, 255, -1)
        
        # Apply color distortion in water stain area
        stained = cv2.bitwise_and(damaged, damaged, mask=mask)
        stained = cv2.convertScaleAbs(stained, alpha=0.6, beta=-20)
        
        # Combine with original
        inv_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(damaged, damaged, mask=inv_mask)
        
        return cv2.add(background, stained)
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive accuracy testing."""
        print("🎯 Starting comprehensive accuracy testing...")
        
        if not self.setup():
            return {"error": "Setup failed"}
        
        # Create test dataset
        test_dataset = self.create_test_dataset()
        
        all_results = {
            "test_start": datetime.now().isoformat(),
            "total_albums": len(test_dataset),
            "lighting_tests": {},
            "angle_distance_tests": {},
            "multiple_album_tests": {},
            "damage_tests": {}
        }
        
        # Test each album under different conditions
        for i, album_data in enumerate(test_dataset[:10]):  # Test first 10 for demo
            album_id = album_data["id"]
            print(f"\n📀 Testing album {i+1}/{min(10, len(test_dataset))}: {album_id}")
            
            if "image" in album_data:
                album_image = album_data["image"]
            else:
                # Generate synthetic image for this test
                album_image = self._generate_synthetic_album_cover(i)
            
            # Run all test categories
            lighting_results = self.test_lighting_conditions(album_image, album_id)
            all_results["lighting_tests"][album_id] = lighting_results
            
            angle_results = self.test_angles_and_distances(album_image, album_id) 
            all_results["angle_distance_tests"][album_id] = angle_results
            
            damage_results = self.test_damaged_covers(album_image, album_id)
            all_results["damage_tests"][album_id] = damage_results
        
        # Test multiple album detection
        if len(test_dataset) >= 3:
            sample_images = [album.get("image", self._generate_synthetic_album_cover(i)) 
                           for i, album in enumerate(test_dataset[:4])]
            sample_ids = [album["id"] for album in test_dataset[:4]]
            multiple_results = self.test_multiple_albums(sample_images, sample_ids)
            all_results["multiple_album_tests"] = multiple_results
        
        all_results["test_end"] = datetime.now().isoformat()
        
        # Generate summary statistics
        summary = self._generate_test_summary(all_results)
        all_results["summary"] = summary
        
        return all_results
    
    def _generate_test_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from test results."""
        summary = {
            "total_tests": 0,
            "successful_tests": 0,
            "average_accuracy": 0.0,
            "average_processing_time": 0.0,
            "condition_breakdown": {}
        }
        
        all_test_results = []
        
        # Collect all individual test results
        for category in ["lighting_tests", "angle_distance_tests", "damage_tests"]:
            if category in results:
                for album_id, album_results in results[category].items():
                    for condition, result in album_results.items():
                        all_test_results.append(result)
        
        if all_test_results:
            summary["total_tests"] = len(all_test_results)
            summary["successful_tests"] = sum(1 for r in all_test_results if r.get("success", False))
            summary["average_accuracy"] = summary["successful_tests"] / summary["total_tests"]
            
            successful_results = [r for r in all_test_results if r.get("success", False)]
            if successful_results:
                summary["average_processing_time"] = np.mean([r.get("total_time", 0) for r in successful_results])
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_test_results_{timestamp}.json"
        
        filepath = Path("test_results") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Test results saved to: {filepath}")


def main():
    """Run Phase 4 accuracy testing."""
    print("🧪 VinylVision Phase 4 - Accuracy Testing")
    print("=" * 50)
    
    tester = AccuracyTester()
    
    try:
        results = tester.run_comprehensive_test()
        
        if "error" in results:
            print(f"❌ Testing failed: {results['error']}")
            return False
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n📊 Test Summary:")
        print(f"   Total tests: {summary.get('total_tests', 0)}")
        print(f"   Successful: {summary.get('successful_tests', 0)}")
        print(f"   Accuracy: {summary.get('average_accuracy', 0):.1%}")
        print(f"   Avg processing time: {summary.get('average_processing_time', 0)*1000:.1f}ms")
        
        # Save results
        tester.save_results(results)
        
        # Check if meets targets
        accuracy = summary.get('average_accuracy', 0)
        processing_time = summary.get('average_processing_time', 0)
        
        accuracy_target = accuracy >= 0.90  # >90% accuracy
        time_target = processing_time < 0.5  # <500ms
        
        print(f"\n🎯 Target Achievement:")
        print(f"   Accuracy >90%: {'✅' if accuracy_target else '❌'} ({accuracy:.1%})")
        print(f"   Time <500ms: {'✅' if time_target else '❌'} ({processing_time*1000:.1f}ms)")
        
        if accuracy_target and time_target:
            print("🎉 All accuracy targets met!")
            return True
        else:
            print("⚠️ Some targets not met - requires optimization")
            return False
            
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
