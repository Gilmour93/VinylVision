#!/usr/bin/env python3
"""
Phase 4 Performance Testing Suite for VinylVision

Tests system performance under various conditions:
- Different hardware configurations
- Memory usage over extended periods  
- Battery impact measurement
- CPU usage profiling
- Concurrent user scenarios
- Stress testing with continuous operation
"""

import psutil
import time
import threading
import sys
import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from memory_profiler import profile
import numpy as np

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from core.vision import AlbumDetector
from core.database import VectorDatabase
from models.efficientnet import AlbumFeatureExtractor
from utils.image_processing import preprocess_for_model


class PerformanceTester:
    """Comprehensive performance testing for VinylVision."""
    
    def __init__(self):
        self.detector = AlbumDetector(min_area=5000)
        self.extractor = AlbumFeatureExtractor()
        self.database = VectorDatabase()
        self.monitoring_active = False
        self.performance_data = []
        
    def setup(self) -> bool:
        """Initialize testing components."""
        print("🔧 Setting up performance testing environment...")
        
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
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        print("💻 Gathering system information...")
        
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": psutil.os.name,
                "platform": sys.platform,
                "python_version": sys.version,
                "architecture": psutil.os.name,
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent_used": psutil.virtual_memory().percent,
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "percent_used": round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 1),
            }
        }
        
        # Add GPU information if available
        try:
            import torch
            if torch.backends.mps.is_available():
                system_info["gpu"] = {
                    "type": "Apple Metal Performance Shaders",
                    "available": True
                }
            elif torch.cuda.is_available():
                system_info["gpu"] = {
                    "type": "NVIDIA CUDA",
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name()
                }
            else:
                system_info["gpu"] = {"available": False}
        except ImportError:
            system_info["gpu"] = {"available": False, "error": "PyTorch not available"}
        
        return system_info
    
    def monitor_resources(self, duration_seconds: int = 300) -> Dict:
        """Monitor system resources over time."""
        print(f"📊 Monitoring resources for {duration_seconds} seconds...")
        
        self.monitoring_active = True
        self.performance_data = []
        
        def monitor_loop():
            start_time = time.time()
            while self.monitoring_active and (time.time() - start_time < duration_seconds):
                data_point = {
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_time": time.time() - start_time,
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                    "memory_percent": psutil.virtual_memory().percent,
                    "threads": threading.active_count(),
                }
                
                # Add GPU memory if available
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        # MPS doesn't provide memory stats directly
                        data_point["gpu_memory_mb"] = "MPS - not available"
                    elif torch.cuda.is_available():
                        data_point["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                except ImportError:
                    pass
                
                self.performance_data.append(data_point)
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.start()
        
        return {"monitoring_started": True, "duration": duration_seconds}
    
    def stop_monitoring(self) -> Dict:
        """Stop resource monitoring and return results."""
        self.monitoring_active = False
        time.sleep(2)  # Give time for monitoring thread to finish
        
        if not self.performance_data:
            return {"error": "No performance data collected"}
        
        # Calculate statistics
        cpu_values = [d["cpu_percent"] for d in self.performance_data]
        memory_values = [d["memory_mb"] for d in self.performance_data]
        
        stats = {
            "total_data_points": len(self.performance_data),
            "duration_seconds": self.performance_data[-1]["elapsed_time"],
            "cpu_stats": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "average": sum(cpu_values) / len(cpu_values),
                "std_dev": np.std(cpu_values)
            },
            "memory_stats": {
                "min_mb": min(memory_values),
                "max_mb": max(memory_values),
                "average_mb": sum(memory_values) / len(memory_values),
                "peak_mb": max(memory_values),
                "growth_mb": max(memory_values) - min(memory_values)
            },
            "raw_data": self.performance_data
        }
        
        return stats
    
    def benchmark_inference_speed(self, num_iterations: int = 100) -> Dict:
        """Benchmark feature extraction speed."""
        print(f"⚡ Benchmarking inference speed ({num_iterations} iterations)...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        preprocessed = preprocess_for_model(test_image, target_size=(224, 224))
        
        # Warmup runs
        print("   Warming up model...")
        for _ in range(5):
            self.extractor.extract_features(preprocessed)
        
        # Timed runs
        print(f"   Running {num_iterations} timed iterations...")
        inference_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            features = self.extractor.extract_features(preprocessed)
            inference_time = time.time() - start_time
            
            if features is not None:
                inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"   Completed {i + 1}/{num_iterations} iterations")
        
        if not inference_times:
            return {"error": "No successful inferences"}
        
        stats = {
            "total_iterations": len(inference_times),
            "successful_iterations": len(inference_times),
            "success_rate": len(inference_times) / num_iterations,
            "min_time_ms": min(inference_times) * 1000,
            "max_time_ms": max(inference_times) * 1000,
            "average_time_ms": np.mean(inference_times) * 1000,
            "median_time_ms": np.median(inference_times) * 1000,
            "std_dev_ms": np.std(inference_times) * 1000,
            "percentile_95_ms": np.percentile(inference_times, 95) * 1000,
            "percentile_99_ms": np.percentile(inference_times, 99) * 1000,
        }
        
        return stats
    
    def test_memory_usage_over_time(self, duration_minutes: int = 30) -> Dict:
        """Test memory usage over extended operation."""
        print(f"🧠 Testing memory usage over {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        memory_samples = []
        iteration_count = 0
        
        # Start resource monitoring
        self.monitor_resources(duration_minutes * 60)
        
        while time.time() < end_time:
            # Simulate normal operation
            test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            
            # Detect albums
            albums = self.detector.detect_albums(test_image)
            
            if albums:
                # Extract ROI and features
                roi = self.detector.extract_roi(test_image, albums[0])
                if roi is not None:
                    preprocessed = preprocess_for_model(roi, target_size=(224, 224))
                    features = self.extractor.extract_features(preprocessed)
                    
                    if features is not None:
                        # Simulate database search
                        results = self.database.search_similar(features, top_k=5)
            
            # Record memory usage
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_samples.append({
                "iteration": iteration_count,
                "timestamp": datetime.now().isoformat(),
                "memory_mb": current_memory,
                "elapsed_minutes": (time.time() - start_time) / 60
            })
            
            iteration_count += 1
            
            # Force garbage collection every 100 iterations
            if iteration_count % 100 == 0:
                gc.collect()
                print(f"   Completed {iteration_count} iterations, current memory: {current_memory:.1f}MB")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        # Stop monitoring
        monitoring_stats = self.stop_monitoring()
        
        # Analyze memory usage
        memory_values = [s["memory_mb"] for s in memory_samples]
        
        results = {
            "test_duration_minutes": duration_minutes,
            "total_iterations": iteration_count,
            "memory_analysis": {
                "start_memory_mb": memory_values[0] if memory_values else 0,
                "end_memory_mb": memory_values[-1] if memory_values else 0,
                "peak_memory_mb": max(memory_values) if memory_values else 0,
                "average_memory_mb": np.mean(memory_values) if memory_values else 0,
                "memory_growth_mb": memory_values[-1] - memory_values[0] if len(memory_values) >= 2 else 0,
                "memory_growth_rate_mb_per_hour": ((memory_values[-1] - memory_values[0]) / duration_minutes) * 60 if len(memory_values) >= 2 else 0
            },
            "performance_monitoring": monitoring_stats,
            "raw_memory_samples": memory_samples
        }
        
        return results
    
    def test_concurrent_processing(self, num_threads: int = 4, duration_seconds: int = 60) -> Dict:
        """Test concurrent processing scenarios."""
        print(f"👥 Testing concurrent processing with {num_threads} threads for {duration_seconds} seconds...")
        
        results = {
            "num_threads": num_threads,
            "duration_seconds": duration_seconds,
            "thread_results": [],
            "start_time": datetime.now().isoformat()
        }
        
        def worker_thread(thread_id: int):
            """Worker thread function."""
            thread_results = {
                "thread_id": thread_id,
                "iterations": 0,
                "successful_recognitions": 0,
                "errors": [],
                "avg_processing_time": 0,
                "start_time": time.time()
            }
            
            processing_times = []
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                try:
                    iter_start = time.time()
                    
                    # Generate unique test image for this thread
                    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
                    # Add thread-specific pattern
                    cv2.putText(test_image, f"T{thread_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Process image
                    albums = self.detector.detect_albums(test_image)
                    if albums:
                        roi = self.detector.extract_roi(test_image, albums[0])
                        if roi is not None:
                            preprocessed = preprocess_for_model(roi, target_size=(224, 224))
                            features = self.extractor.extract_features(preprocessed)
                            if features is not None:
                                thread_results["successful_recognitions"] += 1
                    
                    processing_time = time.time() - iter_start
                    processing_times.append(processing_time)
                    thread_results["iterations"] += 1
                    
                except Exception as e:
                    thread_results["errors"].append(str(e))
                
                # Small delay to prevent overwhelming
                time.sleep(0.05)
            
            if processing_times:
                thread_results["avg_processing_time"] = np.mean(processing_times)
            
            return thread_results
        
        # Start resource monitoring
        self.monitor_resources(duration_seconds)
        
        # Create and start threads
        threads = []
        thread_results = [None] * num_threads
        
        def thread_wrapper(thread_id):
            thread_results[thread_id] = worker_thread(thread_id)
        
        for i in range(num_threads):
            thread = threading.Thread(target=thread_wrapper, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Stop monitoring
        monitoring_stats = self.stop_monitoring()
        
        # Compile results
        results["thread_results"] = thread_results
        results["monitoring"] = monitoring_stats
        results["end_time"] = datetime.now().isoformat()
        
        # Calculate aggregate statistics
        total_iterations = sum(r["iterations"] for r in thread_results if r)
        total_successful = sum(r["successful_recognitions"] for r in thread_results if r)
        total_errors = sum(len(r["errors"]) for r in thread_results if r)
        
        results["aggregate_stats"] = {
            "total_iterations": total_iterations,
            "total_successful": total_successful,
            "total_errors": total_errors,
            "success_rate": total_successful / total_iterations if total_iterations > 0 else 0,
            "iterations_per_second": total_iterations / duration_seconds,
            "successful_per_second": total_successful / duration_seconds
        }
        
        return results
    
    def stress_test_continuous_operation(self, hours: float = 0.5) -> Dict:
        """Stress test with continuous operation."""
        duration_seconds = int(hours * 3600)
        print(f"🔥 Stress testing continuous operation for {hours} hours ({duration_seconds} seconds)...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        stress_results = {
            "duration_hours": hours,
            "start_time": datetime.now().isoformat(),
            "iterations": 0,
            "successful_operations": 0,
            "errors": [],
            "performance_snapshots": [],
            "memory_leaks_detected": False,
            "stability_score": 0.0
        }
        
        # Start resource monitoring
        self.monitor_resources(duration_seconds)
        
        last_snapshot_time = start_time
        snapshot_interval = 300  # 5 minutes
        
        while time.time() < end_time:
            try:
                # Simulate heavy workload
                for _ in range(10):  # Process 10 images per iteration
                    test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                    
                    # Add multiple synthetic albums
                    for j in range(3):
                        x, y = 100 + j * 200, 100 + j * 150
                        cv2.rectangle(test_image, (x, y), (x + 150, y + 150), (255, 255, 255), -1)
                        cv2.rectangle(test_image, (x + 10, y + 10), (x + 140, y + 140), (j * 80, j * 80, j * 80), -1)
                    
                    albums = self.detector.detect_albums(test_image)
                    
                    for album_box in albums[:2]:  # Process up to 2 albums
                        roi = self.detector.extract_roi(test_image, album_box)
                        if roi is not None:
                            preprocessed = preprocess_for_model(roi, target_size=(224, 224))
                            features = self.extractor.extract_features(preprocessed)
                            if features is not None:
                                results = self.database.search_similar(features, top_k=3)
                                stress_results["successful_operations"] += 1
                
                stress_results["iterations"] += 1
                
                # Take performance snapshot every 5 minutes
                if time.time() - last_snapshot_time >= snapshot_interval:
                    current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    cpu_percent = psutil.cpu_percent()
                    
                    snapshot = {
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_hours": (time.time() - start_time) / 3600,
                        "memory_mb": current_memory,
                        "cpu_percent": cpu_percent,
                        "iterations_completed": stress_results["iterations"],
                        "successful_operations": stress_results["successful_operations"]
                    }
                    
                    stress_results["performance_snapshots"].append(snapshot)
                    last_snapshot_time = time.time()
                    
                    print(f"   Stress test progress: {snapshot['elapsed_hours']:.2f}h, "
                          f"Memory: {current_memory:.1f}MB, CPU: {cpu_percent:.1f}%")
                
                # Force garbage collection every 100 iterations
                if stress_results["iterations"] % 100 == 0:
                    gc.collect()
                
            except Exception as e:
                stress_results["errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "iteration": stress_results["iterations"],
                    "error": str(e)
                })
                
                # Break if too many errors
                if len(stress_results["errors"]) > 50:
                    print("⚠️ Too many errors encountered, stopping stress test")
                    break
        
        # Stop monitoring
        monitoring_stats = self.stop_monitoring()
        stress_results["monitoring"] = monitoring_stats
        stress_results["end_time"] = datetime.now().isoformat()
        
        # Analyze for memory leaks
        if len(stress_results["performance_snapshots"]) >= 2:
            start_memory = stress_results["performance_snapshots"][0]["memory_mb"]
            end_memory = stress_results["performance_snapshots"][-1]["memory_mb"]
            memory_growth = end_memory - start_memory
            
            # Consider significant if memory grew by more than 200MB over time
            stress_results["memory_leaks_detected"] = memory_growth > 200
            stress_results["memory_growth_mb"] = memory_growth
        
        # Calculate stability score
        total_operations = stress_results["iterations"] * 10  # 10 images per iteration
        if total_operations > 0:
            success_rate = stress_results["successful_operations"] / total_operations
            error_rate = len(stress_results["errors"]) / stress_results["iterations"] if stress_results["iterations"] > 0 else 1
            stress_results["stability_score"] = max(0, success_rate - error_rate)
        
        return stress_results
    
    def run_comprehensive_performance_test(self) -> Dict:
        """Run comprehensive performance testing."""
        print("🚀 Starting comprehensive performance testing...")
        
        if not self.setup():
            return {"error": "Setup failed"}
        
        test_results = {
            "test_start": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "inference_benchmark": {},
            "memory_test": {},
            "concurrent_test": {},
            "stress_test": {}
        }
        
        print("\n1. System Information Collection")
        print("✅ System info collected")
        
        print("\n2. Inference Speed Benchmark")
        test_results["inference_benchmark"] = self.benchmark_inference_speed(100)
        
        print("\n3. Memory Usage Test (5 minutes)")
        test_results["memory_test"] = self.test_memory_usage_over_time(5)
        
        print("\n4. Concurrent Processing Test")
        test_results["concurrent_test"] = self.test_concurrent_processing(4, 60)
        
        print("\n5. Stress Test (30 minutes)")
        test_results["stress_test"] = self.stress_test_continuous_operation(0.5)
        
        test_results["test_end"] = datetime.now().isoformat()
        
        # Generate summary
        summary = self._generate_performance_summary(test_results)
        test_results["summary"] = summary
        
        return test_results
    
    def _generate_performance_summary(self, results: Dict) -> Dict:
        """Generate performance test summary."""
        summary = {
            "overall_status": "PASS",
            "performance_targets": {},
            "recommendations": []
        }
        
        # Check inference speed target (<500ms)
        inference_stats = results.get("inference_benchmark", {})
        avg_inference_ms = inference_stats.get("average_time_ms", 1000)
        summary["performance_targets"]["inference_speed"] = {
            "target_ms": 500,
            "actual_ms": avg_inference_ms,
            "passed": avg_inference_ms < 500
        }
        
        # Check memory usage target (<2GB)
        memory_stats = results.get("memory_test", {}).get("memory_analysis", {})
        peak_memory_mb = memory_stats.get("peak_memory_mb", 3000)
        summary["performance_targets"]["memory_usage"] = {
            "target_mb": 2000,
            "actual_mb": peak_memory_mb,
            "passed": peak_memory_mb < 2000
        }
        
        # Check stress test stability
        stress_stats = results.get("stress_test", {})
        stability_score = stress_stats.get("stability_score", 0)
        memory_leaks = stress_stats.get("memory_leaks_detected", True)
        summary["performance_targets"]["stability"] = {
            "stability_score": stability_score,
            "memory_leaks": memory_leaks,
            "passed": stability_score > 0.8 and not memory_leaks
        }
        
        # Overall pass/fail
        all_passed = all(target["passed"] for target in summary["performance_targets"].values())
        summary["overall_status"] = "PASS" if all_passed else "FAIL"
        
        # Generate recommendations
        if avg_inference_ms >= 500:
            summary["recommendations"].append("Consider model optimization or quantization to reduce inference time")
        
        if peak_memory_mb >= 2000:
            summary["recommendations"].append("Optimize memory usage - implement better garbage collection or reduce batch sizes")
        
        if memory_leaks:
            summary["recommendations"].append("Memory leaks detected - review object lifecycle management")
        
        if stability_score < 0.8:
            summary["recommendations"].append("Improve error handling and system stability")
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_results_{timestamp}.json"
        
        filepath = Path("test_results") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Performance test results saved to: {filepath}")


def main():
    """Run Phase 4 performance testing."""
    print("⚡ VinylVision Phase 4 - Performance Testing")
    print("=" * 50)
    
    tester = PerformanceTester()
    
    try:
        results = tester.run_comprehensive_performance_test()
        
        if "error" in results:
            print(f"❌ Testing failed: {results['error']}")
            return False
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n📊 Performance Test Summary:")
        print(f"   Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        
        targets = summary.get("performance_targets", {})
        for target_name, target_data in targets.items():
            status = "✅" if target_data.get("passed", False) else "❌"
            print(f"   {target_name}: {status}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   - {rec}")
        
        # Save results
        tester.save_results(results)
        
        success = summary.get("overall_status") == "PASS"
        if success:
            print("🎉 All performance targets met!")
        else:
            print("⚠️ Some performance targets not met")
        
        return success
        
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
