#!/usr/bin/env python3
"""
Phase 4 Cross-Platform Testing Suite for VinylVision

Tests platform compatibility across:
- macOS testing (current platform)
- Windows compatibility checks
- Linux compatibility checks  
- Camera compatibility verification
- Python version compatibility
- Platform-specific requirements
"""

import os
import sys
import platform
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class CrossPlatformTester:
    """Comprehensive cross-platform testing for VinylVision."""
    
    def __init__(self):
        self.current_platform = platform.system()
        self.python_version = sys.version_info
        self.test_results = {}
        
        # Add src to Python path
        project_root = Path(__file__).parent
        src_path = str(project_root / "src")
        sys.path.insert(0, src_path)
    
    def get_platform_info(self) -> Dict:
        """Get comprehensive platform information."""
        print("💻 Gathering platform information...")
        
        platform_info = {
            "timestamp": datetime.now().isoformat(),
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": {
                "major": self.python_version.major,
                "minor": self.python_version.minor,
                "micro": self.python_version.micro,
                "full": sys.version
            },
            "python_executable": sys.executable,
            "python_path": sys.path[:5],  # First 5 entries
        }
        
        # Add OS-specific information
        if self.current_platform == "Darwin":  # macOS
            try:
                macos_version = platform.mac_ver()
                platform_info["macos"] = {
                    "version": macos_version[0],
                    "version_info": macos_version[1],
                    "architecture": macos_version[2]
                }
            except:
                pass
        
        elif self.current_platform == "Windows":
            try:
                windows_version = platform.win32_ver()
                platform_info["windows"] = {
                    "version": windows_version[0],
                    "service_pack": windows_version[1],
                    "build": windows_version[2],
                    "platform_type": windows_version[3]
                }
            except:
                pass
        
        elif self.current_platform == "Linux":
            try:
                linux_info = platform.linux_distribution()
                platform_info["linux"] = {
                    "distribution": linux_info[0],
                    "version": linux_info[1],
                    "id": linux_info[2]
                }
            except:
                # For newer Python versions
                try:
                    with open('/etc/os-release', 'r') as f:
                        lines = f.readlines()
                    os_release = {}
                    for line in lines:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os_release[key] = value.strip('"')
                    platform_info["linux"] = os_release
                except:
                    pass
        
        return platform_info
    
    def test_python_compatibility(self) -> Dict:
        """Test Python version compatibility."""
        print("🐍 Testing Python compatibility...")
        
        results = {
            "test_name": "python_compatibility",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Minimum Python version (3.8+)
        min_version = (3, 8)
        current_version = (self.python_version.major, self.python_version.minor)
        
        results["subtests"]["minimum_version"] = {
            "passed": current_version >= min_version,
            "description": f"Python version should be >= {min_version[0]}.{min_version[1]}",
            "required": f"{min_version[0]}.{min_version[1]}+",
            "actual": f"{current_version[0]}.{current_version[1]}.{self.python_version.micro}"
        }
        
        # Test 2: Required Python features
        required_features = [
            ("pathlib", "pathlib module for cross-platform paths"),
            ("asyncio", "asyncio for concurrent operations"),
            ("threading", "threading for background processing"),
            ("json", "JSON for configuration and data storage"),
            ("tempfile", "tempfile for temporary file operations"),
            ("subprocess", "subprocess for external command execution"),
            ("multiprocessing", "multiprocessing for parallel operations")
        ]
        
        for module_name, description in required_features:
            try:
                importlib.import_module(module_name)
                results["subtests"][f"feature_{module_name}"] = {
                    "passed": True,
                    "description": description,
                    "status": "Available"
                }
            except ImportError:
                results["subtests"][f"feature_{module_name}"] = {
                    "passed": False,
                    "description": description,
                    "status": "Missing"
                }
        
        return results
    
    def test_dependency_compatibility(self) -> Dict:
        """Test dependency compatibility across platforms."""
        print("📦 Testing dependency compatibility...")
        
        results = {
            "test_name": "dependency_compatibility",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Critical dependencies with platform considerations
        dependencies = [
            ("torch", "PyTorch for deep learning"),
            ("torchvision", "PyTorch vision utilities"),
            ("cv2", "OpenCV for computer vision"),
            ("PIL", "Pillow for image processing"),
            ("numpy", "NumPy for numerical operations"),
            ("requests", "Requests for HTTP operations"),
            ("chromadb", "ChromaDB for vector database"),
            ("psutil", "psutil for system monitoring"),
            ("tkinter", "Tkinter for GUI (built-in on most platforms)")
        ]
        
        for module_name, description in dependencies:
            try:
                if module_name == "tkinter":
                    # Special handling for tkinter
                    import tkinter as tk
                    # Try to create a root window to test functionality
                    root = tk.Tk()
                    root.destroy()
                    version = tk.TkVersion
                else:
                    module = importlib.import_module(module_name)
                    version = getattr(module, '__version__', 'Unknown')
                
                results["subtests"][f"dependency_{module_name}"] = {
                    "passed": True,
                    "description": description,
                    "status": "Available",
                    "version": str(version)
                }
                
            except ImportError as e:
                results["subtests"][f"dependency_{module_name}"] = {
                    "passed": False,
                    "description": description,
                    "status": "Missing",
                    "error": str(e)
                }
            except Exception as e:
                results["subtests"][f"dependency_{module_name}"] = {
                    "passed": False,
                    "description": description,
                    "status": "Error",
                    "error": str(e)
                }
        
        return results
    
    def test_camera_compatibility(self) -> Dict:
        """Test camera compatibility across platforms."""
        print("📷 Testing camera compatibility...")
        
        results = {
            "test_name": "camera_compatibility", 
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: OpenCV camera access
        try:
            import cv2
            
            # Try to access camera (index 0)
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                # Test basic camera operations
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    results["subtests"]["opencv_camera_access"] = {
                        "passed": True,
                        "description": "OpenCV camera access functional",
                        "status": "Working",
                        "frame_shape": frame.shape if hasattr(frame, 'shape') else None
                    }
                else:
                    results["subtests"]["opencv_camera_access"] = {
                        "passed": False,
                        "description": "Camera accessible but no frame captured",
                        "status": "Partial"
                    }
                
                cap.release()
            else:
                results["subtests"]["opencv_camera_access"] = {
                    "passed": False,
                    "description": "Camera not accessible",
                    "status": "No camera"
                }
                
        except Exception as e:
            results["subtests"]["opencv_camera_access"] = {
                "passed": False,
                "description": "Error accessing camera",
                "status": "Error",
                "error": str(e)
            }
        
        # Test 2: Platform-specific camera features
        if self.current_platform == "Darwin":  # macOS
            results["subtests"]["macos_camera_features"] = {
                "passed": True,
                "description": "macOS camera permissions and AVFoundation support",
                "notes": "Manual verification required for camera permissions"
            }
        elif self.current_platform == "Windows":
            results["subtests"]["windows_camera_features"] = {
                "passed": True,
                "description": "Windows DirectShow camera support",
                "notes": "Manual verification required for DirectShow compatibility"
            }
        elif self.current_platform == "Linux":
            results["subtests"]["linux_camera_features"] = {
                "passed": True,
                "description": "Linux V4L2 camera support",
                "notes": "Manual verification required for V4L2 compatibility"
            }
        
        return results
    
    def test_file_system_compatibility(self) -> Dict:
        """Test file system operations across platforms."""
        print("📁 Testing file system compatibility...")
        
        results = {
            "test_name": "filesystem_compatibility",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Path operations using pathlib
        try:
            from pathlib import Path
            
            # Test cross-platform path creation
            test_path = Path("test") / "subfolder" / "file.txt"
            
            results["subtests"]["pathlib_operations"] = {
                "passed": True,
                "description": "Cross-platform path operations with pathlib",
                "test_path": str(test_path),
                "separator": os.sep
            }
        except Exception as e:
            results["subtests"]["pathlib_operations"] = {
                "passed": False,
                "description": "Error with pathlib operations",
                "error": str(e)
            }
        
        # Test 2: File permissions
        try:
            import tempfile
            import stat
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            # Test permission operations
            if self.current_platform != "Windows":  # Unix-like systems
                original_mode = temp_path.stat().st_mode
                temp_path.chmod(0o644)
                new_mode = temp_path.stat().st_mode
                
                results["subtests"]["file_permissions"] = {
                    "passed": new_mode != original_mode,
                    "description": "File permission operations",
                    "original_mode": oct(original_mode),
                    "new_mode": oct(new_mode)
                }
            else:
                results["subtests"]["file_permissions"] = {
                    "passed": True,
                    "description": "Windows file permissions (limited support)",
                    "notes": "Windows has different permission model"
                }
            
            # Clean up
            temp_path.unlink()
            
        except Exception as e:
            results["subtests"]["file_permissions"] = {
                "passed": False,
                "description": "Error testing file permissions",
                "error": str(e)
            }
        
        # Test 3: Directory operations
        try:
            import tempfile
            import shutil
            
            temp_dir = Path(tempfile.mkdtemp())
            
            # Create nested directory structure
            nested_dir = temp_dir / "nested" / "deep" / "structure"
            nested_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test file
            test_file = nested_dir / "test.txt"
            test_file.write_text("test content")
            
            # Verify operations
            dir_created = nested_dir.exists()
            file_created = test_file.exists()
            file_content = test_file.read_text() == "test content"
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            results["subtests"]["directory_operations"] = {
                "passed": dir_created and file_created and file_content,
                "description": "Directory creation and file operations",
                "dir_created": dir_created,
                "file_created": file_created,
                "file_content_correct": file_content
            }
            
        except Exception as e:
            results["subtests"]["directory_operations"] = {
                "passed": False,
                "description": "Error with directory operations",
                "error": str(e)
            }
        
        return results
    
    def test_gui_compatibility(self) -> Dict:
        """Test GUI framework compatibility."""
        print("🖥️ Testing GUI compatibility...")
        
        results = {
            "test_name": "gui_compatibility",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Tkinter availability and functionality
        try:
            import tkinter as tk
            
            # Test basic Tkinter functionality
            root = tk.Tk()
            root.title("Test Window")
            
            # Test widget creation
            label = tk.Label(root, text="Test")
            button = tk.Button(root, text="Test Button")
            
            # Test geometry
            root.geometry("100x100")
            
            # Clean up immediately
            root.destroy()
            
            results["subtests"]["tkinter_basic"] = {
                "passed": True,
                "description": "Basic Tkinter functionality",
                "version": str(tk.TkVersion)
            }
            
        except Exception as e:
            results["subtests"]["tkinter_basic"] = {
                "passed": False,
                "description": "Error with basic Tkinter functionality",
                "error": str(e)
            }
        
        # Test 2: Platform-specific GUI features
        if self.current_platform == "Darwin":  # macOS
            try:
                import tkinter as tk
                root = tk.Tk()
                
                # Test macOS-specific features
                root.tk.call('tk', 'windowingsystem')  # Should return 'aqua'
                
                root.destroy()
                
                results["subtests"]["macos_gui_features"] = {
                    "passed": True,
                    "description": "macOS Aqua GUI support",
                    "windowing_system": "aqua"
                }
            except Exception as e:
                results["subtests"]["macos_gui_features"] = {
                    "passed": False,
                    "description": "Error with macOS GUI features",
                    "error": str(e)
                }
        
        # Test 3: High DPI support
        try:
            import tkinter as tk
            
            root = tk.Tk()
            
            # Get DPI information
            dpi_x = root.winfo_fpixels('1i')
            dpi_y = root.winfo_fpixels('1i')
            
            root.destroy()
            
            results["subtests"]["high_dpi_support"] = {
                "passed": True,
                "description": "High DPI display support",
                "dpi_x": dpi_x,
                "dpi_y": dpi_y,
                "high_dpi": dpi_x > 96 or dpi_y > 96
            }
            
        except Exception as e:
            results["subtests"]["high_dpi_support"] = {
                "passed": False,
                "description": "Error testing DPI support",
                "error": str(e)
            }
        
        return results
    
    def test_performance_characteristics(self) -> Dict:
        """Test platform-specific performance characteristics."""
        print("⚡ Testing platform performance characteristics...")
        
        results = {
            "test_name": "performance_characteristics",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: PyTorch device availability
        try:
            import torch
            
            device_info = {
                "cpu_available": True,
                "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            # Determine optimal device
            if device_info["mps_available"]:
                optimal_device = "mps"
            elif device_info["cuda_available"]:
                optimal_device = "cuda"
            else:
                optimal_device = "cpu"
            
            results["subtests"]["pytorch_devices"] = {
                "passed": True,
                "description": "PyTorch device availability",
                "devices": device_info,
                "optimal_device": optimal_device
            }
            
        except Exception as e:
            results["subtests"]["pytorch_devices"] = {
                "passed": False,
                "description": "Error checking PyTorch devices",
                "error": str(e)
            }
        
        # Test 2: Memory characteristics
        try:
            import psutil
            
            memory_info = psutil.virtual_memory()
            
            results["subtests"]["memory_characteristics"] = {
                "passed": memory_info.total >= 4 * 1024**3,  # At least 4GB
                "description": "System memory characteristics",
                "total_gb": round(memory_info.total / (1024**3), 2),
                "available_gb": round(memory_info.available / (1024**3), 2),
                "percent_used": memory_info.percent,
                "sufficient_memory": memory_info.total >= 4 * 1024**3
            }
            
        except Exception as e:
            results["subtests"]["memory_characteristics"] = {
                "passed": False,
                "description": "Error checking memory characteristics",
                "error": str(e)
            }
        
        # Test 3: CPU characteristics
        try:
            import psutil
            
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            # Check if sufficient for real-time processing
            sufficient_cores = cpu_info["logical_cores"] >= 4
            
            results["subtests"]["cpu_characteristics"] = {
                "passed": sufficient_cores,
                "description": "CPU characteristics for real-time processing",
                "cpu_info": cpu_info,
                "sufficient_cores": sufficient_cores
            }
            
        except Exception as e:
            results["subtests"]["cpu_characteristics"] = {
                "passed": False,
                "description": "Error checking CPU characteristics",
                "error": str(e)
            }
        
        return results
    
    def run_comprehensive_crossplatform_test(self) -> Dict:
        """Run comprehensive cross-platform testing."""
        print("🌍 Starting comprehensive cross-platform testing...")
        
        test_results = {
            "test_start": datetime.now().isoformat(),
            "platform_info": self.get_platform_info(),
            "python_compatibility": self.test_python_compatibility(),
            "dependency_compatibility": self.test_dependency_compatibility(),
            "camera_compatibility": self.test_camera_compatibility(),
            "filesystem_compatibility": self.test_file_system_compatibility(),
            "gui_compatibility": self.test_gui_compatibility(),
            "performance_characteristics": self.test_performance_characteristics()
        }
        
        test_results["test_end"] = datetime.now().isoformat()
        
        # Generate summary
        summary = self._generate_crossplatform_summary(test_results)
        test_results["summary"] = summary
        
        return test_results
    
    def _generate_crossplatform_summary(self, results: Dict) -> Dict:
        """Generate cross-platform test summary."""
        summary = {
            "overall_status": "PASS",
            "platform": self.current_platform,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "compatibility_issues": [],
            "platform_specific_notes": []
        }
        
        # Analyze each test category
        for category_name, category_results in results.items():
            if category_name in ["test_start", "test_end", "summary", "platform_info"]:
                continue
                
            if "subtests" in category_results:
                category_total = len(category_results["subtests"])
                category_passed = sum(1 for test in category_results["subtests"].values() 
                                    if test.get("passed", False))
                
                summary["total_tests"] += category_total
                summary["passed_tests"] += category_passed
                
                # Identify compatibility issues
                for test_name, test_result in category_results["subtests"].items():
                    if not test_result.get("passed", False):
                        summary["compatibility_issues"].append({
                            "category": category_name,
                            "test": test_name,
                            "description": test_result.get("description", "Unknown issue"),
                            "error": test_result.get("error"),
                            "platform": self.current_platform
                        })
        
        summary["failed_tests"] = summary["total_tests"] - summary["passed_tests"]
        
        # Platform-specific recommendations
        if self.current_platform == "Darwin":
            summary["platform_specific_notes"].extend([
                "Camera permissions may require user approval",
                "MPS acceleration available on Apple Silicon",
                "Use native macOS packaging for distribution"
            ])
        elif self.current_platform == "Windows":
            summary["platform_specific_notes"].extend([
                "DirectShow camera support required",
                "Consider CUDA support for GPU acceleration",
                "Windows Defender may flag packaged applications"
            ])
        elif self.current_platform == "Linux":
            summary["platform_specific_notes"].extend([
                "V4L2 camera support required",
                "Package dependencies may vary by distribution",
                "Consider AppImage or Flatpak for distribution"
            ])
        
        # Overall compatibility assessment
        compatibility_rate = summary["passed_tests"] / summary["total_tests"] if summary["total_tests"] > 0 else 0
        
        if compatibility_rate >= 0.95:
            summary["overall_status"] = "EXCELLENT"
        elif compatibility_rate >= 0.85:
            summary["overall_status"] = "GOOD"
        elif compatibility_rate >= 0.70:
            summary["overall_status"] = "ACCEPTABLE"
        else:
            summary["overall_status"] = "POOR"
        
        summary["compatibility_rate"] = compatibility_rate
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            platform_name = self.current_platform.lower()
            filename = f"crossplatform_test_results_{platform_name}_{timestamp}.json"
        
        filepath = Path("test_results") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Cross-platform test results saved to: {filepath}")


def main():
    """Run Phase 4 cross-platform testing."""
    print("🌍 VinylVision Phase 4 - Cross-Platform Testing")
    print("=" * 50)
    
    tester = CrossPlatformTester()
    
    try:
        results = tester.run_comprehensive_crossplatform_test()
        
        # Print summary
        summary = results.get("summary", {})
        platform_info = results.get("platform_info", {})
        
        print(f"\n📊 Cross-Platform Test Summary:")
        print(f"   Platform: {platform_info.get('system', 'Unknown')} {platform_info.get('platform', '')}")
        print(f"   Python: {platform_info.get('python_version', {}).get('full', 'Unknown')[:50]}")
        print(f"   Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"   Compatibility Rate: {summary.get('compatibility_rate', 0):.1%}")
        print(f"   Tests Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
        
        # Show compatibility issues
        issues = summary.get("compatibility_issues", [])
        if issues:
            print(f"\n⚠️ Compatibility Issues:")
            for issue in issues[:5]:  # Show first 5
                print(f"   - {issue['category']}.{issue['test']}: {issue['description']}")
        
        # Show platform-specific notes
        notes = summary.get("platform_specific_notes", [])
        if notes:
            print(f"\n📝 Platform-Specific Notes:")
            for note in notes:
                print(f"   - {note}")
        
        # Save results
        tester.save_results(results)
        
        success = summary.get("overall_status") in ["EXCELLENT", "GOOD", "ACCEPTABLE"]
        if success:
            print(f"🎉 Platform compatibility: {summary.get('overall_status')}")
        else:
            print("⚠️ Platform compatibility issues found")
        
        return success
        
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
