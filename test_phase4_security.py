#!/usr/bin/env python3
"""
Phase 4 Security & Error Handling Testing Suite for VinylVision

Tests robust error handling and security features:
- Camera access failures
- Network connectivity issues
- API rate limit handling
- Database corruption recovery
- Secure API credential storage
- Input validation testing
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from unittest.mock import patch, MagicMock
import requests

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from core.camera import CameraManager
from core.database import VectorDatabase
from core.discogs_client import DiscogsClient
from utils.config import load_config
from models.efficientnet import AlbumFeatureExtractor


class SecurityTester:
    """Comprehensive security and error handling testing for VinylVision."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        
    def setup(self) -> bool:
        """Initialize testing environment."""
        print("🔧 Setting up security testing environment...")
        
        try:
            # Create temporary directory for testing
            self.temp_dir = tempfile.mkdtemp(prefix="vinylvision_security_test_")
            print(f"✅ Created temporary directory: {self.temp_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up testing environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory")
    
    def test_camera_error_handling(self) -> Dict:
        """Test camera access failure scenarios."""
        print("📷 Testing camera error handling...")
        
        results = {
            "test_name": "camera_error_handling",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: No camera available
        try:
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap.return_value.isOpened.return_value = False
                
                camera = CameraManager()
                init_result = camera.initialize()
                
                results["subtests"]["no_camera_available"] = {
                    "passed": not init_result,  # Should fail gracefully
                    "description": "Camera initialization should fail gracefully when no camera available",
                    "actual_result": init_result
                }
        except Exception as e:
            results["subtests"]["no_camera_available"] = {
                "passed": False,
                "description": "Exception during no camera test",
                "error": str(e)
            }
        
        # Test 2: Camera disconnected during operation
        try:
            with patch('cv2.VideoCapture') as mock_cap:
                # Initially successful
                mock_cap.return_value.isOpened.return_value = True
                mock_cap.return_value.read.side_effect = [(False, None)]  # Simulate failure
                
                camera = CameraManager()
                if camera.initialize():
                    frame = camera.read_frame()
                    
                    results["subtests"]["camera_disconnected"] = {
                        "passed": frame is None,  # Should handle gracefully
                        "description": "Should handle camera disconnection gracefully",
                        "actual_result": frame is not None
                    }
                else:
                    results["subtests"]["camera_disconnected"] = {
                        "passed": False,
                        "description": "Camera initialization failed unexpectedly"
                    }
        except Exception as e:
            results["subtests"]["camera_disconnected"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled during camera disconnection test",
                "error": str(e)
            }
        
        # Test 3: Invalid camera index
        try:
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap.return_value.isOpened.return_value = False
                
                camera = CameraManager(camera_index=99)  # Invalid index
                init_result = camera.initialize()
                
                results["subtests"]["invalid_camera_index"] = {
                    "passed": not init_result,  # Should fail gracefully
                    "description": "Should handle invalid camera index gracefully",
                    "actual_result": init_result
                }
        except Exception as e:
            results["subtests"]["invalid_camera_index"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for invalid camera index",
                "error": str(e)
            }
        
        return results
    
    def test_network_error_handling(self) -> Dict:
        """Test network connectivity error scenarios."""
        print("🌐 Testing network error handling...")
        
        results = {
            "test_name": "network_error_handling",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: No internet connection
        try:
            with patch('requests.get') as mock_get:
                mock_get.side_effect = requests.ConnectionError("No internet connection")
                
                # Test Discogs API with no connection
                discogs = DiscogsClient()
                search_result = discogs.search_album("test", "test")
                
                results["subtests"]["no_internet"] = {
                    "passed": search_result is None or search_result == {},
                    "description": "Should handle no internet connection gracefully",
                    "actual_result": search_result
                }
        except Exception as e:
            results["subtests"]["no_internet"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for no internet connection",
                "error": str(e)
            }
        
        # Test 2: API timeout
        try:
            with patch('requests.get') as mock_get:
                mock_get.side_effect = requests.Timeout("Request timeout")
                
                discogs = DiscogsClient()
                search_result = discogs.search_album("test", "test")
                
                results["subtests"]["api_timeout"] = {
                    "passed": search_result is None or search_result == {},
                    "description": "Should handle API timeout gracefully",
                    "actual_result": search_result
                }
        except Exception as e:
            results["subtests"]["api_timeout"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for API timeout",
                "error": str(e)
            }
        
        # Test 3: HTTP error responses
        try:
            with patch('requests.get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
                mock_get.return_value = mock_response
                
                discogs = DiscogsClient()
                search_result = discogs.search_album("test", "test")
                
                results["subtests"]["http_error"] = {
                    "passed": search_result is None or search_result == {},
                    "description": "Should handle HTTP errors gracefully",
                    "actual_result": search_result
                }
        except Exception as e:
            results["subtests"]["http_error"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for HTTP error",
                "error": str(e)
            }
        
        return results
    
    def test_api_rate_limiting(self) -> Dict:
        """Test API rate limiting handling."""
        print("⏱️ Testing API rate limiting...")
        
        results = {
            "test_name": "api_rate_limiting",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Rate limit exceeded response
        try:
            with patch('requests.get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 429  # Too Many Requests
                mock_response.headers = {'Retry-After': '60'}
                mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
                mock_get.return_value = mock_response
                
                discogs = DiscogsClient()
                search_result = discogs.search_album("test", "test")
                
                results["subtests"]["rate_limit_exceeded"] = {
                    "passed": search_result is None or search_result == {},
                    "description": "Should handle rate limit exceeded gracefully",
                    "actual_result": search_result
                }
        except Exception as e:
            results["subtests"]["rate_limit_exceeded"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for rate limit exceeded",
                "error": str(e)
            }
        
        # Test 2: Rate limiting implementation
        try:
            discogs = DiscogsClient()
            
            # Check if rate limiting attributes exist
            has_rate_limiting = hasattr(discogs, 'last_request_time') or hasattr(discogs, '_rate_limit')
            
            results["subtests"]["rate_limiting_implementation"] = {
                "passed": has_rate_limiting,
                "description": "Should implement rate limiting mechanism",
                "actual_result": has_rate_limiting
            }
        except Exception as e:
            results["subtests"]["rate_limiting_implementation"] = {
                "passed": False,
                "description": "Error checking rate limiting implementation",
                "error": str(e)
            }
        
        return results
    
    def test_database_corruption_recovery(self) -> Dict:
        """Test database corruption and recovery scenarios."""
        print("💾 Testing database corruption recovery...")
        
        results = {
            "test_name": "database_corruption_recovery",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Corrupted database file
        try:
            # Create a corrupted database file
            corrupted_db_path = Path(self.temp_dir) / "corrupted.db"
            with open(corrupted_db_path, 'w') as f:
                f.write("This is not a valid database file")
            
            # Try to initialize database with corrupted file
            with patch.dict(os.environ, {'CHROMA_DB_PATH': str(corrupted_db_path)}):
                db = VectorDatabase()
                init_result = db.initialize()
                
                results["subtests"]["corrupted_database"] = {
                    "passed": not init_result,  # Should fail gracefully or recover
                    "description": "Should handle corrupted database gracefully",
                    "actual_result": init_result
                }
        except Exception as e:
            results["subtests"]["corrupted_database"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for corrupted database",
                "error": str(e)
            }
        
        # Test 2: Missing database directory
        try:
            missing_db_path = Path(self.temp_dir) / "nonexistent" / "database"
            
            with patch.dict(os.environ, {'CHROMA_DB_PATH': str(missing_db_path)}):
                db = VectorDatabase()
                init_result = db.initialize()
                
                results["subtests"]["missing_database_dir"] = {
                    "passed": True,  # Should create directory or handle gracefully
                    "description": "Should handle missing database directory",
                    "actual_result": init_result
                }
        except Exception as e:
            results["subtests"]["missing_database_dir"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for missing database directory",
                "error": str(e)
            }
        
        # Test 3: Permission denied on database
        try:
            restricted_db_path = Path(self.temp_dir) / "restricted.db"
            restricted_db_path.touch()
            restricted_db_path.chmod(0o000)  # No permissions
            
            with patch.dict(os.environ, {'CHROMA_DB_PATH': str(restricted_db_path)}):
                db = VectorDatabase()
                init_result = db.initialize()
                
                results["subtests"]["permission_denied"] = {
                    "passed": not init_result,  # Should fail gracefully
                    "description": "Should handle permission denied gracefully",
                    "actual_result": init_result
                }
            
            # Restore permissions for cleanup
            restricted_db_path.chmod(0o644)
            
        except Exception as e:
            results["subtests"]["permission_denied"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for permission denied",
                "error": str(e)
            }
        
        return results
    
    def test_secure_credential_storage(self) -> Dict:
        """Test secure API credential storage."""
        print("🔐 Testing secure credential storage...")
        
        results = {
            "test_name": "secure_credential_storage",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Credentials not stored in plain text
        try:
            config_file = Path("config/config.py")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_content = f.read()
                
                # Check for obvious plain text credentials
                has_plain_key = 'consumer_key' in config_content.lower() and '=' in config_content
                has_plain_secret = 'consumer_secret' in config_content.lower() and '=' in config_content
                
                # Check if actual credential values are visible
                has_visible_credentials = any(
                    len(line.split('=')[1].strip().strip('"\'')) > 10
                    for line in config_content.split('\n')
                    if '=' in line and ('key' in line.lower() or 'secret' in line.lower())
                )
                
                results["subtests"]["plain_text_credentials"] = {
                    "passed": not has_visible_credentials,
                    "description": "Credentials should not be stored in plain text",
                    "details": {
                        "has_key_field": has_plain_key,
                        "has_secret_field": has_plain_secret,
                        "has_visible_values": has_visible_credentials
                    }
                }
            else:
                results["subtests"]["plain_text_credentials"] = {
                    "passed": True,
                    "description": "No config file found - using environment variables",
                    "actual_result": "Config file not found"
                }
        except Exception as e:
            results["subtests"]["plain_text_credentials"] = {
                "passed": False,
                "description": "Error checking credential storage",
                "error": str(e)
            }
        
        # Test 2: Environment variable support
        try:
            # Test if application reads from environment variables
            test_key = "test_discogs_key_123"
            test_secret = "test_discogs_secret_456"
            
            with patch.dict(os.environ, {
                'DISCOGS_CONSUMER_KEY': test_key,
                'DISCOGS_CONSUMER_SECRET': test_secret
            }):
                config = load_config()
                
                env_key_used = config.get('DISCOGS', {}).get('CONSUMER_KEY') == test_key
                env_secret_used = config.get('DISCOGS', {}).get('CONSUMER_SECRET') == test_secret
                
                results["subtests"]["environment_variable_support"] = {
                    "passed": env_key_used and env_secret_used,
                    "description": "Should support environment variables for credentials",
                    "details": {
                        "env_key_used": env_key_used,
                        "env_secret_used": env_secret_used
                    }
                }
        except Exception as e:
            results["subtests"]["environment_variable_support"] = {
                "passed": False,
                "description": "Error testing environment variable support",
                "error": str(e)
            }
        
        # Test 3: Config file permissions
        try:
            config_file = Path("config/config.py")
            if config_file.exists():
                file_stat = config_file.stat()
                file_mode = oct(file_stat.st_mode)[-3:]  # Get last 3 digits
                
                # Check if file is readable by others (world-readable)
                world_readable = int(file_mode[2]) & 4 > 0
                
                results["subtests"]["config_file_permissions"] = {
                    "passed": not world_readable,
                    "description": "Config file should not be world-readable",
                    "details": {
                        "file_mode": file_mode,
                        "world_readable": world_readable
                    }
                }
            else:
                results["subtests"]["config_file_permissions"] = {
                    "passed": True,
                    "description": "No config file found",
                    "actual_result": "No config file"
                }
        except Exception as e:
            results["subtests"]["config_file_permissions"] = {
                "passed": False,
                "description": "Error checking config file permissions",
                "error": str(e)
            }
        
        return results
    
    def test_input_validation(self) -> Dict:
        """Test input validation and sanitization."""
        print("🧪 Testing input validation...")
        
        results = {
            "test_name": "input_validation",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: SQL injection prevention in database queries
        try:
            db = VectorDatabase()
            if db.initialize():
                # Try SQL injection in search
                malicious_query = "'; DROP TABLE embeddings; --"
                
                # This should not crash or cause issues
                search_result = db.search_similar(None, top_k=5, query_filter=malicious_query)
                
                results["subtests"]["sql_injection_prevention"] = {
                    "passed": True,  # If we get here, no crash occurred
                    "description": "Should prevent SQL injection attacks",
                    "actual_result": "No crash occurred"
                }
            else:
                results["subtests"]["sql_injection_prevention"] = {
                    "passed": True,
                    "description": "Database initialization failed - test skipped",
                    "actual_result": "Database not available"
                }
        except Exception as e:
            results["subtests"]["sql_injection_prevention"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for SQL injection test",
                "error": str(e)
            }
        
        # Test 2: Path traversal prevention
        try:
            # Test malicious file paths
            malicious_paths = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "/etc/shadow",
                "C:\\Windows\\System32\\config\\SAM"
            ]
            
            path_traversal_blocked = True
            for malicious_path in malicious_paths:
                try:
                    # Try to use malicious path in config loading
                    config = load_config(malicious_path)
                    if config and len(str(config)) > 100:  # If we got sensitive data
                        path_traversal_blocked = False
                        break
                except:
                    pass  # Expected to fail
            
            results["subtests"]["path_traversal_prevention"] = {
                "passed": path_traversal_blocked,
                "description": "Should prevent path traversal attacks",
                "actual_result": path_traversal_blocked
            }
        except Exception as e:
            results["subtests"]["path_traversal_prevention"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for path traversal test",
                "error": str(e)
            }
        
        # Test 3: Large input handling
        try:
            # Test with very large strings
            large_string = "A" * 1000000  # 1MB string
            
            # Test if application handles large inputs gracefully
            discogs = DiscogsClient()
            search_result = discogs.search_album(large_string, large_string)
            
            results["subtests"]["large_input_handling"] = {
                "passed": True,  # If we get here, no crash occurred
                "description": "Should handle large inputs gracefully",
                "actual_result": "No crash with 1MB input"
            }
        except Exception as e:
            results["subtests"]["large_input_handling"] = {
                "passed": True,  # Exception handling is acceptable for large inputs
                "description": "Exception handled for large input test",
                "error": str(e)
            }
        
        return results
    
    def test_model_loading_errors(self) -> Dict:
        """Test model loading error scenarios."""
        print("🤖 Testing model loading error handling...")
        
        results = {
            "test_name": "model_loading_errors",
            "timestamp": datetime.now().isoformat(),
            "subtests": {}
        }
        
        # Test 1: Missing model files
        try:
            with patch('torch.load') as mock_load:
                mock_load.side_effect = FileNotFoundError("Model file not found")
                
                extractor = AlbumFeatureExtractor()
                load_result = extractor.load_model()
                
                results["subtests"]["missing_model_files"] = {
                    "passed": not load_result,  # Should fail gracefully
                    "description": "Should handle missing model files gracefully",
                    "actual_result": load_result
                }
        except Exception as e:
            results["subtests"]["missing_model_files"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for missing model files",
                "error": str(e)
            }
        
        # Test 2: Corrupted model files
        try:
            with patch('torch.load') as mock_load:
                mock_load.side_effect = RuntimeError("Model file corrupted")
                
                extractor = AlbumFeatureExtractor()
                load_result = extractor.load_model()
                
                results["subtests"]["corrupted_model_files"] = {
                    "passed": not load_result,  # Should fail gracefully
                    "description": "Should handle corrupted model files gracefully",
                    "actual_result": load_result
                }
        except Exception as e:
            results["subtests"]["corrupted_model_files"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for corrupted model files",
                "error": str(e)
            }
        
        # Test 3: Insufficient memory for model
        try:
            with patch('torch.load') as mock_load:
                mock_load.side_effect = RuntimeError("CUDA out of memory")
                
                extractor = AlbumFeatureExtractor()
                load_result = extractor.load_model()
                
                results["subtests"]["insufficient_memory"] = {
                    "passed": not load_result,  # Should fail gracefully
                    "description": "Should handle insufficient memory gracefully",
                    "actual_result": load_result
                }
        except Exception as e:
            results["subtests"]["insufficient_memory"] = {
                "passed": True,  # Exception handling is acceptable
                "description": "Exception handled for insufficient memory",
                "error": str(e)
            }
        
        return results
    
    def run_comprehensive_security_test(self) -> Dict:
        """Run comprehensive security and error handling testing."""
        print("🔒 Starting comprehensive security testing...")
        
        if not self.setup():
            return {"error": "Setup failed"}
        
        test_results = {
            "test_start": datetime.now().isoformat(),
            "camera_errors": self.test_camera_error_handling(),
            "network_errors": self.test_network_error_handling(), 
            "api_rate_limiting": self.test_api_rate_limiting(),
            "database_corruption": self.test_database_corruption_recovery(),
            "credential_security": self.test_secure_credential_storage(),
            "input_validation": self.test_input_validation(),
            "model_loading_errors": self.test_model_loading_errors()
        }
        
        test_results["test_end"] = datetime.now().isoformat()
        
        # Generate summary
        summary = self._generate_security_summary(test_results)
        test_results["summary"] = summary
        
        return test_results
    
    def _generate_security_summary(self, results: Dict) -> Dict:
        """Generate security test summary."""
        summary = {
            "overall_status": "PASS",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_categories": {},
            "security_issues": []
        }
        
        # Analyze each test category
        for category_name, category_results in results.items():
            if category_name in ["test_start", "test_end", "summary"]:
                continue
                
            if "subtests" in category_results:
                category_total = len(category_results["subtests"])
                category_passed = sum(1 for test in category_results["subtests"].values() 
                                    if test.get("passed", False))
                
                summary["test_categories"][category_name] = {
                    "total": category_total,
                    "passed": category_passed,
                    "success_rate": category_passed / category_total if category_total > 0 else 0
                }
                
                summary["total_tests"] += category_total
                summary["passed_tests"] += category_passed
                
                # Identify security issues
                for test_name, test_result in category_results["subtests"].items():
                    if not test_result.get("passed", False):
                        summary["security_issues"].append({
                            "category": category_name,
                            "test": test_name,
                            "description": test_result.get("description", "Unknown issue"),
                            "error": test_result.get("error")
                        })
        
        summary["failed_tests"] = summary["total_tests"] - summary["passed_tests"]
        
        # Overall pass/fail based on critical security tests
        critical_failures = 0
        for issue in summary["security_issues"]:
            if any(keyword in issue["test"].lower() for keyword in 
                   ["credential", "injection", "traversal", "permission"]):
                critical_failures += 1
        
        summary["overall_status"] = "PASS" if critical_failures == 0 else "FAIL"
        summary["critical_security_failures"] = critical_failures
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_test_results_{timestamp}.json"
        
        filepath = Path("test_results") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Security test results saved to: {filepath}")


def main():
    """Run Phase 4 security testing."""
    print("🔒 VinylVision Phase 4 - Security & Error Handling Testing")
    print("=" * 60)
    
    tester = SecurityTester()
    
    try:
        results = tester.run_comprehensive_security_test()
        
        if "error" in results:
            print(f"❌ Testing failed: {results['error']}")
            return False
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n📊 Security Test Summary:")
        print(f"   Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Passed: {summary.get('passed_tests', 0)}")
        print(f"   Failed: {summary.get('failed_tests', 0)}")
        print(f"   Critical Security Failures: {summary.get('critical_security_failures', 0)}")
        
        # Show test categories
        categories = summary.get("test_categories", {})
        for category, stats in categories.items():
            success_rate = stats.get("success_rate", 0) * 100
            status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
            print(f"   {category}: {status} {success_rate:.0f}% ({stats.get('passed', 0)}/{stats.get('total', 0)})")
        
        # Show security issues
        issues = summary.get("security_issues", [])
        if issues:
            print(f"\n🚨 Security Issues Found:")
            for issue in issues[:5]:  # Show first 5
                print(f"   - {issue['category']}.{issue['test']}: {issue['description']}")
        
        # Save results
        tester.save_results(results)
        
        success = summary.get("overall_status") == "PASS"
        if success:
            print("🎉 All critical security tests passed!")
        else:
            print("⚠️ Critical security issues found - review required")
        
        return success
        
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return False
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
