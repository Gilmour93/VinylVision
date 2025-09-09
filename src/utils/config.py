"""
Configuration management for VinylVision.

Handles loading and validation of application settings.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    device_id: int = 0
    target_fps: int = 3
    resolution_width: int = 1280
    resolution_height: int = 720


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_name: str = 'efficientnet-b0'
    device: str = 'auto'
    confidence_threshold: float = 0.8
    feature_dim: int = 512


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str = 'data/embeddings'
    cache_path: str = 'data/cache'
    max_results: int = 5


@dataclass
class DiscogsConfig:
    """Discogs API configuration settings."""
    consumer_key: str = ''
    consumer_secret: str = ''
    rate_limit_delay: float = 1.0
    max_search_results: int = 10


@dataclass
class UIConfig:
    """User interface configuration settings."""
    window_width: int = 1200
    window_height: int = 800
    show_confidence: bool = True
    show_fps: bool = True


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: str = 'config/config.py'):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.camera = CameraConfig()
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.discogs = DiscogsConfig()
        self.ui = UIConfig()
        
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if configuration loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                return self._load_from_environment()
            
            # Import config as Python module
            config_dir = os.path.dirname(self.config_file)
            config_name = os.path.splitext(os.path.basename(self.config_file))[0]
            
            import sys
            if config_dir not in sys.path:
                sys.path.insert(0, config_dir)
            
            config_module = __import__(config_name)
            
            # Load configuration sections
            if hasattr(config_module, 'CAMERA_CONFIG'):
                self._update_camera_config(config_module.CAMERA_CONFIG)
            
            if hasattr(config_module, 'MODEL_CONFIG'):
                self._update_model_config(config_module.MODEL_CONFIG)
            
            if hasattr(config_module, 'DATABASE_CONFIG'):
                self._update_database_config(config_module.DATABASE_CONFIG)
            
            if hasattr(config_module, 'DISCOGS_CONFIG'):
                self._update_discogs_config(config_module.DISCOGS_CONFIG)
            
            if hasattr(config_module, 'UI_CONFIG'):
                self._update_ui_config(config_module.UI_CONFIG)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._load_from_environment()
    
    def _load_from_environment(self) -> bool:
        """
        Load configuration from environment variables.
        
        Returns:
            bool: True if any environment variables were loaded
        """
        loaded = False
        
        # Discogs configuration from environment
        if 'DISCOGS_CONSUMER_KEY' in os.environ:
            self.discogs.consumer_key = os.environ['DISCOGS_CONSUMER_KEY']
            loaded = True
        
        if 'DISCOGS_CONSUMER_SECRET' in os.environ:
            self.discogs.consumer_secret = os.environ['DISCOGS_CONSUMER_SECRET']
            loaded = True
        
        # Camera configuration
        if 'CAMERA_DEVICE_ID' in os.environ:
            try:
                self.camera.device_id = int(os.environ['CAMERA_DEVICE_ID'])
                loaded = True
            except ValueError:
                logger.warning("Invalid CAMERA_DEVICE_ID environment variable")
        
        if loaded:
            logger.info("Configuration loaded from environment variables")
        else:
            logger.warning("No configuration found, using defaults")
        
        return True  # Always return True to allow app to start with defaults
    
    def _update_camera_config(self, config_dict: Dict[str, Any]):
        """Update camera configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.camera, key):
                setattr(self.camera, key, value)
    
    def _update_model_config(self, config_dict: Dict[str, Any]):
        """Update model configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
    
    def _update_database_config(self, config_dict: Dict[str, Any]):
        """Update database configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.database, key):
                setattr(self.database, key, value)
    
    def _update_discogs_config(self, config_dict: Dict[str, Any]):
        """Update Discogs configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.discogs, key):
                setattr(self.discogs, key, value)
    
    def _update_ui_config(self, config_dict: Dict[str, Any]):
        """Update UI configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.ui, key):
                setattr(self.ui, key, value)
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check required Discogs credentials
        if not self.discogs.consumer_key or not self.discogs.consumer_secret:
            logger.error("Discogs API credentials not configured")
            return False
        
        # Validate camera settings
        if self.camera.device_id < 0:
            logger.error("Invalid camera device ID")
            return False
        
        # Validate model settings
        if self.model.confidence_threshold < 0 or self.model.confidence_threshold > 1:
            logger.error("Model confidence threshold must be between 0 and 1")
            return False
        
        # Validate directories
        os.makedirs(self.database.db_path, exist_ok=True)
        os.makedirs(self.database.cache_path, exist_ok=True)
        
        logger.info("Configuration validation passed")
        return True
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'camera': self.camera.__dict__,
            'model': self.model.__dict__,
            'database': self.database.__dict__,
            'discogs': {k: v if k not in ['consumer_key', 'consumer_secret'] else '***' 
                       for k, v in self.discogs.__dict__.items()},
            'ui': self.ui.__dict__
        }
    
    def save_user_settings(self, settings_file: str = 'config/user_settings.json'):
        """
        Save user-modifiable settings to JSON file.
        
        Args:
            settings_file: Path to settings file
        """
        try:
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            
            user_settings = {
                'camera': {
                    'device_id': self.camera.device_id,
                    'target_fps': self.camera.target_fps
                },
                'model': {
                    'confidence_threshold': self.model.confidence_threshold,
                    'device': self.model.device
                },
                'ui': self.ui.__dict__
            }
            
            with open(settings_file, 'w') as f:
                json.dump(user_settings, f, indent=2)
            
            logger.info(f"User settings saved to {settings_file}")
            
        except Exception as e:
            logger.error(f"Error saving user settings: {e}")
    
    def load_user_settings(self, settings_file: str = 'config/user_settings.json'):
        """
        Load user settings from JSON file.
        
        Args:
            settings_file: Path to settings file
        """
        try:
            if not os.path.exists(settings_file):
                return
            
            with open(settings_file, 'r') as f:
                user_settings = json.load(f)
            
            # Update configurations
            if 'camera' in user_settings:
                self._update_camera_config(user_settings['camera'])
            
            if 'model' in user_settings:
                self._update_model_config(user_settings['model'])
            
            if 'ui' in user_settings:
                self._update_ui_config(user_settings['ui'])
            
            logger.info(f"User settings loaded from {settings_file}")
            
        except Exception as e:
            logger.error(f"Error loading user settings: {e}")


# Global configuration instance
_config_manager = None


def load_config(config_file: str = 'config/config.py') -> ConfigManager:
    """
    Load application configuration.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        ConfigManager: Configured application settings
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
        _config_manager.load_config()
    
    return _config_manager


def get_config() -> ConfigManager:
    """
    Get the current configuration manager instance.
    
    Returns:
        ConfigManager: Current configuration instance
    """
    global _config_manager
    
    if _config_manager is None:
        return load_config()
    
    return _config_manager
