"""
VinylVision Configuration Example
Copy this file to config.py and fill in your actual values
"""

# Discogs API Configuration
DISCOGS_CONFIG = {
    "consumer_key": "your_consumer_key_here",
    "consumer_secret": "your_consumer_secret_here",
    "user_token": "your_user_token_here",  # Optional: for authenticated requests
    "user_agent": "VinylVision/1.0 +https://github.com/pmoneynz/VinylVision"
}

# API Rate Limiting
API_RATE_LIMIT = {
    "requests_per_minute": 60,  # Discogs authenticated rate limit
    "retry_delay": 1.0,  # seconds
    "max_retries": 3
}

# Computer Vision Model Configuration
MODEL_CONFIG = {
    "model_name": "efficientnet-b0",
    "input_size": (224, 224),
    "batch_size": 1,
    "confidence_threshold": 0.8,
    "max_results": 5
}

# Vector Database Configuration
DATABASE_CONFIG = {
    "db_path": "data/embeddings/vinyl_db",
    "collection_name": "album_covers",
    "embedding_dimension": 512,
    "similarity_threshold": 0.75
}

# Camera Configuration
CAMERA_CONFIG = {
    "device_id": 0,  # Usually 0 for default camera
    "fps": 3,  # Frames per second for processing
    "resolution": (640, 480),
    "auto_exposure": True,
    "auto_focus": True
}

# UI Configuration
UI_CONFIG = {
    "window_title": "VinylVision - Album Cover Recognition",
    "window_size": (1200, 800),
    "theme": "default",
    "show_confidence": True,
    "show_fps": True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "use_gpu": True,  # Set to False if no GPU available
    "num_threads": 4,
    "memory_limit_gb": 2,
    "cache_size_mb": 500
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_file": "logs/vinylvision.log",
    "max_log_size_mb": 10,
    "backup_count": 5
}

# Data Paths
PATHS = {
    "models": "models/",
    "cache": "data/cache/",
    "logs": "logs/",
    "temp": "temp/",
    "user_data": "user_data/"
}

# Feature Flags
FEATURES = {
    "offline_mode": True,
    "auto_update_db": True,
    "export_functionality": True,
    "debug_mode": False
}
