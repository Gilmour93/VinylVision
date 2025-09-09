"""
VinylVision - Main Application Entry Point

Real-time vinyl record album cover recognition powered by computer vision.
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add src directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from ui.enhanced_window import EnhancedVinylVisionWindow
    from ui.main_window import VinylVisionMainWindow
except ImportError as e:
    print(f"Failed to import UI components: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def main():
    """Main application entry point."""
    # Configure logging
    logger.add("logs/vinylvision.log", rotation="10 MB", level="INFO")
    logger.info("Starting VinylVision application")
    
    try:
        # Create and run the enhanced application
        print("Starting VinylVision with enhanced UI...")
        app = EnhancedVinylVisionWindow()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
        
    finally:
        logger.info("VinylVision application ended")


if __name__ == "__main__":
    main()
