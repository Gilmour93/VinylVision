#!/usr/bin/env python3
"""
VinylVision Demo Launcher

Quick demo launcher for VinylVision application.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch VinylVision demo."""
    print("🎵 VinylVision - Real-time Album Recognition")
    print("=" * 50)
    print()
    
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    try:
        print("Starting VinylVision Enhanced UI...")
        
        # Import and run the enhanced UI
        from main import main as run_main
        run_main()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Virtual environment is activated: source venv/bin/activate")
        print("2. All dependencies are installed: pip install -r requirements.txt")
        print("3. Camera is available and not in use by other applications")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
