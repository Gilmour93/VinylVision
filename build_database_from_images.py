3#!/usr/bin/env python3
"""
Build VinylVision Vector Database from Album Cover Images

This script processes the album cover images in /images directory and builds
a comprehensive vector database for recognition.
"""

import os
import sys
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from core.database import VectorDatabase
from models.efficientnet import AlbumFeatureExtractor
from utils.image_processing import preprocess_for_model


class DatabaseBuilder:
    """Builds vector database from album cover images."""
    
    def __init__(self):
        self.database = VectorDatabase()
        self.extractor = AlbumFeatureExtractor()
        self.images_dir = Path("images")
        self.csv_file = Path("database.csv")
        
        # Statistics
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        
    def load_metadata(self) -> Dict[str, Dict]:
        """Load album metadata from CSV file."""
        print("📊 Loading album metadata from CSV...")
        
        metadata = {}
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    artist = row['Artist Name'].strip()
                    album = row['Album Title'].strip()
                    
                    # Create key that matches image filename pattern
                    key = f"{artist}-{album}"
                    
                    if key not in metadata:
                        metadata[key] = {
                            'artist': artist,
                            'album': album,
                            'year': row.get('Year', '').strip(),
                            'genre': row.get('Genre', '').strip(),
                            'style': row.get('Style', '').strip(),
                            'label': row.get('Record Label', '').strip(),
                            'discogs_id': row.get('Discogs Release ID', '').strip(),
                            'tracks': []
                        }
                    
                    # Add track info
                    track = row.get('Track Title', '').strip()
                    if track:
                        metadata[key]['tracks'].append(track)
            
            print(f"✅ Loaded metadata for {len(metadata)} unique albums")
            return metadata
            
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return {}
    
    def find_image_files(self) -> List[Path]:
        """Find all album cover image files."""
        print("🔍 Scanning for album cover images...")
        
        if not self.images_dir.exists():
            print(f"❌ Images directory not found: {self.images_dir}")
            return []
        
        # Find all JPEG files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
            image_files.extend(self.images_dir.glob(ext))
        
        print(f"✅ Found {len(image_files)} image files")
        return image_files
    
    def extract_album_info_from_filename(self, filename: str) -> Tuple[str, str]:
        """Extract artist and album from filename."""
        # Remove extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Split on first dash (artist-album format)
        if '-' in name_without_ext:
            parts = name_without_ext.split('-', 1)
            artist = parts[0].strip()
            album = parts[1].strip()
            return artist, album
        else:
            # If no dash, treat entire name as album with unknown artist
            return "Unknown Artist", name_without_ext.strip()
    
    def process_image(self, image_path: Path, metadata: Dict) -> Optional[Dict]:
        """Process a single album cover image."""
        try:
            # Extract album info from filename
            artist, album = self.extract_album_info_from_filename(image_path.name)
            key = f"{artist}-{album}"
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️ Could not load image: {image_path.name}")
                return None
            
            # Preprocess image
            processed_image = preprocess_for_model(image, target_size=(224, 224), enhance=True)
            
            # Extract features
            features = self.extractor.extract_features(processed_image)
            if features is None:
                print(f"⚠️ Could not extract features: {image_path.name}")
                return None
            
            # Get metadata if available
            album_metadata = metadata.get(key, {})
            if not album_metadata:
                # Create basic metadata from filename
                album_metadata = {
                    'artist': artist,
                    'album': album,
                    'year': 'Unknown',
                    'genre': 'Unknown',
                    'style': 'Unknown',
                    'label': 'Unknown',
                    'discogs_id': '',
                    'tracks': []
                }
            
            # Add image path to metadata
            album_metadata['image_path'] = str(image_path)
            album_metadata['filename'] = image_path.name
            
            return {
                'id': key,
                'features': features,
                'metadata': album_metadata
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            return None
    
    def build_database(self, batch_size: int = 100, max_albums: int = None):
        """Build the complete vector database."""
        print("🚀 Building VinylVision Vector Database")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Initialize components
        print("🔧 Initializing components...")
        
        if not self.extractor.load_model():
            print("❌ Failed to load feature extraction model")
            return False
        print("✅ Feature extraction model loaded")
        
        if not self.database.initialize():
            print("❌ Failed to initialize vector database")
            return False
        print("✅ Vector database initialized")
        
        # Load metadata
        metadata = self.load_metadata()
        
        # Find image files
        image_files = self.find_image_files()
        if not image_files:
            print("❌ No image files found")
            return False
        
        # Limit number of albums if specified
        if max_albums:
            image_files = image_files[:max_albums]
            print(f"📝 Processing first {len(image_files)} albums")
        
        print(f"\n🎵 Processing {len(image_files)} album covers...")
        print("=" * 60)
        
        # Process images in batches
        batch_data = []
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processing albums")):
            result = self.process_image(image_path, metadata)
            
            self.processed_count += 1
            
            if result:
                batch_data.append(result)
                self.success_count += 1
                
                # Log progress for known albums
                artist = result['metadata']['artist']
                album = result['metadata']['album']
                if i < 10 or (i + 1) % 100 == 0:
                    print(f"✅ {i+1:4d}: {artist} - {album}")
            else:
                self.error_count += 1
            
            # Store batch when it's full
            if len(batch_data) >= batch_size:
                self._store_batch(batch_data)
                batch_data = []
        
        # Store remaining batch
        if batch_data:
            self._store_batch(batch_data)
        
        # Print final statistics
        self._print_final_stats()
        
        return True
    
    def _store_batch(self, batch_data: List[Dict]):
        """Store a batch of processed albums to the database."""
        try:
            embeddings = []
            metadatas = []
            ids = []
            
            for item in batch_data:
                embeddings.append(item['features'].tolist())
                metadatas.append(item['metadata'])
                ids.append(item['id'])
            
            # Add to database
            self.database.add_embeddings(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            print(f"❌ Error storing batch: {e}")
    
    def _print_final_stats(self):
        """Print final processing statistics."""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 DATABASE BUILD COMPLETE")
        print("=" * 60)
        print(f"📈 Total albums processed: {self.processed_count}")
        print(f"✅ Successfully added: {self.success_count}")
        print(f"❌ Errors encountered: {self.error_count}")
        print(f"📊 Success rate: {self.success_count/self.processed_count*100:.1f}%")
        print(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
        print(f"⚡ Processing rate: {self.processed_count/elapsed_time:.1f} albums/second")
        
        # Database statistics
        try:
            collection = self.database.collection
            if collection:
                total_count = collection.count()
                print(f"💾 Database now contains: {total_count} albums")
        except:
            pass
        
        print("\n🎉 VinylVision database ready for recognition!")
    
    def test_recognition(self, test_image_path: str = "images/Aerosmith-Toys in the Attic.jpeg"):
        """Test recognition with a specific album."""
        print(f"\n🧪 Testing recognition with: {test_image_path}")
        
        try:
            # Load test image
            image = cv2.imread(test_image_path)
            if image is None:
                print(f"❌ Could not load test image: {test_image_path}")
                return
            
            # Process image
            processed = preprocess_for_model(image, target_size=(224, 224))
            features = self.extractor.extract_features(processed)
            
            if features is None:
                print("❌ Could not extract features from test image")
                return
            
            # Search database
            results = self.database.search_similar(features, n_results=5)
            
            if results:
                print("🎯 Recognition results:")
                for i, result in enumerate(results):
                    metadata = result.get('metadata', {})
                    confidence = result.get('confidence', 0)
                    artist = metadata.get('artist', 'Unknown')
                    album = metadata.get('album', 'Unknown')
                    print(f"   {i+1}. {confidence:.3f} - {artist} - {album}")
            else:
                print("❌ No matches found")
                
        except Exception as e:
            print(f"❌ Test error: {e}")


def main():
    """Main function to build the database."""
    print("🎵 VinylVision Database Builder")
    print("Building vector database from album cover images...")
    print()
    
    builder = DatabaseBuilder()
    
    # Ask user for options
    print("Options:")
    print("1. Build full database (4700+ albums, ~30-60 minutes)")
    print("2. Build test database (150 albums, ~3-4 minutes - includes Aerosmith)")
    print("3. Build medium database (500 albums, ~10-15 minutes)")
    
    choice = input("Enter choice (1/2/3) [default 2]: ").strip() or "2"
    
    max_albums = None
    if choice == "2":
        max_albums = 150
        print("📝 Building test database with 150 albums (includes Aerosmith)...")
    elif choice == "3":
        max_albums = 500
        print("📝 Building medium database with 500 albums...")
    else:
        print("📝 Building full database with all albums...")
    
    # Build database
    success = builder.build_database(max_albums=max_albums)
    
    if success:
        # Test recognition
        builder.test_recognition()
        
        print("\n🚀 Database build completed successfully!")
        print("💡 You can now run the main application:")
        print("   python demo.py")
        
    else:
        print("❌ Database build failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
