"""
Album data pipeline for VinylVision.

Handles album cover download, feature extraction, and database storage.
"""

import os
import requests
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image
import io
from loguru import logger

try:
    # Try relative imports first (when used as package)
    from .discogs_client import DiscogsClient
    from .database import VectorDatabase
    from ..models.efficientnet import AlbumFeatureExtractor
    from ..utils.image_processing import preprocess_for_model
except ImportError:
    # Fallback to absolute imports (when used directly)
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent))
    
    from core.discogs_client import DiscogsClient
    from core.database import VectorDatabase
    from models.efficientnet import AlbumFeatureExtractor
    from utils.image_processing import preprocess_for_model


class AlbumDataPipeline:
    """Manages the complete album data processing pipeline."""
    
    def __init__(self, 
                 discogs_key: str,
                 discogs_secret: str,
                 db_path: str = "data/embeddings"):
        """
        Initialize album data pipeline.
        
        Args:
            discogs_key: Discogs API consumer key
            discogs_secret: Discogs API consumer secret  
            db_path: Path to vector database
        """
        self.discogs_client = DiscogsClient(discogs_key, discogs_secret)
        self.database = VectorDatabase(db_path)
        self.feature_extractor = AlbumFeatureExtractor()
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize Discogs client
            if not self.discogs_client.initialize():
                logger.error("Failed to initialize Discogs client")
                return False
            
            # Initialize database
            if not self.database.initialize():
                logger.error("Failed to initialize database")
                return False
            
            # Initialize feature extractor
            if not self.feature_extractor.load_model():
                logger.error("Failed to initialize feature extractor")
                return False
            
            self.is_initialized = True
            logger.info("Album data pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing album pipeline: {e}")
            return False
    
    """
    def download_album_cover(self, image_url: str) -> Optional[np.ndarray]:

        try:
            if not image_url:
                logger.warning("No image URL provided")
                return None
            
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            logger.debug(f"Downloaded album cover: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error downloading album cover from {image_url}: {e}")
            return None
    """


    def download_album_cover(self, image_url: str) -> Optional[np.ndarray]:
        """
        Download album cover image from URL.
        """
        try:
            if not image_url:
                logger.warning("No image URL provided")
                return None
            
            # Headers specifici per il CDN delle immagini di Discogs (i.discogs.com)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.discogs.com/'
            }
            
            response = requests.get(image_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Conversione in PIL Image e RGB
            image = Image.open(io.BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            logger.debug(f"Downloaded album cover: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error downloading album cover from {image_url}: {e}")
            return None
    
    def process_album(self, album_query: str) -> List[Dict[str, Any]]:
        """
        Process album by searching Discogs and extracting features.
        
        Args:
            album_query: Search query for the album
            
        Returns:
            List[Dict[str, Any]]: List of processed album data
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return []
        
        try:
            # Search for albums on Discogs
            albums = self.discogs_client.search_albums(album_query, max_results=5)
            
            if not albums:
                logger.warning(f"No albums found for query: {album_query}")
                return []
            
            processed_albums = []
            
            for album_data in albums:
                try:
                    processed_album = self._process_single_album(album_data)
                    if processed_album:
                        processed_albums.append(processed_album)
                        
                except Exception as e:
                    logger.error(f"Error processing album {album_data.get('id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Processed {len(processed_albums)} albums for query: {album_query}")
            return processed_albums
            
        except Exception as e:
            logger.error(f"Error processing album query '{album_query}': {e}")
            return []
    
    def _process_single_album(self, album_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single album: download cover, extract features, store in database.
        
        Args:
            album_data: Album metadata from Discogs
            
        Returns:
            Optional[Dict[str, Any]]: Processed album data or None if failed
        """
        try:
            album_id = album_data.get('id')
            if not album_id:
                logger.error("Album missing ID")
                return None
            
            # Check if album already exists in database
            existing_count = self.database.get_album_count()
            if existing_count > 0:
                pass
            
            # Download album cover
            image_url = album_data.get('image_url')
            if not image_url:
                logger.warning(f"No image URL for album {album_id}")
                return None
            
            album_image = self.download_album_cover(image_url)
            if album_image is None:
                logger.warning(f"Failed to download cover for album {album_id}")
                return None
            
            # --- NOVITÀ: Salva la copertina sul disco locale ---
            covers_dir = "data/covers"
            os.makedirs(covers_dir, exist_ok=True)
            local_cover_path = os.path.join(covers_dir, f"{album_id}.jpg")
            
            # Salva l'array RGB come JPEG
            Image.fromarray(album_image).save(local_cover_path, quality=90)
            
            # Aggiungi il path locale ai metadati per ChromaDB
            album_data['local_cover_path'] = local_cover_path
            # ----------------------------------------------------
            
            # Preprocess image for model
            preprocessed_image = preprocess_for_model(
                album_image, 
                target_size=(224, 224),
                enhance=True
            )
            
            # Extract features specificando is_bgr=False (l'immagine scaricata è RGB)
            features = self.feature_extractor.extract_features(preprocessed_image, is_bgr=False)
            if features is None:
                logger.error(f"Failed to extract features for album {album_id}")
                return None
            
            # Store in database
            success = self.database.add_album(
                album_id=str(album_id),
                embedding=features,
                metadata=album_data
            )
            
            if success:
                processed_data = {
                    'album_id': album_id,
                    'metadata': album_data,
                    'features_shape': features.shape,
                    'image_shape': album_image.shape,
                    'stored_in_db': True
                }
                logger.info(f"Successfully processed album: {album_data.get('artist', 'Unknown')} - {album_data.get('title', 'Unknown')}")
                return processed_data
            else:
                logger.error(f"Failed to store album {album_id} in database")
                return None
                
        except Exception as e:
            logger.error(f"Error processing single album: {e}")
            return None

    def sync_user_collection(self, username: str, max_albums: Optional[int] = None) -> Dict[str, Any]:
        """
        Scarica, processa e indicizza in modo incrementale la collezione Discogs.
        """
        import time
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return {"error": "Pipeline non inizializzata"}
            
        stats = {
            "total_found": 0,
            "imported": 0,
            "skipped": 0,
            "errors": 0
        }
        
        logger.info(f"Avvio sincronizzazione incrementale per: {username}")
        page = 1
        
        while True:
            data = self.discogs_client.get_user_collection(username, page=page, per_page=50)
            if not data or 'releases' not in data:
                break
                
            releases = data.get('releases', [])
            if not releases:
                break
                
            total_items = data.get('pagination', {}).get('items', len(releases))
            stats["total_found"] = total_items
            pages_total = data.get('pagination', {}).get('pages', 1)
            
            for item in releases:
                if max_albums and stats["imported"] >= max_albums:
                    return stats
                    
                basic_info = item.get('basic_information', {})
                release_id = str(basic_info.get('id'))
                artists = ", ".join([a.get('name', '') for a in basic_info.get('artists', [])])
                title = basic_info.get('title', 'Unknown Title')
                
                # CONTROLLO INCREMENTALE: se l'album c'è già, saltalo subito!
                if self.database.album_exists(release_id):
                    stats["skipped"] += 1
                    logger.debug(f"[GIÀ PRESENTE] Saltato: {artists} - {title}")
                    continue
                
                # Se è un disco nuovo, procedi con download ed embedding
                cover_image = basic_info.get('cover_image') or basic_info.get('thumb')
                album_data = {
                    'id': release_id,
                    'artist': artists,
                    'title': title,
                    'year': basic_info.get('year', 0),
                    'image_url': cover_image,
                    'genre': ", ".join(basic_info.get('genres', [])),
                    'label': basic_info.get('labels', [{}])[0].get('name', 'Unknown') if basic_info.get('labels') else 'Unknown'
                }
                
                try:
                    res = self._process_single_album(album_data)
                    if res and res.get('stored_in_db'):
                        stats["imported"] += 1
                        logger.info(f"[{stats['imported']} NUOVO] Indicizzato: {artists} - {title}")
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    logger.error(f"Errore su {artists} - {title}: {e}")
                    stats["errors"] += 1
                
                # Pausa solo per i download effettivi (rate limit)
                time.sleep(1.0)
            
            if page >= pages_total:
                break
            page += 1
            
        logger.info(f"Sincronizzazione completata! Nuovi importati: {stats['imported']}, Già presenti: {stats['skipped']}")
        return stats
    
    def search_similar_albums(self, 
                             image: np.ndarray,
                             n_results: int = 5,
                             confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        # ...
        try:
            preprocessed = preprocess_for_model(
                image,
                target_size=(224, 224),
                enhance=True
            )
            
            # is_bgr=True perché il frame dalla webcam OpenCV è in formato BGR
            features = self.feature_extractor.extract_features(preprocessed, is_bgr=True)
            if features is None:
                return []
            
            results = self.database.search_similar(
                query_embedding=features,
                n_results=n_results,
                confidence_threshold=confidence_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Error searching similar albums: {e}")
            return []
    
    def build_database_from_popular_albums(self, num_albums: int = 100) -> Dict[str, Any]:
        """
        Build initial database with popular albums.
        
        Args:
            num_albums: Number of albums to process
            
        Returns:
            Dict[str, Any]: Build statistics
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return {"error": "Pipeline not initialized"}
        
        # List of popular album search queries
        popular_queries = [
            "The Beatles Abbey Road",
            "Pink Floyd Dark Side of the Moon",
            "Led Zeppelin IV",
            "The Rolling Stones Sticky Fingers",
            "Fleetwood Mac Rumours",
            "Nirvana Nevermind",
            "Michael Jackson Thriller",
            "Queen A Night at the Opera",
            "The Beatles Sgt Pepper",
            "Bob Dylan Highway 61 Revisited",
            "The Velvet Underground Nico",
            "David Bowie Ziggy Stardust",
            "Radiohead OK Computer",
            "The Beach Boys Pet Sounds",
            "Marvin Gaye What's Going On",
            "Prince Purple Rain",
            "The Clash London Calling",
            "Johnny Cash At Folsom Prison",
            "Miles Davis Kind of Blue",
            "John Coltrane A Love Supreme",
            "Aretha Franklin I Never Loved a Man",
            "Stevie Wonder Songs in the Key of Life",
            "The Who Tommy",
            "Jimi Hendrix Are You Experienced",
            "Cream Disraeli Gears",
            "The Doors The Doors",
            "Janis Joplin Pearl",
            "Carole King Tapestry",
            "Simon and Garfunkel Bridge Over Troubled Water",
            "Joni Mitchell Blue"
        ]
        
        stats = {
            "albums_processed": 0,
            "albums_stored": 0,
            "queries_completed": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
        
        import time
        stats["start_time"] = time.time()
        
        try:
            logger.info(f"Building database with {num_albums} albums from popular queries")
            
            albums_needed = num_albums
            query_index = 0
            
            while albums_needed > 0 and query_index < len(popular_queries):
                query = popular_queries[query_index]
                query_index += 1
                
                try:
                    # Process albums for this query
                    processed = self.process_album(query)
                    stats["queries_completed"] += 1
                    
                    for album in processed:
                        if album.get("stored_in_db", False):
                            stats["albums_stored"] += 1
                            albums_needed -= 1
                            
                            if albums_needed <= 0:
                                break
                        
                        stats["albums_processed"] += 1
                    
                    # Log progress
                    if stats["queries_completed"] % 5 == 0:
                        logger.info(f"Progress: {stats['albums_stored']} albums stored, {stats['queries_completed']} queries completed")
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
                    stats["errors"] += 1
                    continue
            
            stats["end_time"] = time.time()
            duration = stats["end_time"] - stats["start_time"]
            
            final_count = self.database.get_album_count()
            
            logger.info(f"Database build completed:")
            logger.info(f"  Total albums in database: {final_count}")
            logger.info(f"  Albums processed: {stats['albums_processed']}")
            logger.info(f"  Albums stored: {stats['albums_stored']}")
            logger.info(f"  Queries completed: {stats['queries_completed']}")
            logger.info(f"  Duration: {duration:.1f} seconds")
            
            stats["final_database_count"] = final_count
            stats["duration_seconds"] = duration
            
            return stats
            
        except Exception as e:
            logger.error(f"Error building database: {e}")
            stats["error"] = str(e)
            return stats
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        if not self.is_initialized:
            return {"error": "Pipeline not initialized"}
        
        try:
            album_count = self.database.get_album_count()
            model_info = self.feature_extractor.get_model_info()
            
            return {
                "database_album_count": album_count,
                "feature_extractor_model": model_info.get("model_name", "Unknown"),
                "feature_dimension": model_info.get("feature_dim", "Unknown"),
                "device": model_info.get("device", "Unknown"),
                "is_initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close all pipeline components."""
        if self.database:
            self.database.close()
        logger.info("Album data pipeline closed")
