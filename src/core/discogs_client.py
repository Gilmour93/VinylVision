"""
Discogs API integration for album metadata retrieval.

Handles authentication, rate limiting, and data fetching from Discogs.
"""

import time
from typing import Dict, Any, Optional, List
import discogs_client
from loguru import logger


class DiscogsClient:
    """Manages Discogs API interactions."""
    
    def __init__(self, consumer_key: str, consumer_secret: str):
        """
        Initialize Discogs client.
        
        Args:
            consumer_key: Discogs API consumer key
            consumer_secret: Discogs API consumer secret
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.client = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 60 requests per minute = 1 per second
        
    def initialize(self) -> bool:
        """
        Initialize Discogs API client.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.client = discogs_client.Client(
                'VinylVision/1.0',
                consumer_key=self.consumer_key,
                consumer_secret=self.consumer_secret
            )
            
            # Test connection
            self._rate_limit()
            test_search = self.client.search('test', type='release')
            logger.info("Discogs API client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Discogs client: {e}")
            return False
    
    def _rate_limit(self):
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_albums(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for albums on Discogs.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of album metadata
        """
        try:
            if self.client is None:
                logger.error("Discogs client not initialized")
                return []
            
            self._rate_limit()
            
            # Search for releases
            results = self.client.search(query, type='release')
            
            albums = []
            for i, release in enumerate(results[:max_results]):
                try:
                    self._rate_limit()
                    
                    album_data = {
                        'id': str(release.id),
                        'title': getattr(release, 'title', 'Unknown'),
                        'artist': self._get_artist_name(release),
                        'year': getattr(release, 'year', None),
                        'label': self._get_label_name(release),
                        'genre': self._get_genres(release),
                        'image_url': self._get_image_url(release),
                        'discogs_url': release.url if hasattr(release, 'url') else None
                    }
                    albums.append(album_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing release {release.id}: {e}")
                    continue
            
            logger.info(f"Found {len(albums)} albums for query: {query}")
            return albums
            
        except Exception as e:
            logger.error(f"Error searching albums: {e}")
            return []
    
    def get_album_details(self, release_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific album.
        
        Args:
            release_id: Discogs release ID
            
        Returns:
            Optional[Dict[str, Any]]: Album details or None if not found
        """
        try:
            if self.client is None:
                logger.error("Discogs client not initialized")
                return None
            
            self._rate_limit()
            release = self.client.release(int(release_id))
            
            album_data = {
                'id': str(release.id),
                'title': getattr(release, 'title', 'Unknown'),
                'artist': self._get_artist_name(release),
                'year': getattr(release, 'year', None),
                'label': self._get_label_name(release),
                'genre': self._get_genres(release),
                'styles': self._get_styles(release),
                'format': self._get_format(release),
                'country': getattr(release, 'country', None),
                'image_url': self._get_image_url(release),
                'tracklist': self._get_tracklist(release),
                'discogs_url': release.url if hasattr(release, 'url') else None
            }
            
            return album_data
            
        except Exception as e:
            logger.error(f"Error getting album details for {release_id}: {e}")
            return None
    
    def _get_artist_name(self, release) -> str:
        """Extract artist name from release."""
        try:
            if hasattr(release, 'artists') and release.artists:
                return release.artists[0].name
            return 'Unknown Artist'
        except:
            return 'Unknown Artist'
    
    def _get_label_name(self, release) -> str:
        """Extract label name from release."""
        try:
            if hasattr(release, 'labels') and release.labels:
                return release.labels[0].name
            return 'Unknown Label'
        except:
            return 'Unknown Label'
    
    def _get_genres(self, release) -> List[str]:
        """Extract genres from release."""
        try:
            if hasattr(release, 'genres'):
                return list(release.genres)
            return []
        except:
            return []
    
    def _get_styles(self, release) -> List[str]:
        """Extract styles from release."""
        try:
            if hasattr(release, 'styles'):
                return list(release.styles)
            return []
        except:
            return []
    
    def _get_format(self, release) -> str:
        """Extract format from release."""
        try:
            if hasattr(release, 'formats') and release.formats:
                return release.formats[0]['name']
            return 'Unknown Format'
        except:
            return 'Unknown Format'
    
    def _get_image_url(self, release) -> Optional[str]:
        """Extract image URL from release."""
        try:
            if hasattr(release, 'images') and release.images:
                return release.images[0]['uri']
            return None
        except:
            return None
    
    def _get_tracklist(self, release) -> List[Dict[str, str]]:
        """Extract tracklist from release."""
        try:
            if hasattr(release, 'tracklist'):
                tracks = []
                for track in release.tracklist:
                    track_data = {
                        'position': getattr(track, 'position', ''),
                        'title': getattr(track, 'title', ''),
                        'duration': getattr(track, 'duration', '')
                    }
                    tracks.append(track_data)
                return tracks
            return []
        except:
            return []
