"""
Vector database operations for album embeddings storage and retrieval.

Uses ChromaDB for efficient similarity search.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from loguru import logger
import numpy as np


class VectorDatabase:
    """Manages album embeddings and metadata storage."""
    
    def __init__(self, db_path: str = "data/embeddings"):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to database storage directory
        """
        self.db_path = db_path
        self.client = None
        self.collection = None
        
    def initialize(self) -> bool:
        """
        Initialize ChromaDB client and collection.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Ensure database directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="album_embeddings",
                metadata={"description": "VinylVision album cover embeddings"}
            )
            
            logger.info(f"Vector database initialized at {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            return False
    
    def add_album(self, 
                  album_id: str, 
                  embedding: np.ndarray, 
                  metadata: Dict[str, Any]) -> bool:
        try:
            if self.collection is None:
                logger.error("Database not initialized")
                return False
            
            # Normalizza il vettore prima di salvarlo nel DB
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding_list = (embedding / norm).tolist()
            else:
                embedding_list = embedding.tolist()
            
            clean_metadata = self._clean_metadata(metadata)
            
            self.collection.add(
                ids=[str(album_id)],
                embeddings=[embedding_list],
                metadatas=[clean_metadata]
            )
            
            logger.debug(f"Added album {album_id} to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding album to database: {e}")
            return False
    
    def add_embeddings(self,
                      embeddings: List[List[float]], 
                      metadatas: List[Dict[str, Any]],
                      ids: List[str]) -> bool:
        """
        Add multiple embeddings to database in batch.
        
        Args:
            embeddings: List of feature vectors
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
            
        Returns:
            bool: True if addition successful, False otherwise
        """
        try:
            if self.collection is None:
                logger.error("Database not initialized")
                return False
            
            # Clean all metadata
            clean_metadatas = [self._clean_metadata(metadata) for metadata in metadatas]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=clean_metadatas
            )
            
            logger.info(f"Added {len(ids)} albums to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings batch: {e}")
            return False
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to only include types supported by ChromaDB.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Dict[str, Any]: Cleaned metadata with supported types only
        """
        clean_meta = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value:  # Only if list is not empty
                    if all(isinstance(item, str) for item in value):
                        clean_meta[key] = ", ".join(value)
                    else:
                        clean_meta[key] = ", ".join(str(item) for item in value)
            elif value is not None:
                # Convert other types to string
                clean_meta[key] = str(value)
        
        return clean_meta

    def album_exists(self, album_id: str) -> bool:
        """Verifica se un album è già presente nel database."""
        try:
            if self.collection is None:
                return False
            result = self.collection.get(ids=[str(album_id)])
            return bool(result and result.get('ids') and len(result['ids']) > 0)
        except Exception:
            return False
    
    def search_similar(self, 
                      query_embedding: np.ndarray,
                      n_results: int = 5,
                      confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar albums using cosine similarity."""
        try:
            if self.collection is None:
                logger.error("Database not initialized")
                return []
            
            # Normalizza il vettore di query (L2 norm) per renderlo comparabile
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_norm = (query_embedding / norm).tolist()
            else:
                query_norm = query_embedding.tolist()
            
            results = self.collection.query(
                query_embeddings=[query_norm],
                n_results=n_results
            )
            
            similar_albums = []
            if results['ids'] and results['ids'][0]:
                for i, album_id in enumerate(results['ids'][0]):
                    raw_distance = results['distances'][0][i]
                    
                    # Con 'cosine' in ChromaDB: distance = 1 - cosine_similarity
                    # Quindi similarity = 1 - distance
                    # Con L2: calcoliamo una somiglianza percentuale normalizzata
                    similarity = max(0.0, 1.0 - raw_distance)
                    
                    # LOG DI DIAGNOSTICA: vedi cosa rileva la webcam in tempo reale
                    logger.info(f"Match candidato ID: {album_id} | Distanza grezza: {raw_distance:.4f} | Somiglianza calcolata: {similarity:.2%}")
                    
                    if similarity >= confidence_threshold:
                        album_data = {
                            'id': album_id,
                            'similarity': similarity,
                            'confidence': similarity,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                        }
                        similar_albums.append(album_data)
            
            return similar_albums
            
        except Exception as e:
            logger.error(f"Error searching similar albums: {e}")
            return []
    
    def get_album_count(self) -> int:
        """
        Get total number of albums in database.
        
        Returns:
            int: Number of albums stored
        """
        try:
            if self.collection is None:
                return 0
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting album count: {e}")
            return 0
    
    def close(self):
        """Close database connection."""
        if self.client:
            logger.info("Vector database closed")
