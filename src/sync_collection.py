import sys
from utils.config import load_config
from core.album_pipeline import AlbumDataPipeline

def main():
    if len(sys.argv) < 2:
        print("\n[USAGE]: python .\\src\\sync_collection.py <YOUR_DISCOGS_USERNAME> [ALBUM_LIMIT]")
        print("Example: python .\\src\\sync_collection.py my_username")
        print("Example (first 10 albums): python .\\src\\sync_collection.py my_username 10\n")
        return

    username = sys.argv[1]
    max_albums = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"--> Initializing synchronization for '{username}'...")
    config = load_config()

    pipeline = AlbumDataPipeline(
        discogs_key=config.discogs.consumer_key,
        discogs_secret=config.discogs.consumer_secret
    )

    if not pipeline.initialize():
        print("[ERROR] Failed to initialize pipeline.")
        return

    stats = pipeline.sync_user_collection(username, max_albums=max_albums)
    print("\n" + "="*40)
    print("SUMMARY OF SYNCHRONIZATION")
    print("="*40)
    print(f"Total found on Discogs: {stats.get('total_found', 0)}")
    print(f"Successfully imported:    {stats.get('imported', 0)}")
    print(f"Errors:                    {stats.get('errors', 0)}")
    print(f"Total albums in DB now:  {pipeline.database.get_album_count()}")
    print("="*40)

if __name__ == "__main__":
    main()