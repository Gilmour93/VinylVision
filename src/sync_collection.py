import sys
from utils.config import load_config
from core.album_pipeline import AlbumDataPipeline

def main():
    if len(sys.argv) < 2:
        print("\n[USO]: python .\\src\\sync_collection.py <TUO_USERNAME_DISCOGS> [LIMITE_ALBUM]")
        print("Esempio: python .\\src\\sync_collection.py mio_username")
        print("Esempio (primi 10 dischi): python .\\src\\sync_collection.py mio_username 10\n")
        return

    username = sys.argv[1]
    max_albums = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"--> Inizializzazione sincronizzazione collezione per '{username}'...")
    config = load_config()

    pipeline = AlbumDataPipeline(
        discogs_key=config.discogs.consumer_key,
        discogs_secret=config.discogs.consumer_secret
    )

    if not pipeline.initialize():
        print("[ERRORE] Inizializzazione pipeline fallita.")
        return

    stats = pipeline.sync_user_collection(username, max_albums=max_albums)
    print("\n" + "="*40)
    print("RIEPILOGO SINCRONIZZAZIONE")
    print("="*40)
    print(f"Totale trovati su Discogs: {stats.get('total_found', 0)}")
    print(f"Importati con successo:    {stats.get('imported', 0)}")
    print(f"Errori:                    {stats.get('errors', 0)}")
    print(f"Totale dischi nel DB ora:  {pipeline.database.get_album_count()}")
    print("="*40)

if __name__ == "__main__":
    main()