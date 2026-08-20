import sys
from utils.config import load_config
from core.album_pipeline import AlbumDataPipeline

def main():
    album_query = None
    if len(sys.argv) > 1:
        album_query = " ".join(sys.argv[1:])

    print(f"--> Caricamento configurazione e pipeline per: '{album_query}'...")
    config = load_config()

    pipeline = AlbumDataPipeline(
        discogs_key=config.discogs.consumer_key,
        discogs_secret=config.discogs.consumer_secret
    )

    if not pipeline.initialize():
        print("[ERRORE] Inizializzazione pipeline fallita.")
        return

    print("--> Ricerca su Discogs, download copertina e calcolo embedding...")
    results = pipeline.process_album(album_query)

    if results:
        print(f"\n[SUCCESSO] Album inserito correttamente nel database locale!")
        for res in results:
            meta = res.get('metadata', {})
            print(f" - Titolo: {meta.get('title')}")
            print(f" - Artista: {meta.get('artist')}")
            print(f" - Anno: {meta.get('year')}")
        print(f"\nTotale album nel DB: {pipeline.database.get_album_count()}")
    else:
        print("[FALLITO] Nessun album trovato o errore durante l'estrazione delle feature.")

if __name__ == "__main__":
    main()