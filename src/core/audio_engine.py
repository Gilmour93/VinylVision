"""
Audio Engine for VinylVision: Real-time FFT with Auto-Gain, 
Shazam Fingerprinting with Measured Latency Compensation, and Synced Lyrics.
"""

import threading
import time
import asyncio
import numpy as np
import sounddevice as sd
import requests
import io
import wave
import re
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
from shazamio import Shazam


class AudioEngine:
    def __init__(self, sample_rate: int = 44100, block_size: int = 1024, n_fft_bands: int = 24):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_fft_bands = n_fft_bands
        
        self.running = False
        self.stream: Optional[sd.InputStream] = None
        self.fft_data = np.zeros(self.n_fft_bands, dtype=np.float32)
        
        # Auto-Gain per equalizzatore FFT
        self.peak_energy = 0.05
        
        # Gestione Tracking Brano, Offset e Testi
        self.current_artist: Optional[str] = None
        self.current_track: Optional[str] = None
        self.current_duration: float = 0.0
        self.current_offset: float = 0.0
        self.playback_start_time: float = 0.0
        self.lyrics_lines: List[Tuple[float, str]] = []
        self.is_identifying = False

        self.shazam = Shazam()

    def start(self):
        """Avvia la cattura streaming audio con rilevamento automatico del formato supportato."""
        if self.running:
            return
            
        fallback_configs = [
            (self.sample_rate, 1),
            (48000, 1),
            (44100, 2),
            (48000, 2),
            (16000, 1)
        ]
        
        try:
            default_device = sd.query_devices(kind='input')
            native_rate = int(default_device.get('default_samplerate', 44100))
            fallback_configs.insert(0, (native_rate, 1))
            fallback_configs.insert(1, (native_rate, 2))
        except Exception:
            pass

        stream_opened = False
        for rate, channels in fallback_configs:
            try:
                self.stream = sd.InputStream(
                    channels=channels,
                    samplerate=rate,
                    blocksize=self.block_size,
                    callback=self._audio_callback
                )
                self.stream.start()
                self.sample_rate = rate
                self.channels = channels
                self.running = True
                stream_opened = True
                logger.info(f"Audio Engine avviato con successo: {rate} Hz, {channels} ch")
                break
            except Exception:
                continue

        if not stream_opened:
            logger.error("Impossibile aprire il microfono: nessun formato audio supportato trovato.")
            self.running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Calcolo FFT in real-time con supporto sia mono che multi-canale."""
        if status:
            pass
            
        audio_samples = indata[:, 0]
        windowed = audio_samples * np.hanning(len(audio_samples))
        fft_vals = np.abs(np.fft.rfft(windowed))
        
        # Raggruppamento logaritmico in bande
        bands = np.zeros(self.n_fft_bands, dtype=np.float32)
        indices = np.logspace(0, np.log10(len(fft_vals) - 1), self.n_fft_bands + 1).astype(int)
        
        for i in range(self.n_fft_bands):
            start_idx = indices[i]
            end_idx = max(start_idx + 1, indices[i+1])
            band_val = np.mean(fft_vals[start_idx:end_idx])
            bands[i] = band_val

        current_max = float(np.max(bands))
        self.peak_energy = max(self.peak_energy * 0.992, current_max, 0.005)
        
        normalized = bands / self.peak_energy
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = np.sqrt(normalized)

        self.fft_data = self.fft_data * 0.6 + normalized * 0.4

    def stop(self):
        """Ferma la cattura audio."""
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        logger.info("Audio Engine stopped")

    def get_fft_spectrum(self) -> np.ndarray:
        """Restituisce le altezze delle barre (0.0 - 1.0)."""
        return self.fft_data.copy()

    def get_7_lyrics_lines(self) -> Tuple[str, str, str, str, str, str, str]:
        """Restituisce le 7 righe sincronizzate (p3, p2, p1, curr, n1, n2, n3) rispetto all'offset reale misurato."""
        lines = self.lyrics_lines
        if not lines:
            return "", "", "", "", "", "", ""

        now = time.time()
        start_t = self.playback_start_time
        
        # Posizione temporale esatta nel brano
        elapsed = max(0.0, now - start_t) if start_t > 0.0 else self.current_offset

        # Trova la linea attiva (l'ultima con timestamp <= elapsed)
        current_idx = 0
        for idx, item in enumerate(lines):
            t_sec = 0.0
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    t_sec = float(item[0])
                except (ValueError, TypeError):
                    t_sec = 0.0
            elif isinstance(item, dict):
                t_sec = float(item.get('time', item.get('timestamp', item.get('seconds', 0.0))))

            if elapsed >= t_sec:
                current_idx = idx
            else:
                break

        def _get_text(i):
            if 0 <= i < len(lines):
                it = lines[i]
                if isinstance(it, (tuple, list)) and len(it) >= 2:
                    return str(it[1]).strip()
                elif isinstance(it, dict):
                    return str(it.get('text', it.get('line', ''))).strip()
                return str(it).strip()
            return ""

        p3 = _get_text(current_idx - 3)
        p2 = _get_text(current_idx - 2)
        p1 = _get_text(current_idx - 1)
        curr = _get_text(current_idx)
        n1 = _get_text(current_idx + 1)
        n2 = _get_text(current_idx + 2)
        n3 = _get_text(current_idx + 3)

        return p3, p2, p1, curr, n1, n2, n3

    def trigger_background_identify(self):
        """Avvia la scansione acustica Shazam in un thread separato."""
        if self.is_identifying or not self.running:
            return
        threading.Thread(target=self._run_async_shazam, daemon=True).start()

    def _run_async_shazam(self):
        self.is_identifying = True
        try:
            asyncio.run(self._identify_task())
        except Exception as e:
            logger.error(f"Errore identificazione Shazam: {e}")
        finally:
            self.is_identifying = False

    async def _identify_task(self):
        logger.info("[🎙] Registrazione 4.0s per Shazam...")
        record_sec = 4.0
        
        # Misurazione precisa: inizio esatto della finestra di campionamento
        t_record_start = time.time()

        try:
            # 1. Registra i campioni dal microfono
            audio_rec = sd.rec(
                int(record_sec * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            t_record_end = time.time()

            # 2. Crea buffer WAV in memoria
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_rec.tobytes())

            wav_bytes = wav_buffer.getvalue()

            # 3. Invio fingerprint ed elaborazione di rete
            logger.info("[🔍] Invio fingerprint a Shazam...")
            t_network_start = time.time()
            out = await self.shazam.recognize(wav_bytes)
            t_response_received = time.time()

            if not out or not out.get('track'):
                logger.info("[✖] Shazam: Nessun brano riconosciuto in questo frammento.")
                return

            track_info = out['track']
            title = track_info.get('title', '')
            artist = track_info.get('subtitle', '')
            matches = out.get('matches', [])
            raw_offset = float(matches[0].get('offset', 0.0)) if matches else 0.0

            # 4. MISURAZIONE RIGOROSA DEL TEMPO:
            # raw_offset corrisponde al timestamp della canzone all'istante t_record_start.
            # Per allineare il testo al cantato in cassa, anticipiamo di 0.6s
            SYNC_LEAD_SEC = 0.60  

            # T0 assoluto della traccia
            measured_playback_start = t_record_start - raw_offset - SYNC_LEAD_SEC
            
            # Posizione attuale in tempo reale al momento del rendering
            current_exact_offset = t_response_received - measured_playback_start
            network_delay = t_response_received - t_network_start

            # 5. Estrazione durata brano
            duration_sec = 0.0
            for section in track_info.get('sections', []):
                for meta in section.get('metadata', []):
                    if meta.get('title', '').lower() in ['duration', 'durata', 'length']:
                        parts = meta.get('text', '').split(':')
                        if len(parts) == 2:
                            duration_sec = float(parts[0]) * 60 + float(parts[1])

            if duration_sec == 0.0:
                try:
                    q = urllib.parse.quote(f"{artist} {title}")
                    url = f"https://itunes.apple.com/search?term={q}&entity=song&limit=1"
                    req = urllib.request.Request(url, headers={'User-Agent': 'VinylVision/1.0'})
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        data = json.loads(resp.read().decode())
                        if data.get('resultCount', 0) > 0:
                            duration_sec = data['results'][0].get('trackTimeMillis', 0) / 1000.0
                except Exception:
                    pass

            self.current_duration = duration_sec
            dur_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}" if duration_sec > 0 else "N/D"
            logger.info(
                f"[✔] Shazam Match: {artist} - {title} | "
                f"Posizione reale: {current_exact_offset:.2f}s | "
                f"Durata: {dur_str} | "
                f"Latenza rete misurata: {network_delay:.2f}s"
            )

            is_new_song = (self.current_track != title or self.current_artist != artist)
            self.current_track = title
            self.current_artist = artist
            self.current_offset = current_exact_offset
            self.playback_start_time = measured_playback_start

            if is_new_song:
                self._fetch_lyrics(artist, title)

        except Exception as e:
            logger.error(f"Errore durante l'identificazione audio: {e}")

    def _fetch_lyrics(self, artist: str, track: str):
        """Scarica i testi sincronizzati (.lrc) da LRCLIB."""
        try:
            clean_track = re.sub(r'\(.*?\)|\[.*?\]', '', track).strip()
            clean_track = re.sub(r'\s*-\s*(Remaster|Remastered|Live|Mono|Stereo|Edit|Single Version).*$', '', clean_track, flags=re.IGNORECASE).strip()
            clean_artist = artist.split('feat.')[0].split('Featuring')[0].strip()

            logger.info(f"Ricerca testi LRCLIB per: '{clean_artist}' - '{clean_track}'")

            headers = {"User-Agent": "VinylVision/1.0"}
            plain_fallback = None

            # 1. Endpoint /api/get
            resp = requests.get(
                "https://lrclib.net/api/get", 
                params={"artist_name": clean_artist, "track_name": clean_track}, 
                headers=headers, 
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("syncedLyrics"):
                    self.lyrics_lines = self._parse_lrc(data["syncedLyrics"])
                    logger.info(f"[✔] Testi sincronizzati agganciati (Direct): {len(self.lyrics_lines)} righe")
                    return
                elif data.get("plainLyrics"):
                    plain_fallback = data.get("plainLyrics")

            # 2. Endpoint /api/search
            resp_search = requests.get(
                "https://lrclib.net/api/search", 
                params={"q": f"{clean_artist} {clean_track}"}, 
                headers=headers, 
                timeout=5
            )
            if resp_search.status_code == 200:
                results = resp_search.json()
                if isinstance(results, list):
                    for res in results:
                        if res.get("syncedLyrics"):
                            self.lyrics_lines = self._parse_lrc(res["syncedLyrics"])
                            logger.info(f"[✔] Testi sincronizzati agganciati (Search): {len(self.lyrics_lines)} righe")
                            return
                        if not plain_fallback and res.get("plainLyrics"):
                            plain_fallback = res.get("plainLyrics")

            # 3. Fallback testo piano distribuito
            if plain_fallback:
                raw_lines = [l.strip() for l in plain_fallback.splitlines() if l.strip()]
                if raw_lines:
                    self.lyrics_lines = [(i * 4.5, line) for i, line in enumerate(raw_lines)]
                    return

            self.lyrics_lines = [(0.0, "Testo non disponibile")]

        except Exception as e:
            logger.error(f"Errore recupero testi: {e}")
            self.lyrics_lines = [(0.0, "Errore connessione testi")]

    def _parse_lrc(self, lrc_text: str) -> List[Tuple[float, str]]:
        lines = []
        for raw_line in lrc_text.splitlines():
            raw_line = raw_line.strip()
            if raw_line.startswith('[') and ']' in raw_line:
                tag_end = raw_line.find(']')
                time_str = raw_line[1:tag_end]
                text = raw_line[tag_end+1:].strip()
                try:
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        minutes = float(parts[0])
                        seconds = float(parts[1])
                        total_sec = minutes * 60.0 + seconds
                        lines.append((total_sec, text))
                except ValueError:
                    continue
        lines.sort(key=lambda x: x[0])
        return lines