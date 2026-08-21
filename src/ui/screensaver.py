import os
import random
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from loguru import logger
from typing import Optional, Callable, Dict, Any, List
from ui.theme import Colors, Fonts


class ScreensaverOverlay:
    """
    Full-screen ambient screensaver cycling through local album covers in data/covers/.
    Features smooth cross-fade transitions and authentic metadata from local vector DB.
    """
    def __init__(self, root: tk.Tk, config: dict, on_dismiss: Callable[[], None], pipeline: Optional[Any] = None):
        self.root = root
        self.config = config
        self.pipeline = pipeline
        self.on_dismiss = on_dismiss
        
        screensaver_cfg = config.get("screensaver", {}) if isinstance(config, dict) else {}
        self.interval_ms = int(screensaver_cfg.get("slideshow_interval_sec", 10) * 1000)
        
        self.is_active = False
        self._after_id = None
        self._fade_after_id = None
        self.overlay_frame: Optional[tk.Frame] = None
        
        self.canvas: Optional[tk.Canvas] = None
        self._active_img_ref = None
        
        # Cross-fade state
        self._current_pil_slide: Optional[Image.Image] = None
        self._target_pil_slide: Optional[Image.Image] = None
        self._fade_step = 0
        self._total_fade_steps = 12
        
        self.collection_items: List[Dict[str, Any]] = []
        self._load_local_collection()

    def _load_local_collection(self):
        """Loads album metadata from database and maps them to local cover files."""
        self.collection_items.clear()
        covers_dir = os.path.join("data", "covers")

        db = getattr(self.pipeline, 'database', None) or getattr(self.pipeline, 'db', None)
        loaded_ids = set()

        if db is not None:
            try:
                coll = getattr(db, 'collection', None)
                if coll and hasattr(coll, 'get'):
                    records = coll.get(include=['metadatas'])
                    metas = records.get('metadatas', [])
                    ids = records.get('ids', [])

                    for rec_id, meta in zip(ids, metas):
                        if not meta or not isinstance(meta, dict):
                            continue

                        title = meta.get('title') or meta.get('album') or ''
                        artist = meta.get('artist') or meta.get('artists') or ''
                        year = meta.get('year') or meta.get('released') or ''
                        disc_id = meta.get('id') or meta.get('discogs_id') or rec_id

                        cover_path = None
                        for ext in ['jpg', 'png', 'jpeg', 'webp']:
                            p = os.path.join(covers_dir, f"{disc_id}.{ext}")
                            if os.path.isfile(p):
                                cover_path = p
                                break

                        if not cover_path and meta.get('cover_image_path') and os.path.isfile(meta['cover_image_path']):
                            cover_path = meta['cover_image_path']

                        if cover_path and (title or artist):
                            self.collection_items.append({
                                "title": str(title).strip(),
                                "artist": str(artist).strip(),
                                "year": str(year).strip() if str(year).strip() not in ['N/A', '--', '0'] else "",
                                "cover_path": cover_path
                            })
                            loaded_ids.add(str(disc_id))

            except Exception as e:
                logger.warning(f"[Screensaver] Could not extract collection metadata from DB: {e}")

        if os.path.exists(covers_dir):
            for fname in os.listdir(covers_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue
                
                fpath = os.path.join(covers_dir, fname)
                basename = os.path.splitext(fname)[0]
                
                if basename in loaded_ids:
                    continue

                if " - " in basename:
                    parts = basename.split(" - ")
                    self.collection_items.append({
                        "artist": parts[0].strip(),
                        "title": parts[1].strip(),
                        "year": "",
                        "cover_path": fpath
                    })
                elif not basename.isdigit():
                    self.collection_items.append({
                        "artist": "",
                        "title": basename.replace("_", " ").strip(),
                        "year": "",
                        "cover_path": fpath
                    })

        logger.info(f"[Screensaver] Loaded {len(self.collection_items)} enriched albums for slideshow.")

    def show(self):
        """Activates full-screen overlay and begins slideshow."""
        if self.is_active:
            return
        
        self.is_active = True
        self._load_local_collection()
        self._current_pil_slide = None
        
        self.overlay_frame = tk.Frame(self.root, bg="#0a0a0c")
        self.overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay_frame.lift()
        
        self.canvas = tk.Canvas(self.overlay_frame, bg="#0a0a0c", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.overlay_frame.bind("<Button-1>", self._handle_wake)
        self.canvas.bind("<Button-1>", self._handle_wake)
        self.root.bind("<Key>", self._handle_wake)

        self._trigger_next_slide()
        logger.info("🌙 [Screensaver] Standby activated with smooth cross-fade.")

    def _handle_wake(self, event=None):
        """Dismisses screensaver on manual input."""
        if not self.is_active:
            return
        self.hide()
        if self.on_dismiss:
            self.on_dismiss()

    def hide(self):
        """Removes overlay and cleans up resources."""
        if not self.is_active:
            return
        
        self.is_active = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self._fade_after_id:
            self.root.after_cancel(self._fade_after_id)
            self._fade_after_id = None

        if self.overlay_frame:
            self.overlay_frame.destroy()
            self.overlay_frame = None

        self._active_img_ref = None
        self._current_pil_slide = None
        self._target_pil_slide = None
        logger.info("☀️ [Screensaver] User woke the device. Resuming operations...")

    def _generate_slide_image(self, item: Optional[Dict[str, Any]], w: int, h: int) -> Image.Image:
        """Composes a complete RGB frame: ambient blur background, high-res cover and crisp metadata."""
        if not item or not os.path.exists(item.get("cover_path", "")):
            # Fallback frame scuro minimale
            return Image.new("RGB", (w, h), (10, 10, 12))

        try:
            raw_img = Image.open(item["cover_path"]).convert("RGB")

            # 1. Sfondo ambient sfocato e scurito
            bg_img = raw_img.resize((w, h), Image.Resampling.BILINEAR)
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=30))
            bg_img = ImageEnhance.Brightness(bg_img).enhance(0.28)

            # 2. Copertina frontale calibrata (420px)
            cover_size = min(int(h * 0.74), int(w * 0.50), 420)
            fg_img = raw_img.resize((cover_size, cover_size), Image.Resampling.LANCZOS)
            
            top_margin = 36
            cx = w // 2
            cy = top_margin + (cover_size // 2)
            
            # Incolla la copertina sullo sfondo
            paste_x = cx - (cover_size // 2)
            paste_y = top_margin
            bg_img.paste(fg_img, (paste_x, paste_y))

            # 3. Disegna testo direttamente sull'immagine per dissolvere tutto insieme
            draw = ImageDraw.Draw(bg_img)
            
            title = item.get("title", "")
            artist = item.get("artist", "")
            year = item.get("year", "")
            
            # Caricamento font di sistema con fallback robusto
            try:
                font_title = Fonts.get_pil_font(Fonts.SCREENSAVER_TITLE)
                font_sub = Fonts.get_pil_font(Fonts.SCREENSAVER_ARTIST)
                font_hint = Fonts.get_pil_font(Fonts.SCREENSAVER_HINT)
            except Exception:
                font_title = ImageFont.load_default()
                font_sub = ImageFont.load_default()
                font_hint = ImageFont.load_default()

            base_y = cy + (cover_size // 2) + 36

            if title:
                draw.text((cx, base_y), title, fill=(255, 255, 255), font=font_title, anchor="mm")
                base_y += 36

            sub_text = artist.upper() if artist else ""
            if year:
                sub_text = f"{sub_text} • {year}" if sub_text else year

            if sub_text:
                draw.text((cx, base_y), sub_text, fill=(0, 168, 255), font=font_sub, anchor="mm")

            draw.text((cx, h - 12), "Touch to reactivate", fill=(85, 85, 102), font=font_hint, anchor="mm")

            return bg_img
        except Exception as e:
            logger.debug(f"[Screensaver] Generation error: {e}")
            return Image.new("RGB", (w, h), (10, 10, 12))

    def _trigger_next_slide(self):
        """Picks a new record and initiates the cross-fade animation."""
        if not self.is_active or not self.canvas:
            return

        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w < 100 or h < 100:
            w, h = 1024, 576

        item = random.choice(self.collection_items) if self.collection_items else None
        next_slide = self._generate_slide_image(item, w, h)

        if self._current_pil_slide is None:
            # Primo avvio immediato
            self._current_pil_slide = next_slide
            self._render_pil_to_canvas(next_slide)
            self._after_id = self.root.after(self.interval_ms, self._trigger_next_slide)
        else:
            # Avvia la sequenza di cross-fade
            self._target_pil_slide = next_slide
            self._fade_step = 0
            self._animate_cross_fade()

    def _animate_cross_fade(self):
        """Performs incremental alpha blend between current and incoming slide."""
        if not self.is_active or not self.canvas or self._target_pil_slide is None or self._current_pil_slide is None:
            return

        self._fade_step += 1
        alpha = self._fade_step / float(self._total_fade_steps)

        # Esegue la dissolvenza incrociata matematica
        blended = Image.blend(self._current_pil_slide, self._target_pil_slide, alpha)
        self._render_pil_to_canvas(blended)

        if self._fade_step < self._total_fade_steps:
            # 35ms per frame = ~400ms di transizione morbida e fluida
            self._fade_after_id = self.root.after(35, self._animate_cross_fade)
        else:
            # Transizione completata: assegna la nuova slide come corrente
            self._current_pil_slide = self._target_pil_slide
            self._target_pil_slide = None
            self._after_id = self.root.after(self.interval_ms, self._trigger_next_slide)

    def _render_pil_to_canvas(self, pil_img: Image.Image):
        """Updates the Canvas with the current composited PhotoImage."""
        self._active_img_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._active_img_ref, anchor=tk.NW)