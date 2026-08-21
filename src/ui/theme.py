"""
VinylVision Hi-Fi Design Tokens & Theme Engine.
Centralized color palettes, typography scales, and styling constants.
Modern OLED Hi-Fi Dark Aesthetic.
"""
from PIL import ImageFont


# ==========================================
# 1. COLOR PALETTE (OLED Hi-Fi Studio Dark)
# ==========================================
class Colors:
    # Surfaces & Backdrops
    BG_ROOT         = "#0B0B0E"  # Nero profondo OLED
    BG_CARD         = "#141419"  # Superficie card primaria
    BG_CARD_ALT     = "#1A1A22"  # Riquadri interni (Pressing / Format)
    BG_DARK_BOX     = "#101015"  # Box Discogs Marketplace
    BG_OVERLAY      = "#070709"  # Sfondo Screensaver / Standby
    BG_CANVAS_EMPTY = "#0E0E12"  # Placeholder copertina vuota

    # Hi-Fi Accents
    ACCENT_PRIMARY  = "#00C0FF"  # Cyan Hi-Fi (Artista, Indicatori primari, Focus)
    ACCENT_SUCCESS  = "#00E676"  # Verde Smeraldo (Logo, Active state)
    ACCENT_WARNING  = "#FFB300"  # Ambra Calda (Stelle Rating Discogs)
    ACCENT_DANGER   = "#FF4555"  # Rosso Corallo (Standby)
    ACCENT_DISCOGS  = "#FF8A00"  # Arancio Discogs Marketplace

    # Standard Typography & Neutrals
    TEXT_PRIMARY    = "#F4F4F8"  # Titoli album, valori in evidenza (Bianco caldo)
    TEXT_SECONDARY  = "#D0D0DC"  # Testi secondari e valori tabelle
    TEXT_MUTED      = "#828294"  # Label descrittive, Have/Want, Metadati, Prezzi
    TEXT_DIM        = "#4E4E5E"  # Note a fondo pagina, hint touch
    TEXT_CAPTION    = "#656578"  # Etichette sezioni in maiuscolo

    # Synchronized Lyrics Focus Hierarchy (Progressive Fade Scale)
    LYRICS_ACTIVE   = "#FFFFFF"  # Riga attiva cantata al momento (Focus 100%)
    LYRICS_NEAR     = "#9C9CB0"  # Righe a distanza ±1 (65% opacità percepita)
    LYRICS_FAR      = "#5E5E70"  # Righe a distanza ±2 (40% opacità percepita)
    LYRICS_DISTANT  = "#343440"  # Righe a distanza ±3 (20% opacità percepita)
    LYRICS_TRACK    = "#00C0FF"  # Titolo brano in riproduzione (Shazam Track)

    # Audio Spectrum & Borders
    SPECTRUM_BAR    = "#00A8E8"  # Barre visualizzatore audio FFT
    SPECTRUM_PEAK   = "#00E676"  # Picco massimo visualizzatore FFT
    BORDER_SUBTLE   = "#22222C"  # Bordi card e divisori orizzontali
    BORDER_FOCUS    = "#323242"  # Bordi hover, canvas vuoti, slider track


# ==========================================
# 2. TYPOGRAPHY SCALES
# ==========================================
class Fonts:
    FAMILY_MAIN = "Segoe UI"
    FAMILY_ALT  = "Helvetica"
    FAMILY_MONO = "Consolas"

    # Dashboard & Kiosk
    LOGO               = (FAMILY_MAIN, 12, "bold")
    HERO_TITLE         = (FAMILY_MAIN, 20, "bold")
    ALBUM_TITLE        = (FAMILY_MAIN, 14, "bold")
    ALBUM_ARTIST       = (FAMILY_MAIN, 11, "bold")
    ALBUM_META         = (FAMILY_MAIN, 8)
    SECTION_HEADER     = (FAMILY_MAIN, 8, "bold")
    DETAIL_LABEL       = (FAMILY_MAIN, 8)
    DETAIL_VALUE       = (FAMILY_MAIN, 8, "bold")
    PRICE_TAG          = (FAMILY_MAIN, 9, "bold")

    # Settings & Calibration View
    SETTINGS_TITLE     = (FAMILY_MAIN, 13, "bold")
    SETTINGS_LABEL     = (FAMILY_MAIN, 10, "bold")
    SETTINGS_DETAIL    = (FAMILY_MAIN, 9)

    # Synchronized Lyrics Engine (7-Line Centered Hierarchy)
    LYRICS_TRACK_TITLE = (FAMILY_MAIN, 14, "bold")  # Ingrandito (Punto 4)
    LYRICS_LINE_ACTIVE = (FAMILY_MAIN, 14, "bold")  # Riga cantata ora
    LYRICS_LINE_NEAR   = (FAMILY_MAIN, 12)          # ±1 verso
    LYRICS_LINE_FAR    = (FAMILY_MAIN, 10)          # ±2 versi
    LYRICS_LINE_DIST   = (FAMILY_MAIN, 9)           # ±3 versi

    # Screensaver & Standby Slideshow
    SCREENSAVER_TITLE  = (FAMILY_MAIN, 28, "bold")
    SCREENSAVER_ARTIST = (FAMILY_MAIN, 15, "bold")
    SCREENSAVER_HINT   = (FAMILY_MAIN, 13, "normal")

    @classmethod
    def get_pil_font(cls, font_tuple: tuple) -> ImageFont.ImageFont:
        """
        Converts a VinylVision font tuple into a cross-platform PIL ImageFont,
        resolving system fonts on Windows, macOS, and Linux/Raspberry Pi.
        """
        size = font_tuple[1]
        weight = font_tuple[2] if len(font_tuple) > 2 else "normal"
        is_bold = "bold" in str(weight).lower()

        if is_bold:
            candidates = [
                "segoeuib.ttf",
                "arialbd.ttf",
                "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
            ]
        else:
            candidates = [
                "segoeui.ttf",
                "arial.ttf",
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
            ]

        for font_name in candidates:
            try:
                return ImageFont.truetype(font_name, size)
            except (OSError, IOError):
                continue

        return ImageFont.load_default()