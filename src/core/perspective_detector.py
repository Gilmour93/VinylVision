"""
Robust square/vinyl stand perspective detector with background rejection & ROI weighting.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-Left
    rect[2] = pts[np.argmax(s)]       # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-Right
    rect[3] = pts[np.argmax(diff)]    # Bottom-Left

    return rect


class PerspectiveDetector:
    """Rejects background furniture and isolates the foreground vinyl album."""

    def __init__(self, min_area_ratio: float = 0.08, max_area_ratio: float = 0.75):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect_corners(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        total_area = float(w * h)
        frame_center = np.array([w / 2.0, h / 2.0])

        # 1. Maschera di ritaglio periferica (Elimina pareti/armadi ai bordi estremi)
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255

        # 2. Pre-processing: Riduciamo rumore ad alta frequenza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Filtro Bilaterale forte per eliminare le texture dei mobili e la grafica del disco
        smoothed = cv2.bilateralFilter(gray, d=11, sigmaColor=90, sigmaSpace=90)

        # Equalizzazione adattiva locale per staccare il vinile dallo sfondo
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)

        candidates = []

        # 3. Multi-thresholding mirato
        canny_configs = [(40, 120), (25, 80), (60, 160)]
        for low_th, high_th in canny_configs:
            edges = cv2.Canny(enhanced, low_th, high_th)
            edges = cv2.bitwise_and(edges, edges, mask=mask)

            # Chiusura morfologica con kernel rettangolare per saldare i 4 lati del cartone
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            for cnt in contours:
                area = cv2.contourArea(cnt)
                area_ratio = area / total_area

                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri <= 0:
                    continue

                # Test approssimazione poligonale
                for eps in [0.03, 0.02, 0.04]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2).astype(np.float32)
                        score = self._evaluate_geometry(pts, frame_center, total_area, w, h)
                        if score > 0:
                            candidates.append((score, pts))
                        break

                # MinAreaRect orientato (molto robusto contro linee di sfondo parzialmente intersecate)
                rect = cv2.minAreaRect(cnt)
                (rw, rh) = rect[1]
                if rw > 0 and rh > 0:
                    ar = min(rw, rh) / max(rw, rh)
                    if 0.70 <= ar <= 1.30:  # Quasi perfettamente quadrato
                        box = cv2.boxPoints(rect).astype(np.float32)
                        score = self._evaluate_geometry(box, frame_center, total_area, w, h) * 0.90
                        if score > 0:
                            candidates.append((score, box))

        if candidates:
            # Ordina per punteggio (forma quadrata + centralità + area congrua)
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_pts = candidates[0][1]
            ordered = order_points(best_pts)
            return [(int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))) for pt in ordered]

        logger.warning("[PerspectiveDetector] No vinyl-like quadrilateral isolated.")
        return None

    def _evaluate_geometry(self, pts: np.ndarray, frame_center: np.ndarray, total_area: float, w: int, h: int) -> float:
        ordered = order_points(pts)

        # 1. Rapporto Lati (Aspect Ratio)
        top = np.linalg.norm(ordered[1] - ordered[0])
        right = np.linalg.norm(ordered[2] - ordered[1])
        bottom = np.linalg.norm(ordered[3] - ordered[2])
        left = np.linalg.norm(ordered[0] - ordered[3])

        avg_w = (top + bottom) / 2.0
        avg_h = (right + left) / 2.0

        if avg_w == 0 or avg_h == 0:
            return 0.0

        aspect_ratio = min(avg_w, avg_h) / max(avg_w, avg_h)
        if aspect_ratio < 0.70:  # Rifiuta rettangoli allungati tipici di ante dell'armadio
            return 0.0

        # 2. Penalizzazione Posizione: i mobili sono spesso in alto o ai lati estremi
        poly_center = pts.mean(axis=0)
        dist_from_center = np.linalg.norm(poly_center - frame_center)
        max_possible_dist = np.linalg.norm(frame_center)
        centrality = max(0.0, 1.0 - (dist_from_center / (max_possible_dist * 0.75)))

        # 3. Penalizzazione se tocca il bordo inferiore o superiore estremo
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        if min_y < h * 0.05 or max_y > h * 0.95:
            return 0.0

        # 4. Area Score
        area = cv2.contourArea(pts.astype(np.int32))
        area_ratio = area / total_area
        area_score = 1.0 if (0.15 <= area_ratio <= 0.65) else 0.4

        # Il vinile deve essere quadrato e centrato
        final_score = (aspect_ratio * 0.50) + (centrality * 0.35) + (area_score * 0.15)
        return float(final_score)