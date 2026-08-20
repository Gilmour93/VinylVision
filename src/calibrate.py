import cv2
import numpy as np
import json
import os
from utils.config import load_config
from core.camera import CameraManager

class PerspectiveCalibrator:
    def __init__(self):
        config = load_config()
        self.camera = CameraManager(config.camera.device_id, target_fps=30)
        self.camera.initialize()
        
        self.window_name = "Calibrazione Vinile (Trascina i 4 punti o premi 's' per salvare)"
        self.config_path = "user_data/calibration.json"
        
        # Punti predefiniti (rettangolo centrale iniziale)
        self.points = []
        self.selected_point = -1
        self.load_points()

    def load_points(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.points = [tuple(p) for p in data.get('corners', [])]
            except Exception:
                self.points = []

    def save_points(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({'corners': self.points}, f, indent=2)
        print(f"\n[OK] Coordinate salvate correttamente in {self.config_path}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, pt in enumerate(self.points):
                if np.hypot(pt[0] - x, pt[1] - y) < 25:
                    self.selected_point = idx
                    return
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point != -1:
            self.points[self.selected_point] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point = -1

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n--- ISTRUZIONI DI CALIBRAZIONE ---")
        print("1. Posiziona un vinile sul supporto al muro.")
        print("2. Trascina i 4 cerchi colorati sui 4 angoli del vinile.")
        print("3. Premi 's' sulla tastiera per salvare.")
        print("4. Premi 'q' o ESC per uscire.\n")

        while True:
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            if len(self.points) != 4:
                # Inizializza 4 punti se non presenti
                pad_w, pad_h = int(w * 0.2), int(h * 0.2)
                self.points = [
                    (pad_w, pad_h),             # Top-Left
                    (w - pad_w, pad_h),         # Top-Right
                    (w - pad_w, h - pad_h),     # Bottom-Right
                    (pad_w, h - pad_h)          # Bottom-Left
                ]

            display = frame.copy()
            pts_array = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

            colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
            labels = ["TL", "TR", "BR", "BL"]
            for idx, (pt, col, lbl) in enumerate(zip(self.points, colors, labels)):
                cv2.circle(display, pt, 10, col, -1)
                cv2.circle(display, pt, 12, (255, 255, 255), 2)
                cv2.putText(display, lbl, (pt[0] + 15, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'):
                self.save_points()
                break
            elif key in (ord('q'), 27):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = PerspectiveCalibrator()
    calibrator.run()