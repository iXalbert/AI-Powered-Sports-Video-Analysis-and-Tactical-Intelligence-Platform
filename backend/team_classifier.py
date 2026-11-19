import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, outlier_threshold: float = 75.0):
        self.kmeans = None
        self.team_colors_bgr = {}
        self.outlier_threshold = outlier_threshold
        self.calibration_samples = []
        self.is_calibrated = False

    def _extract_jersey_color(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 10 or box_h < 20:
            return None

        # Decupăm strict zona pieptului/tricoului (20%-55% pe verticală, 25%-75% pe orizontală)
        ty1 = y1 + int(box_h * 0.20)
        ty2 = y1 + int(box_h * 0.55)
        tx1 = x1 + int(box_w * 0.25)
        tx2 = x1 + int(box_w * 0.75)

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        # Eliminăm pixelii verzi de gazon din crop folosind spațiul HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        non_green_pixels = crop[green_mask == 0]

        if len(non_green_pixels) < 15:
            non_green_pixels = crop.reshape(-1, 3)

        # Convertim în LAB pentru distanțe perceptuale de culoare mult mai precise
        pixels_lab = cv2.cvtColor(non_green_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return np.median(pixels_lab, axis=0)

    def collect_sample(self, frame: np.ndarray, xyxy: np.ndarray):
        color_lab = self._extract_jersey_color(frame, xyxy)
        if color_lab is not None:
            self.calibration_samples.append(color_lab)

    def fit(self):
        if len(self.calibration_samples) < 6:
            return False

        samples = np.array(self.calibration_samples)
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(samples)

        # Convertim centroizii LAB înapoi în BGR pentru desenarea chenarelor în culoarea reală
        for team_id in [0, 1]:
            centroid_lab = np.uint8([[self.kmeans.cluster_centers_[team_id]]])
            bgr = cv2.cvtColor(centroid_lab, cv2.COLOR_LAB2BGR)[0][0]
            self.team_colors_bgr[team_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        self.is_calibrated = True
        return True

    def predict(self, frame: np.ndarray, xyxy: np.ndarray) -> tuple[int, tuple[int, int, int]]:
        """
        Returnează (team_id, bgr_color):
        team_id = 0 (Echipa 1), 1 (Echipa 2), sau -1 (Arbitru / Portar / Outlier)
        """
        if not self.is_calibrated:
            return 0, (0, 255, 255)

        color_lab = self._extract_jersey_color(frame, xyxy)
        if color_lab is None:
            return -1, (180, 180, 180)

        distances = np.linalg.norm(self.kmeans.cluster_centers_ - color_lab, axis=1)
        min_dist = float(np.min(distances))
        best_team = int(np.argmin(distances))

        if min_dist > self.outlier_threshold:
            return -1, (160, 160, 160)

        return best_team, self.team_colors_bgr[best_team]