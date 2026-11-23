import cv2
import numpy as np
from sklearn.cluster import KMeans


COLOR_HINTS_LAB = {
    "Alb": np.array([235.0, 128.0, 128.0]),
    "Negru": np.array([35.0, 128.0, 128.0]),
    "Roșu": np.array([135.0, 195.0, 170.0]),
    "Albastru": np.array([110.0, 145.0, 75.0]),
    "Galben": np.array([215.0, 115.0, 195.0]),
    "Verde": np.array([150.0, 80.0, 165.0]),
    "Portocaliu": np.array([175.0, 165.0, 185.0]),
    "Mov": np.array([105.0, 170.0, 95.0]),
}


class TeamClassifier:
    """
    Clasifică jucătorii în 2 echipe pe baza culorii dominante a tricoului (K-Means în spațiul CIE-LAB),
    ancorând automat clusterele pe culorile citite din tabela de scor sau selectate de utilizator.
    """
    def __init__(self, outlier_threshold: float = 72.0):
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.calibration_samples = []
        self.is_calibrated = False
        self.team_colors_bgr = {0: (255, 140, 0), 1: (0, 215, 255)}
        self.outlier_threshold = outlier_threshold

    def extract_jersey_lab(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 8 or box_h < 14:
            return None

        ty1 = y1 + int(box_h * 0.18)
        ty2 = y1 + int(box_h * 0.55)
        tx1 = x1 + int(box_w * 0.25)
        tx2 = x2 - int(box_w * 0.25)

        torso_crop = frame[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return None

        lab_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2LAB)
        pixels = lab_crop.reshape(-1, 3).astype(np.float32)
        return np.median(pixels, axis=0)

    def collect_sample(self, frame: np.ndarray, bbox: np.ndarray):
        feat = self.extract_jersey_lab(frame, bbox)
        if feat is not None:
            self.calibration_samples.append(feat)

    def fit(
        self,
        color_hint_0: str = "Auto-detect",
        color_hint_1: str = "Auto-detect",
        scoreboard_lab_0: np.ndarray | None = None,
        scoreboard_lab_1: np.ndarray | None = None
    ) -> bool:
        if len(self.calibration_samples) < 6:
            return False

        samples = np.array(self.calibration_samples)
        self.kmeans.fit(samples)

        c0 = self.kmeans.cluster_centers_[0]
        c1 = self.kmeans.cluster_centers_[1]

        target0 = scoreboard_lab_0 if scoreboard_lab_0 is not None else COLOR_HINTS_LAB.get(color_hint_0)
        target1 = scoreboard_lab_1 if scoreboard_lab_1 is not None else COLOR_HINTS_LAB.get(color_hint_1)

        swap_needed = False
        if target0 is not None and target1 is not None:
            cost_direct = np.linalg.norm(c0 - target0) + np.linalg.norm(c1 - target1)
            cost_swapped = np.linalg.norm(c0 - target1) + np.linalg.norm(c1 - target0)
            if cost_swapped < cost_direct:
                swap_needed = True
        elif target0 is not None:
            if np.linalg.norm(c1 - target0) < np.linalg.norm(c0 - target0):
                swap_needed = True
        elif target1 is not None:
            if np.linalg.norm(c0 - target1) < np.linalg.norm(c1 - target1):
                swap_needed = True

        if swap_needed:
            self.kmeans.cluster_centers_ = np.array([c1, c0])
            self.kmeans.labels_ = 1 - self.kmeans.labels_

        for idx in (0, 1):
            center_lab = np.uint8([[self.kmeans.cluster_centers_[idx]]])
            bgr = cv2.cvtColor(center_lab, cv2.COLOR_LAB2BGR)[0][0]
            self.team_colors_bgr[idx] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        self.is_calibrated = True
        return True

    def predict(self, frame: np.ndarray, bbox: np.ndarray) -> tuple[int, tuple[int, int, int]]:
        if not self.is_calibrated:
            return 0, self.team_colors_bgr[0]

        feat = self.extract_jersey_lab(frame, bbox)
        if feat is None:
            return -1, (160, 160, 160)

        centers = self.kmeans.cluster_centers_
        d0 = float(np.linalg.norm(feat - centers[0]))
        d1 = float(np.linalg.norm(feat - centers[1]))
        min_dist = min(d0, d1)

        if min_dist > self.outlier_threshold:
            return -1, (160, 160, 160)

        team_id = 0 if d0 <= d1 else 1
        return team_id, self.team_colors_bgr[team_id]