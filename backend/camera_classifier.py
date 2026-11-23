import cv2
import numpy as np


class TacticalCameraFilter:
    """
    Distince cadrele tactice largi (broadcast main camera) de:
    - close-up-uri pe un singur jucător / portar / antrenor,
    - ecrane de introducere / grafică TV,
    - reluări din spatele porții (unde perspectiva terenului este rotită cu 90°).
    """
    def __init__(self, sport: str = "Fotbal (Football)"):
        self.sport = sport.lower()

    def _compute_pitch_coverage(self, frame_bgr: np.ndarray) -> float:
        """
        Calculează procentul de pixeli din cadru care aparțin suprafeței de joc specifice sportului.
        """
        h, w, _ = frame_bgr.shape
        small = cv2.resize(frame_bgr, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        if "handbal" in self.sport or "handball" in self.sport or "tenis (" in self.sport:
            # Suprafață albastră / taraflex / hardcourt
            mask = (hue >= 85) & (hue <= 135) & (sat >= 45) & (val >= 45)
        elif "baschet" in self.sport or "basketball" in self.sport or "volei" in self.sport:
            # Parchet lemn deschis / portocaliu-galben
            mask = (hue >= 8) & (hue <= 32) & (sat >= 45) & (val >= 70)
        elif "hochei" in self.sport or "hockey" in self.sport:
            # Gheață albă-albăstruie luminoasă
            mask = (sat <= 55) & (val >= 165)
        else:
            # Gazon verde (Fotbal, Rugby)
            mask = (hue >= 32) & (hue <= 88) & (sat >= 38) & (val >= 35)

        return float(np.count_nonzero(mask)) / float(320 * 180)

    def _is_behind_goal_angle(self, frame_bgr: np.ndarray) -> bool:
        """
        Într-o cameră TV laterală normală, liniile lungi ale terenului sunt orizontale/diagonale ușoare (0°..32°).
        Într-o reluare din spatele porții, liniile tușelor și ale careului devin aproape verticale (55°..90°).
        """
        small = cv2.resize(frame_bgr, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=45, maxLineGap=10)
        if lines is None or len(lines) < 4:
            return False

        steep_count = 0
        total = 0
        if lines is not None:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])

        return total >= 4 and (steep_count / total) > 0.62

    def classify_frame(self, frame_bgr: np.ndarray, player_bboxes: np.ndarray) -> tuple[bool, str]:
        """
        Returnează (is_tactical, reason):
        - (True, "TACTICAL_WIDE") dacă cadrul este valid pentru Radar 2D și statistici;
        - (False, motiv) dacă este close-up, reluare sau grafică TV.
        """
        frame_h = frame_bgr.shape[0]
        pitch_ratio = self._compute_pitch_coverage(frame_bgr)

        # 1. Prea puțin teren vizibil (grafică TV, tribune, bancă rezerve, tunel)
        if pitch_ratio < 0.24:
            return False, "Grafică TV / Margine teren"

        # 2. Prea puțini jucători detectați pentru o imagine de ansamblu
        num_players = len(player_bboxes)
        if num_players < 3:
            return False, "Cadru izolat (<3 jucători)"

        # 3. Verificăm dimensiunea jucătorilor în cadru (Close-up / Zoom pe jucător sau portar)
        heights_pct = [(xyxy[3] - xyxy[1]) / float(frame_h) for xyxy in player_bboxes]
        median_h = float(np.median(heights_pct))
        max_h = float(np.max(heights_pct))

        if median_h > 0.24 or max_h > 0.42:
            return False, "Close-up Jucător / Celebrare"

        # 4. Verificăm unghiul din spatele porții (reluare)
        if self._is_behind_goal_angle(frame_bgr):
            return False, "Reluare unghi poartă"

        return True, "TACTICAL_WIDE"