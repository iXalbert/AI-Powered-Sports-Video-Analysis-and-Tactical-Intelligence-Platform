import os
import re
import json
import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


TEAM_ABBR_DB = {
    # Fotbal (Premier League, La Liga, UCL, Liga 1 etc.)
    "ARS": "Arsenal", "LEE": "Leeds United", "MCI": "Manchester City", "LIV": "Liverpool",
    "CHE": "Chelsea", "MUN": "Manchester United", "TOT": "Tottenham", "NEW": "Newcastle",
    "AVL": "Aston Villa", "BHA": "Brighton", "WHU": "West Ham", "FUL": "Fulham",
    "BAR": "FC Barcelona", "FCB": "FC Barcelona", "RMA": "Real Madrid", "ATM": "Atlético Madrid",
    "SEV": "Sevilla", "VAL": "Valencia", "VIL": "Villarreal", "RSO": "Real Sociedad",
    "BAY": "Bayern München", "BVB": "Borussia Dortmund", "RBL": "RB Leipzig", "LEV": "Bayer Leverkusen",
    "PSG": "Paris Saint-Germain", "OM": "Marseille", "OL": "Lyon", "MON": "AS Monaco",
    "INT": "Inter Milano", "ACM": "AC Milan", "MIL": "AC Milan", "JUV": "Juventus",
    "NAP": "Napoli", "ROM": "AS Roma", "LAZ": "Lazio", "ATA": "Atalanta",
    "FCS": "FCSB", "CFR": "CFR Cluj", "RAP": "Rapid București", "DIN": "Dinamo București",
    "CRA": "Universitatea Craiova", "FAR": "Farul Constanța", "ROU": "România",
    # Handbal (EHF Champions League & Naționale)
    "THW": "THW Kiel", "MAG": "SC Magdeburg", "VES": "Veszprém", "SZE": "Pick Szeged",
    "KIE": "Industria Kielce", "AAL": "Aalborg", "NAN": "HBC Nantes", "MON": "Montpellier",
    "CSM": "CSM București", "GJO": "Győri Audi ETO", "MET": "Metz Handball",
    # Baschet (NBA & EuroLeague)
    "LAL": "LA Lakers", "BOS": "Boston Celtics", "GSW": "Golden State Warriors",
    "MIA": "Miami Heat", "DEN": "Denver Nuggets", "MIL": "Milwaukee Bucks",
    "DAL": "Dallas Mavericks", "PHX": "Phoenix Suns", "NYK": "New York Knicks",
    "OLY": "Olympiacos", "PAO": "Panathinaikos", "FEN": "Fenerbahçe", "EFS": "Anadolu Efes",
    # Hochei (NHL)
    "TOR": "Toronto Maple Leafs", "MTL": "Montreal Canadiens", "NYR": "New York Rangers",
    "EDM": "Edmonton Oilers", "COL": "Colorado Avalanche", "TBL": "Tampa Bay Lightning",
}

IGNORED_TOKENS = {"LIVE", "HALF", "GOAL", "TIME", "PEN", "VAR", "EXT", "ADD", "MIN", "MAX", "END", "SET"}


class ScoreboardReader:
    """
    Citește tabela de scor din colțul/partea de sus a ecranului (hibrid: Gemini Flash dacă există API key,
    altfel 100% gratuit și offline cu EasyOCR) pentru a extrage codurile de 3 litere (ex. ARS vs LEE, BAR vs PSG)
    și culorile echipamentelor asociate fiecărei echipe.
    """
    def __init__(self):
        self._reader = None

    def _get_easyocr(self):
        if self._reader is None and easyocr is not None:
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def _sample_adjacent_jersey_lab(self, crop_bgr: np.ndarray, bbox) -> np.ndarray | None:
        """
        Eșantionează culoarea benzii/accentului grafic din jurul textului de 3 litere (stânga, dreapta sau sub text),
        ignorând pixelii neutri de fundal (alb pur / negru pur / gri închis).
        """
        h, w, _ = crop_bgr.shape
        pts = np.array(bbox, dtype=np.float32).reshape(-1, 2)
        x1, x2 = max(0, int(np.min(pts[:, 0]))), min(w, int(np.max(pts[:, 0])))
        y1, y2 = max(0, int(np.min(pts[:, 1]))), min(h, int(np.max(pts[:, 1])))

        if x2 <= x1 or y2 <= y1:
            return None

        pad_x = max(14, int((x2 - x1) * 0.45))
        pad_y = max(6, int((y2 - y1) * 0.35))

        rx1 = max(0, x1 - pad_x)
        rx2 = min(w, x2 + pad_x)
        ry1 = max(0, y1 - pad_y)
        ry2 = min(h, y2 + pad_y)

        patch = crop_bgr[ry1:ry2, rx1:rx2]
        if patch.size == 0:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 1. Căutăm mai întâi pixeli colorați (saturație > 55) — ex. banda roșie ARS, galbenă LEE, albastră BAR
        chromatic_mask = (sat > 55) & (val > 40)
        if np.count_nonzero(chromatic_mask) >= 12:
            pixels = lab[chromatic_mask].astype(np.float32)
            return np.median(pixels, axis=0)

        # 2. Dacă echipamentul e alb sau negru (ex. PSG alb), luăm luminozitatea dominantă din zona de accent
        border_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
        border_mask[max(0, y1 - ry1):max(0, y2 - ry1), max(0, x1 - rx1):max(0, x2 - rx1)] = False
        if np.count_nonzero(border_mask) >= 10:
            pixels = lab[border_mask].astype(np.float32)
            return np.median(pixels, axis=0)

        return None

    def _detect_with_gemini(self, frame_bgr: np.ndarray) -> dict | None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or genai is None:
            return None

        try:
            h, w, _ = frame_bgr.shape
            top_strip = frame_bgr[0:int(h * 0.25), 0:w]
            _, buf = cv2.imencode(".jpg", top_strip)
            client = genai.Client(api_key=api_key)

            prompt = (
                "Inspectează tabela de scor din partea de sus a imaginii. "
                "Extrage codurile de 3 litere ale celor două echipe (ex. ARS, LEE, BAR, PSG), "
                "numele complet al fiecărei echipe și culoarea principală a echipamentului afișată pe tabela de scor "
                "(alege din: Alb, Negru, Roșu, Albastru, Galben, Verde, Portocaliu, Mov). "
                "Returnează strict JSON: "
                '{"detected": true, "team_0_abbr": "...", "team_0_name": "...", "team_0_color": "...", '
                '"team_1_abbr": "...", "team_1_name": "...", "team_1_color": "..."} '
                "Dacă nu există tabelă de scor vizibilă, returnează {\"detected\": false}."
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                    prompt
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            data = json.loads(resp.text)
            if data.get("detected") and data.get("team_0_abbr") and data.get("team_1_abbr"):
                return data
        except Exception:
            return None
        return None

    def detect_from_video(self, video_path: str, start_frame: int = 0, end_frame: int = 400) -> dict | None:
        """
        Scanează câteva cadre distribuite în prima parte a videoclipului (pentru a sări peste intro-urile
        de 5-12 secunde fără tabelă de scor) și extrage echipele + culorile echipamentelor.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        span = max(30, end_frame - start_frame)
        # Eșantionăm 8 cadre între 10% și 65% din interval (unde tabela TV este deja afișată)
        sample_offsets = [int(span * r) for r in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45, 0.55, 0.65)]
        frames_to_check = []

        for off in sample_offsets:
            f_idx = start_frame + off
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                frames_to_check.append(frame)
        cap.release()

        if not frames_to_check:
            return None

        # 1. Dacă utilizatorul a setat GEMINI_API_KEY, încercăm mai întâi Gemini Vision
        for fr in frames_to_check[1:4]:
            gem_res = self._detect_with_gemini(fr)
            if gem_res:
                return {
                    "source": "gemini_vision",
                    "team_0_abbr": gem_res["team_0_abbr"].upper(),
                    "team_0_name": gem_res.get("team_0_name") or TEAM_ABBR_DB.get(gem_res["team_0_abbr"].upper(), gem_res["team_0_abbr"].upper()),
                    "team_0_color_hint": gem_res.get("team_0_color", "Auto-detect"),
                    "team_0_lab": None,
                    "team_1_abbr": gem_res["team_1_abbr"].upper(),
                    "team_1_name": gem_res.get("team_1_name") or TEAM_ABBR_DB.get(gem_res["team_1_abbr"].upper(), gem_res["team_1_abbr"].upper()),
                    "team_1_color_hint": gem_res.get("team_1_color", "Auto-detect"),
                    "team_1_lab": None,
                }

        # 2. Altfel folosim EasyOCR 100% gratuit și offline pe zona tabelei de scor (top 18% din ecran)
        reader = self._get_easyocr()
        if reader is None:
            return None

        pair_votes = {}
        pair_labs = {}

        for fr in frames_to_check:
            h, w, _ = fr.shape
            top_crop = fr[0:int(h * 0.18), 0:w]
            ocr_hits = reader.readtext(top_crop)

            candidates = []
            for bbox, text, conf in ocr_hits:
                clean = re.sub(r"[^A-Za-z]", "", text).upper()
                if len(clean) == 3 and clean not in IGNORED_TOKENS and conf >= 0.35:
                    candidates = []
                    for item in ocr_hits:
                        if len(item) < 3:
                            continue
                        bbox, text, conf = item[0], item[1], float(item[2])
                        clean = re.sub(r"[^A-Za-z]", "", str(text)).upper()
                        if len(clean) == 3 and clean not in IGNORED_TOKENS and conf >= 0.35:
                            pts = np.array(bbox, dtype=np.float32).reshape(-1, 2)
                            x_center = float(np.mean(pts[:, 0]))
                            lab_col = self._sample_adjacent_jersey_lab(top_crop, bbox)
                            candidates.append((x_center, clean, lab_col))

            # Sortăm de la stânga la dreapta (ex: ARS apoi LEE, sau BAR apoi PSG)
            candidates.sort(key=lambda item: item[0])
            if len(candidates) >= 2:
                # Luăm primele două coduri distincte apropiate (tabela de scor)
                c0 = candidates[0]
                c1 = next((c for c in candidates[1:] if c[1] != c0[1] and (c[0] - c0[0]) < w * 0.45), None)
                if c1 is not None:
                    key = (c0[1], c1[1])
                    pair_votes[key] = pair_votes.get(key, 0) + 1
                    pair_labs.setdefault(key, {0: [], 1: []})
                    if c0[2] is not None:
                        pair_labs[key][0].append(c0[2])
                    if c1[2] is not None:
                        pair_labs[key][1].append(c1[2])

        if not pair_votes:
            return None

        best_pair = max(pair_votes, key=pair_votes.get)
        abbr0, abbr1 = best_pair
        lab0 = np.median(pair_labs[best_pair][0], axis=0) if pair_labs[best_pair][0] else None
        lab1 = np.median(pair_labs[best_pair][1], axis=0) if pair_labs[best_pair][1] else None

        return {
            "source": "easyocr_local",
            "team_0_abbr": abbr0,
            "team_0_name": f"{TEAM_ABBR_DB.get(abbr0, abbr0)} ({abbr0})",
            "team_0_color_hint": "Auto-detect",
            "team_0_lab": lab0,
            "team_1_abbr": abbr1,
            "team_1_name": f"{TEAM_ABBR_DB.get(abbr1, abbr1)} ({abbr1})",
            "team_1_color_hint": "Auto-detect",
            "team_1_lab": lab1,
        }