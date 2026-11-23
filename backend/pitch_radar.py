import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


SUPPORTED_SPORTS = [
    "Fotbal (Football)",
    "Baschet (Basketball)",
    "Handbal (Handball)",
    "Volei (Volleyball)",
    "Tenis (Tennis)",
    "Rugby (Rugby)",
    "Hochei pe gheață (Ice Hockey)",
    "Tenis de masă / Badminton"
]


class PitchHomographyMapper:
    """
    Transformă coordonatele pixel din perspectiva camerei TV în coordonate normalizate 2D [0..1, 0..1]
    pe terenul de joc folosind o matrice de omografie (cv2.getPerspectiveTransform).
    - x_norm in [0, 1]: lungimea terenului (stânga -> dreapta)
    - y_norm in [0, 1]: lățimea terenului (flanc sus/stânga -> centru -> flanc jos/dreapta)
    """
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        top_left_pct: tuple[float, float] = (18.0, 22.0),
        top_right_pct: tuple[float, float] = (82.0, 22.0),
        bottom_right_pct: tuple[float, float] = (98.0, 94.0),
        bottom_left_pct: tuple[float, float] = (2.0, 94.0)
    ):
        src_pts = np.float32([
            [frame_w * (top_left_pct[0] / 100.0), frame_h * (top_left_pct[1] / 100.0)],
            [frame_w * (top_right_pct[0] / 100.0), frame_h * (top_right_pct[1] / 100.0)],
            [frame_w * (bottom_right_pct[0] / 100.0), frame_h * (bottom_right_pct[1] / 100.0)],
            [frame_w * (bottom_left_pct[0] / 100.0), frame_h * (bottom_left_pct[1] / 100.0)],
        ])
        dst_pts = np.float32([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def map_point(self, px: float, py: float) -> tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)[0][0]
        nx = float(np.clip(mapped[0], 0.0, 1.0))
        ny = float(np.clip(mapped[1], 0.0, 1.0))
        return nx, ny


def get_sport_palette(sport: str) -> dict:
    s = sport.lower()
    if "baschet" in s or "basketball" in s:
        return {"bg_bgr": (75, 130, 195), "stripe_bgr": (85, 142, 208), "hex": "#c3824b", "type": "basketball"}
    if "handbal" in s or "handball" in s:
        return {"bg_bgr": (180, 115, 45), "stripe_bgr": (192, 126, 54), "hex": "#2d73b4", "type": "handball"}
    if "volei" in s or "volleyball" in s:
        return {"bg_bgr": (65, 135, 215), "stripe_bgr": (75, 145, 225), "hex": "#d78741", "type": "volleyball"}
    if "tenis (" in s or "tennis" in s:
        return {"bg_bgr": (145, 95, 55), "stripe_bgr": (155, 105, 65), "hex": "#375f91", "type": "tennis"}
    if "hochei" in s or "hockey" in s:
        return {"bg_bgr": (235, 225, 215), "stripe_bgr": (242, 232, 222), "hex": "#d7e1eb", "type": "hockey"}
    if "rugby" in s:
        return {"bg_bgr": (42, 118, 46), "stripe_bgr": (50, 132, 54), "hex": "#2e762a", "type": "rugby"}
    if "masă" in s or "badminton" in s:
        return {"bg_bgr": (110, 85, 45), "stripe_bgr": (120, 95, 55), "hex": "#2d556e", "type": "net_court"}
    return {"bg_bgr": (46, 125, 50), "stripe_bgr": (56, 142, 60), "hex": "#2e7d32", "type": "football"}


def draw_2d_pitch_canvas(width: int = 800, height: int = 300, sport: str = "Fotbal (Football)") -> np.ndarray:
    """
    Desenează terenul 2D orizontal adaptat sportului selectat.
    """
    pal = get_sport_palette(sport)
    ctype = pal["type"]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = pal["bg_bgr"]

    stripe_w = max(1, width // 12)
    for i in range(0, 12, 2):
        canvas[:, i * stripe_w:(i + 1) * stripe_w] = pal["stripe_bgr"]

    line_col = (60, 60, 60) if ctype == "hockey" else (240, 240, 240)
    m_x, m_y = 26, 28
    mid_x, mid_y = width // 2, height // 2

    cv2.rectangle(canvas, (m_x, m_y), (width - m_x, height - m_y), line_col, 2)
    cv2.line(canvas, (mid_x, m_y), (mid_x, height - m_y), line_col, 2)

    if ctype in ("football", "rugby"):
        rad = max(22, int((height - 2 * m_y) * 0.18))
        cv2.circle(canvas, (mid_x, mid_y), rad, line_col, 2)
        box_w = int((width - 2 * m_x) * 0.16)
        box_h = int((height - 2 * m_y) * 0.58)
        by1 = (height - box_h) // 2
        cv2.rectangle(canvas, (m_x, by1), (m_x + box_w, by1 + box_h), line_col, 2)
        cv2.rectangle(canvas, (width - m_x - box_w, by1), (width - m_x, by1 + box_h), line_col, 2)
        if ctype == "rugby":
            l22 = int((width - 2 * m_x) * 0.22)
            cv2.line(canvas, (m_x + l22, m_y), (m_x + l22, height - m_y), line_col, 1)
            cv2.line(canvas, (width - m_x - l22, m_y), (width - m_x - l22, height - m_y), line_col, 1)

    elif ctype == "basketball":
        rad = max(20, int((height - 2 * m_y) * 0.16))
        cv2.circle(canvas, (mid_x, mid_y), rad, line_col, 2)
        key_w = int((width - 2 * m_x) * 0.19)
        key_h = int((height - 2 * m_y) * 0.36)
        ky1 = (height - key_h) // 2
        cv2.rectangle(canvas, (m_x, ky1), (m_x + key_w, ky1 + key_h), line_col, 2)
        cv2.rectangle(canvas, (width - m_x - key_w, ky1), (width - m_x, ky1 + key_h), line_col, 2)
        arc_r = int((height - 2 * m_y) * 0.42)
        cv2.ellipse(canvas, (m_x + 15, mid_y), (arc_r, arc_r), 0, -85, 85, line_col, 2)
        cv2.ellipse(canvas, (width - m_x - 15, mid_y), (arc_r, arc_r), 0, 95, 265, line_col, 2)

    elif ctype == "handball":
        rad_6 = int((height - 2 * m_y) * 0.32)
        rad_9 = int((height - 2 * m_y) * 0.46)
        cv2.ellipse(canvas, (m_x, mid_y), (rad_6, rad_6), 0, -90, 90, line_col, 2)
        cv2.ellipse(canvas, (width - m_x, mid_y), (rad_6, rad_6), 0, 90, 270, line_col, 2)
        cv2.ellipse(canvas, (m_x, mid_y), (rad_9, rad_9), 0, -90, 90, line_col, 1)
        cv2.ellipse(canvas, (width - m_x, mid_y), (rad_9, rad_9), 0, 90, 270, line_col, 1)

    elif ctype in ("volleyball", "tennis", "net_court"):
        # Fileul central mai gros
        cv2.line(canvas, (mid_x, m_y - 4), (mid_x, height - m_y + 4), (0, 230, 255), 3)
        off = int((width - 2 * m_x) * 0.17)
        cv2.line(canvas, (mid_x - off, m_y), (mid_x - off, height - m_y), line_col, 1)
        cv2.line(canvas, (mid_x + off, m_y), (mid_x + off, height - m_y), line_col, 1)
        if ctype == "tennis":
            alley = int((height - 2 * m_y) * 0.13)
            cv2.line(canvas, (m_x, m_y + alley), (width - m_x, m_y + alley), line_col, 1)
            cv2.line(canvas, (m_x, height - m_y - alley), (width - m_x, height - m_y - alley), line_col, 1)

    elif ctype == "hockey":
        blue_line = int((width - 2 * m_x) * 0.34)
        cv2.line(canvas, (m_x + blue_line, m_y), (m_x + blue_line, height - m_y), (220, 80, 40), 2)
        cv2.line(canvas, (width - m_x - blue_line, m_y), (width - m_x - blue_line, height - m_y), (220, 80, 40), 2)
        cv2.circle(canvas, (mid_x, mid_y), 30, (40, 40, 220), 2)

    return canvas


def render_live_radar_frame(
    base_pitch: np.ndarray,
    players: list[dict],
    ball_norm_pos: tuple[float, float] | None,
    team_colors_bgr: dict,
    team_names: tuple[str, str],
    attacking_team: str | None,
    is_tactical: bool = True,
    pause_reason: str = ""
) -> np.ndarray:
    """
    Proiectează coordonatele normalizate homografic [0..1, 0..1] pe radarul 2D orizontal.
    Dacă is_tactical=False (close-up / reluare), afișează starea înghețată fără a polua pozițiile.
    """
    radar = base_pitch.copy()
    rh, rw, _ = radar.shape
    m_x, m_y = 26, 28

    for p in players:
        tid = p.get("team_id", -1)
        nx, ny = p.get("pitch_norm", (0.5, 0.5))

        rx = int(m_x + nx * (rw - 2 * m_x))
        ry = int(m_y + ny * (rh - 2 * m_y))

        if tid in (0, 1):
            color = team_colors_bgr.get(tid, (255, 255, 255))
            cv2.circle(radar, (rx, ry), 9, color, -1)
            cv2.circle(radar, (rx, ry), 9, (255, 255, 255), 1)
            if p.get("track_id") is not None:
                cv2.putText(radar, str(p["track_id"]), (rx - 7, ry - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        else:
            cv2.circle(radar, (rx, ry), 6, (180, 180, 180), -1)

    if ball_norm_pos is not None:
        bx = int(m_x + ball_norm_pos[0] * (rw - 2 * m_x))
        by = int(m_y + ball_norm_pos[1] * (rh - 2 * m_y))
        cv2.circle(radar, (bx, by), 7, (0, 230, 255), -1)
        cv2.circle(radar, (bx, by), 9, (0, 0, 0), 2)

    cv2.rectangle(radar, (0, 0), (rw, 22), (20, 20, 20), -1)
    c0 = team_colors_bgr.get(0, (255, 255, 255))
    c1 = team_colors_bgr.get(1, (255, 255, 255))
    cv2.circle(radar, (18, 11), 6, c0, -1)
    cv2.putText(radar, team_names[0][:18], (30, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.circle(radar, (225, 11), 6, c1, -1)
    cv2.putText(radar, team_names[1][:18], (237, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(radar, "Flanc Stg", (rw - 72, m_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1)
    cv2.putText(radar, "Flanc Drp", (rw - 72, rh - m_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1)

    if not is_tactical:
        overlay = radar.copy()
        cv2.rectangle(overlay, (m_x, m_y), (rw - m_x, rh - m_y), (15, 15, 15), -1)
        radar = cv2.addWeighted(overlay, 0.45, radar, 0.55, 0)
        msg = f"PAUZA RADAR ({pause_reason}) — Pozitii inghetate"
        cv2.putText(radar, msg, (rw // 2 - 220, rh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)
    elif attacking_team:
        cv2.putText(radar, f"FAZA ATAC: {attacking_team[:18]}", (rw - 295, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)

    return radar


def generate_2d_tactical_pitch_map(
    norm_x_list: list[float],
    norm_y_list: list[float],
    team_name: str,
    sport: str,
    save_path: str
):
    """
    Generează harta tactică 2D a terenului folosind coordonatele proiectate homografic.
    """
    pal = get_sport_palette(sport)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_facecolor(pal["hex"])

    pitch_w, pitch_h = 100.0, 68.0
    line_c = "#333333" if pal["type"] == "hockey" else "white"

    ax.add_patch(patches.Rectangle((0, 0), pitch_w, pitch_h, fill=False, edgecolor=line_c, linewidth=2))
    ax.plot([50, 50], [0, pitch_h], color=line_c, linewidth=1.8)
    ax.add_patch(patches.Circle((50, 34), 9.15, fill=False, edgecolor=line_c, linewidth=1.8))

    # Zone flancuri orizontale (Flanc Stânga sus, Centru mijloc, Flanc Dreapta jos)
    ax.axhline(22.6, color=line_c, linestyle="--", alpha=0.35, linewidth=1)
    ax.axhline(45.3, color=line_c, linestyle="--", alpha=0.35, linewidth=1)
    ax.text(2, 56, "Flanc Stânga", color=line_c, fontsize=8, alpha=0.8)
    ax.text(2, 33, "Zonă Centrală", color=line_c, fontsize=8, alpha=0.8)
    ax.text(2, 10, "Flanc Dreapta", color=line_c, fontsize=8, alpha=0.8)

    if norm_x_list and norm_y_list:
        nx = np.clip(np.array(norm_x_list), 0.0, 1.0) * pitch_w
        # ny=0 (sus pe teren) -> y=68 in matplotlib; ny=1 (jos pe teren) -> y=0
        ny = (1.0 - np.clip(np.array(norm_y_list), 0.0, 1.0)) * pitch_h

        hb = ax.hexbin(nx, ny, gridsize=22, cmap="YlOrRd", alpha=0.68, mincnt=1, extent=[0, pitch_w, 0, pitch_h])
        plt.colorbar(hb, ax=ax, label="Densitate Pozițională")

        mean_x, mean_y = float(np.mean(nx)), float(np.mean(ny))
        ax.scatter([mean_x], [mean_y], color="cyan", edgecolor="black", s=160, zorder=5, label="Centru Greutate Echipă")
        ax.legend(loc="upper right")

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 70)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Hartă Tactică 2D ({sport}) — {team_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()