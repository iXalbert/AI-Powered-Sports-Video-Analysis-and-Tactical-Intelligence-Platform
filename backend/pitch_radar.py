import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_2d_pitch_canvas(width: int = 800, height: int = 320) -> np.ndarray:
    """
    Desenează un teren de fotbal 2D pe orizontală (OpenCV BGR) pentru radarul live de sub video.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (46, 125, 50)  # Verde gazon

    # Dungi verticale subtile de gazon
    stripe_w = width // 12
    for i in range(0, 12, 2):
        canvas[:, i * stripe_w:(i + 1) * stripe_w] = (56, 142, 60)

    white = (235, 235, 235)
    m_x = 24
    m_y = 28

    # Contur exterior teren
    cv2.rectangle(canvas, (m_x, m_y), (width - m_x, height - m_y), white, 2)

    # Linia de mijloc si cercul median
    mid_x = width // 2
    mid_y = height // 2
    cv2.line(canvas, (mid_x, m_y), (mid_x, height - m_y), white, 2)
    radius = max(25, int((height - 2 * m_y) * 0.18))
    cv2.circle(canvas, (mid_x, mid_y), radius, white, 2)
    cv2.circle(canvas, (mid_x, mid_y), 4, white, -1)

    # Careul mare si mic stanga
    box_w = int((width - 2 * m_x) * 0.16)
    box_h = int((height - 2 * m_y) * 0.58)
    by1 = (height - box_h) // 2
    cv2.rectangle(canvas, (m_x, by1), (m_x + box_w, by1 + box_h), white, 2)

    sbox_w = int(box_w * 0.40)
    sbox_h = int(box_h * 0.45)
    sby1 = (height - sbox_h) // 2
    cv2.rectangle(canvas, (m_x, sby1), (m_x + sbox_w, sby1 + sbox_h), white, 2)

    # Careul mare si mic dreapta
    cv2.rectangle(canvas, (width - m_x - box_w, by1), (width - m_x, by1 + box_h), white, 2)
    cv2.rectangle(canvas, (width - m_x - sbox_w, sby1), (width - m_x, sby1 + sbox_h), white, 2)

    return canvas


def render_live_radar_frame(
    base_pitch: np.ndarray,
    players: list[dict],
    ball_pos: tuple[float, float] | None,
    frame_w: int,
    frame_h: int,
    team_colors_bgr: dict,
    team_names: tuple[str, str],
    attacking_team: str | None
) -> np.ndarray:
    """
    Proiectează pozițiile jucătorilor și ale mingii pe terenul 2D orizontal în timp real.
    """
    radar = base_pitch.copy()
    rh, rw, _ = radar.shape
    m_x = 24
    m_y = 28

    for p in players:
        tid = p.get("team_id", -1)
        px, py = p["position"]
        norm_x = np.clip(px / max(1, frame_w), 0.0, 1.0)
        norm_y = np.clip(py / max(1, frame_h), 0.0, 1.0)

        rx = int(m_x + norm_x * (rw - 2 * m_x))
        ry = int(m_y + norm_y * (rh - 2 * m_y))

        if tid in (0, 1):
            color = team_colors_bgr.get(tid, (255, 255, 255))
            cv2.circle(radar, (rx, ry), 9, color, -1)
            cv2.circle(radar, (rx, ry), 9, (255, 255, 255), 1)
            if p.get("track_id") is not None:
                cv2.putText(radar, str(p["track_id"]), (rx - 7, ry - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.circle(radar, (rx, ry), 6, (180, 180, 180), -1)

    if ball_pos is not None:
        bx = int(m_x + np.clip(ball_pos[0] / max(1, frame_w), 0.0, 1.0) * (rw - 2 * m_x))
        by = int(m_y + np.clip(ball_pos[1] / max(1, frame_h), 0.0, 1.0) * (rh - 2 * m_y))
        cv2.circle(radar, (bx, by), 7, (0, 230, 255), -1)
        cv2.circle(radar, (bx, by), 9, (0, 0, 0), 2)

    # Bara de legenda sus pe radarul orizontal
    cv2.rectangle(radar, (0, 0), (rw, 22), (25, 25, 25), -1)
    c0 = team_colors_bgr.get(0, (255, 255, 255))
    c1 = team_colors_bgr.get(1, (255, 255, 255))
    cv2.circle(radar, (20, 11), 6, c0, -1)
    cv2.putText(radar, team_names[0], (32, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.circle(radar, (220, 11), 6, c1, -1)
    cv2.putText(radar, team_names[1], (232, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    if attacking_team:
        cv2.putText(radar, f"FAZA ATAC: {attacking_team}", (rw - 260, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1)

    return radar


def generate_2d_tactical_pitch_map(
    coords_x: list[float],
    coords_y: list[float],
    frame_w: int,
    frame_h: int,
    team_name: str,
    save_path: str
):
    """
    Generează harta tactică 2D a terenului de fotbal cu zonele de ocupare ale echipei.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#2e7d32")

    pitch_w, pitch_h = 100.0, 68.0
    ax.add_patch(patches.Rectangle((0, 0), pitch_w, pitch_h, fill=False, edgecolor="white", linewidth=2))
    ax.plot([50, 50], [0, pitch_h], color="white", linewidth=1.8)
    ax.add_patch(patches.Circle((50, 34), 9.15, fill=False, edgecolor="white", linewidth=1.8))

    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, fill=False, edgecolor="white", linewidth=1.8))
    ax.add_patch(patches.Rectangle((83.5, 13.84), 16.5, 40.32, fill=False, edgecolor="white", linewidth=1.8))
    ax.add_patch(patches.Rectangle((0, 24.84), 5.5, 18.32, fill=False, edgecolor="white", linewidth=1.5))
    ax.add_patch(patches.Rectangle((94.5, 24.84), 5.5, 18.32, fill=False, edgecolor="white", linewidth=1.5))

    if coords_x and coords_y:
        nx = np.clip(np.array(coords_x) / max(1, frame_w), 0.0, 1.0) * pitch_w
        ny = (1.0 - np.clip(np.array(coords_y) / max(1, frame_h), 0.0, 1.0)) * pitch_h

        hb = ax.hexbin(nx, ny, gridsize=22, cmap="YlOrRd", alpha=0.68, mincnt=1, extent=[0, pitch_w, 0, pitch_h])
        plt.colorbar(hb, ax=ax, label="Densitate Pozițională")

        mean_x, mean_y = float(np.mean(nx)), float(np.mean(ny))
        ax.scatter([mean_x], [mean_y], color="cyan", edgecolor="black", s=160, zorder=5, label="Centru Greutate Echipă")
        ax.legend(loc="upper right")

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 70)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Hartă Tactică 2D Teren — {team_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()