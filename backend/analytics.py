import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backend.pitch_radar import generate_2d_tactical_pitch_map


def extract_10s_clip(video_path: str, center_sec: float, output_clip_path: str, window_half: float = 5.0) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = max(0, int((center_sec - window_half) * fps))
    end_f = min(total_frames, int((center_sec + window_half) * fps))

    if end_f <= start_f:
        cap.release()
        return False

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_clip_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    curr = start_f
    while cap.isOpened() and curr < end_f:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        curr += 1

    cap.release()
    out.release()
    return os.path.exists(output_clip_path)


def generate_visual_analytics(stats: dict, input_video_path: str, session_dir: str) -> dict:
    teams = stats["teams"]
    sport = stats.get("sport", "Fotbal (Football)")
    annotated_video_path = os.path.join(session_dir, "annotated_video.mp4")

    tactical_frames = [f for f in stats["frames"] if f.get("is_tactical", True)]
    if not tactical_frames:
        tactical_frames = stats["frames"]

    # Detectăm orientarea inițială (prima jumătate a cadrelor tactice) pentru a normaliza direcția de atac (stânga -> dreapta)
    mid_idx = max(1, len(tactical_frames) // 2)
    first_half_x = {0: [], 1: []}
    for fr in tactical_frames[:mid_idx]:
        for p in fr["players"]:
            tid = p.get("team_id")
            if tid in (0, 1):
                first_half_x[tid].append(p.get("pitch_norm", (0.5, 0.5))[0])

    # Dacă Echipa 0 începe în dreapta (> 0.52) față de Echipa 1 (< 0.48), o normalizăm spre dreapta
    base_flip = {0: False, 1: False}
    if first_half_x[0] and first_half_x[1]:
        m0 = float(np.mean(first_half_x[0]))
        m1 = float(np.mean(first_half_x[1]))
        if m0 > m1:
            base_flip[0] = True  # Echipa 0 ataca spre stânga în prima parte -> oglindim spre dreapta
        else:
            base_flip[1] = True  # Echipa 1 ataca spre stânga în prima parte -> oglindim spre dreapta

    team_coords = {0: {"nx": [], "ny": []}, 1: {"nx": [], "ny": []}}
    for idx, frame in enumerate(tactical_frames):
        # Verificăm dacă în a doua jumătate a clipului echipele au schimbat terenul (pauză / 2ND HALF)
        frame_x = {0: [], 1: []}
        for p in frame["players"]:
            tid = p.get("team_id")
            if tid in (0, 1):
                frame_x[tid].append(p.get("pitch_norm", (0.5, 0.5))[0])

        halftime_swapped = False
        if idx >= mid_idx and frame_x[0] and frame_x[1] and first_half_x[0] and first_half_x[1]:
            init_diff = float(np.mean(first_half_x[0])) - float(np.mean(first_half_x[1]))
            curr_diff = float(np.mean(frame_x[0])) - float(np.mean(frame_x[1]))
            if (init_diff * curr_diff) < -0.02:
                halftime_swapped = True

        for p in frame["players"]:
            tid = p.get("team_id")
            if tid in (0, 1):
                nx, ny = p.get("pitch_norm", (0.5, 0.5))
                should_flip = base_flip[tid] ^ halftime_swapped
                if should_flip:
                    nx = 1.0 - nx
                    ny = 1.0 - ny
                team_coords[tid]["nx"].append(nx)
                team_coords[tid]["ny"].append(ny)

    heatmap_urls = {}
    for t_id in (0, 1):
        t_info = teams[str(t_id)]
        h_filename = f"pitch_2d_team_{t_id}.png"
        h_path = os.path.join(session_dir, h_filename)
        generate_2d_tactical_pitch_map(
            norm_x_list=team_coords[t_id]["nx"],
            norm_y_list=team_coords[t_id]["ny"],
            team_name=f"{t_info['name']} (Atac Stânga → Dreapta)",
            sport=sport,
            save_path=h_path
        )
        heatmap_urls[str(t_id)] = h_filename

    timeline_path = os.path.join(session_dir, "timeline.png")
    events = stats["timeline_events"]
    t0_name = teams["0"]["name"]
    t1_name = teams["1"]["name"]

    plt.figure(figsize=(10, 3.2))
    if events:
        t0_ev = [e for e in events if e["attacking_team"] == t0_name]
        t1_ev = [e for e in events if e["attacking_team"] == t1_name]
        if t0_ev:
            plt.stem([e["timestamp_sec"] for e in t0_ev], [e["players_count"] for e in t0_ev],
                     linefmt='b-', markerfmt='bo', basefmt=' ', label=f"Atac {t0_name}")
        if t1_ev:
            plt.stem([e["timestamp_sec"] for e in t1_ev], [e["players_count"] for e in t1_ev],
                     linefmt='r-', markerfmt='ro', basefmt=' ', label=f"Atac {t1_name}")
        plt.legend()
    plt.title(f"Timeline Faze & Evenimente ({sport} — Doar Cadre Tactice)")
    plt.xlabel("Timp Meci (secunde)")
    plt.ylabel("Jucători în Cadru")
    start_s = stats["video_info"].get("start_sec", 0)
    end_s = stats["video_info"].get("end_sec", max(10, stats["video_info"]["analyzed_seconds"]))
    plt.xlim(start_s, max(start_s + 5, end_s))
    plt.tight_layout()
    plt.savefig(timeline_path)
    plt.close()

    key_frames = []
    top_events = sorted(events, key=lambda x: x["players_count"], reverse=True)[:3]
    cap = cv2.VideoCapture(input_video_path)

    for idx, ev in enumerate(top_events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev["frame_index"])
        ret, frame = cap.read()
        if ret:
            img_filename = f"key_moment_{idx+1}.jpg"
            img_path = os.path.join(session_dir, img_filename)
            cv2.imwrite(img_path, frame)

            clip_filename = f"key_clip_{idx+1}.mp4"
            clip_path = os.path.join(session_dir, clip_filename)
            rel_sec = max(0.0, ev["timestamp_sec"] - start_s)
            has_clip = extract_10s_clip(annotated_video_path, rel_sec, clip_path, window_half=5.0)

            key_frames.append({
                "filename": img_filename,
                "clip_filename": clip_filename if has_clip else None,
                "path": img_path,
                "timestamp_sec": ev["timestamp_sec"],
                "attacking_team": ev["attacking_team"],
                "flank": ev.get("flank", "Centru"),
                "players_count": ev["players_count"]
            })
    cap.release()

    return {
        "heatmap_team_0": heatmap_urls["0"],
        "heatmap_team_1": heatmap_urls["1"],
        "timeline_path": timeline_path,
        "key_frames": key_frames
    }