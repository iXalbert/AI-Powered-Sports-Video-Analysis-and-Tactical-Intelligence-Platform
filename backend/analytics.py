import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_visual_analytics(stats: dict, input_video_path: str, session_dir: str) -> dict:
    teams = stats["teams"]
    team_coords = {0: {"x": [], "y": []}, 1: {"x": [], "y": []}}

    for frame in stats["frames"]:
        for p in frame["players"]:
            tid = p.get("team_id")
            if tid in (0, 1):
                team_coords[tid]["x"].append(p["position"][0])
                team_coords[tid]["y"].append(p["position"][1])

    heatmap_urls = {}
    for t_id in (0, 1):
        t_info = teams[str(t_id)]
        h_filename = f"heatmap_team_{t_id}.png"
        h_path = os.path.join(session_dir, h_filename)

        plt.figure(figsize=(8, 4.5))
        if team_coords[t_id]["x"]:
            plt.hexbin(team_coords[t_id]["x"], team_coords[t_id]["y"], gridsize=25, cmap='YlOrRd', mincnt=1)
            plt.colorbar(label='Densitate Jucători')
        plt.gca().invert_yaxis()
        plt.title(f"Heatmap Pozițional — {t_info['name']}")
        plt.xlabel("Lățime Teren (px)")
        plt.ylabel("Adâncime Teren (px)")
        plt.tight_layout()
        plt.savefig(h_path)
        plt.close()
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
    plt.title("Timeline Faze de Atac & Presiune pe Echipe")
    plt.xlabel("Timp Meci (secunde)")
    plt.ylabel("Jucători în Cadru")
    plt.xlim(0, max(10, stats["video_info"]["analyzed_seconds"]))
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
            key_frames.append({
                "filename": img_filename,
                "path": img_path,
                "timestamp_sec": ev["timestamp_sec"],
                "attacking_team": ev["attacking_team"],
                "players_count": ev["players_count"]
            })
    cap.release()

    return {
        "heatmap_team_0": heatmap_urls["0"],
        "heatmap_team_1": heatmap_urls["1"],
        "timeline_path": timeline_path,
        "key_frames": key_frames
    }