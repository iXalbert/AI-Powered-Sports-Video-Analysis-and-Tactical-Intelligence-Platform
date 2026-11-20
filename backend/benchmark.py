import os
import time
import csv
import cv2
import torch
import numpy as np
import supervision as sv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import silhouette_score
from backend.team_classifier import TeamClassifier


BENCHMARK_DIR = "benchmarks"
os.makedirs(BENCHMARK_DIR, exist_ok=True)


def run_single_configuration(video_path: str, model_name: str, device: str, max_frames: int = 150) -> dict:
    model = YOLO(model_name)
    classifier = TeamClassifier(outlier_threshold=68.0)

    cap = cv2.VideoCapture(video_path)
    latencies_ms = []
    detections_per_frame = []
    confidences = []
    unique_track_ids = set()
    track_team_history = {}

    # Warm-up scurt (3 cadre) pentru a elimina overhead-ul initial de alocare MPS/CPU
    for _ in range(3):
        ret, warm_frame = cap.read()
        if not ret:
            break
        _ = model(warm_frame, device=device, verbose=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Calibrare rapida TeamClassifier pe primele 30 cadre
    cal_idx = 0
    while cap.isOpened() and cal_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, device=device, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        dets = dets[dets.class_id == 0]
        for xyxy in dets.xyxy:
            classifier.collect_sample(frame, xyxy)
        cal_idx += 1
    classifier.fit()

    # Calculam Silhouette Score pe esantioanele de tricouri colectate
    sil_score = 0.0
    if classifier.is_calibrated and len(classifier.calibration_samples) >= 10:
        samples_arr = np.array(classifier.calibration_samples)
        labels_arr = classifier.kmeans.labels_
        if len(set(labels_arr)) == 2:
            sil_score = float(silhouette_score(samples_arr, labels_arr))

    # Rulare benchmark cronometrat pe max_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device, verbose=False)[0]
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

        dets = sv.Detections.from_ultralytics(results)
        dets = dets[dets.class_id == 0]
        detections_per_frame.append(len(dets))

        if dets.confidence is not None and len(dets.confidence) > 0:
            confidences.extend([float(c) for c in dets.confidence])

        for i, xyxy in enumerate(dets.xyxy):
            tid = int(dets.tracker_id[i]) if dets.tracker_id is not None else None
            if tid is not None:
                unique_track_ids.add(tid)
                team_id, _ = classifier.predict(frame, xyxy)
                if team_id in (0, 1):
                    track_team_history.setdefault(tid, []).append(team_id)

        frame_count += 1

    cap.release()

    # Consistenta atribuirii echipei pe acelasi track_id
    consistencies = []
    for tid, votes in track_team_history.items():
        if len(votes) >= 3:
            majority = max(votes.count(0), votes.count(1))
            consistencies.append((majority / len(votes)) * 100.0)

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "model": model_name,
        "device": device.upper(),
        "frames_evaluated": frame_count,
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "avg_players_per_frame": round(float(np.mean(detections_per_frame)) if detections_per_frame else 0.0, 2),
        "avg_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
        "unique_track_ids": len(unique_track_ids),
        "team_silhouette_score": round(sil_score, 3),
        "team_consistency_pct": round(float(np.mean(consistencies)) if consistencies else 0.0, 1)
    }


def generate_benchmark_artifacts(results: list[dict], output_dir: str = BENCHMARK_DIR) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    perf_chart_path = os.path.join(output_dir, "latency_fps_comparison.png")
    acc_chart_path = os.path.join(output_dir, "detection_stability_comparison.png")

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    labels = [f"{r['model']}\n({r['device']})" for r in results]
    fps_vals = [r["fps"] for r in results]
    lat_vals = [r["avg_latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(labels, fps_vals, color="#2b7bba")
    ax1.set_title("Viteză Inferență (FPS — mai mare = mai bun)")
    ax1.set_ylabel("Cadre pe secundă (FPS)")
    for i, v in enumerate(fps_vals):
        ax1.text(i, v + 0.5, f"{v}", ha="center", fontsize=9)

    ax2.bar(labels, lat_vals, color="#d95f02")
    ax2.set_title("Latență Medie per Cadru (ms — mai mic = mai bun)")
    ax2.set_ylabel("Milisecunde (ms)")
    for i, v in enumerate(lat_vals):
        ax2.text(i, v + 0.5, f"{v} ms", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(perf_chart_path)
    plt.close()

    players_vals = [r["avg_players_per_frame"] for r in results]
    ids_vals = [r["unique_track_ids"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(labels, players_vals, color="#1b9e77")
    ax1.set_title("Media Jucătorilor Detectați / Cadru")
    ax1.set_ylabel("Nr. jucători")
    for i, v in enumerate(players_vals):
        ax1.text(i, v + 0.1, f"{v}", ha="center", fontsize=9)

    ax2.bar(labels, ids_vals, color="#7570b3")
    ax2.set_title("ID-uri Unice Generate (Fragmentare Tracking)")
    ax2.set_ylabel("Total ID-uri")
    for i, v in enumerate(ids_vals):
        ax2.text(i, v + 0.3, f"{v}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(acc_chart_path)
    plt.close()

    return {
        "csv_path": csv_path,
        "perf_chart_path": perf_chart_path,
        "acc_chart_path": acc_chart_path
    }


def run_full_benchmark(
    video_path: str,
    models: list[str] = None,
    devices: list[str] = None,
    max_frames: int = 150,
    progress_callback=None
) -> dict:
    if models is None:
        models = ["yolov8n.pt", "yolov8s.pt"]
    if devices is None:
        devices = ["mps"] if torch.backends.mps.is_available() else ["cpu"]

    configs = [(m, d) for m in models for d in devices]
    all_results = []

    for idx, (m, d) in enumerate(configs):
        if progress_callback:
            progress_callback(int((idx / len(configs)) * 90), f"Se evaluează {m} pe {d.upper()}...")
        res = run_single_configuration(video_path, m, d, max_frames=max_frames)
        all_results.append(res)

    artifacts = generate_benchmark_artifacts(all_results)
    if progress_callback:
        progress_callback(100, "Evaluare finalizată!")

    return {
        "results": all_results,
        "artifacts": artifacts
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Utilizare: python -m backend.benchmark <cale_video.mp4>")
        sys.exit(1)
    devs = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    out = run_full_benchmark(sys.argv[1], models=["yolov8n.pt", "yolov8s.pt"], devices=devs, max_frames=120)
    for row in out["results"]:
        print(row)
    print("Fișiere salvate în:", out["artifacts"])