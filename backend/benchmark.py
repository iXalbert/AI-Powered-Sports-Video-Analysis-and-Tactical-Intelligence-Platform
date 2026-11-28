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
from ultralytics import YOLO, RTDETR
from sklearn.metrics import silhouette_score
from backend.team_classifier import TeamClassifier
from backend.pipeline import resolve_sport_class_ids


BENCHMARK_DIR = "benchmarks"
os.makedirs(BENCHMARK_DIR, exist_ok=True)


def _load_model(model_name: str):
    if "rtdetr" in model_name.lower():
        return RTDETR(model_name)
    return YOLO(model_name)


def run_single_configuration(
    video_path: str,
    model_name: str,
    tracker_name: str,
    device: str,
    max_frames: int = 120
) -> dict:
    model = _load_model(model_name)
    player_class_ids, ball_class_ids = resolve_sport_class_ids(model)
    target_classes = player_class_ids + ball_class_ids
    classifier = TeamClassifier(outlier_threshold=72.0)

    cap = cv2.VideoCapture(video_path)
    latencies_ms = []
    detections_per_frame = []
    ball_detections = 0
    confidences = []
    unique_track_ids = set()
    track_team_history = {}

    # Warm-up (3 cadre)
    for _ in range(3):
        ret, warm_frame = cap.read()
        if not ret:
            break
        _ = model(warm_frame, classes=target_classes, device=device, verbose=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Calibrare TeamClassifier pe primele 30 cadre
    cal_idx = 0
    while cap.isOpened() and cal_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, classes=player_class_ids, device=device, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        for xyxy in dets.xyxy:
            classifier.collect_sample(frame, xyxy)
        cal_idx += 1
    classifier.fit()

    sil_score = 0.0
    if classifier.is_calibrated and len(classifier.calibration_samples) >= 10:
        samples_arr = np.array(classifier.calibration_samples)
        labels_arr = classifier.kmeans.labels_
        if len(set(labels_arr)) == 2:
            sil_score = float(silhouette_score(samples_arr, labels_arr))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_name,
            classes=target_classes,
            conf=0.18,
            device=device,
            verbose=False
        )[0]
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

        all_dets = sv.Detections.from_ultralytics(results)
        dets = all_dets[np.isin(all_dets.class_id, player_class_ids)]
        b_dets = all_dets[np.isin(all_dets.class_id, ball_class_ids)]

        detections_per_frame.append(len(dets))
        if len(b_dets) > 0:
            ball_detections += 1

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

    consistencies = []
    for tid, votes in track_team_history.items():
        if len(votes) >= 3:
            majority = max(votes.count(0), votes.count(1))
            consistencies.append((majority / len(votes)) * 100.0)

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    short_tracker = tracker_name.replace(".yaml", "")

    return {
        "model": model_name,
        "tracker": short_tracker,
        "device": device.upper(),
        "frames_evaluated": frame_count,
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "avg_players_per_frame": round(float(np.mean(detections_per_frame)) if detections_per_frame else 0.0, 2),
        "ball_recall_frames_pct": round((ball_detections / max(1, frame_count)) * 100.0, 1),
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

    labels = [f"{r['model'].replace('.pt','')} | {r['tracker']} ({r['device']})" for r in results]
    y_pos = np.arange(len(labels))
    fig_h = max(4.2, len(labels) * 0.65 + 1.5)

    fps_vals = [r["fps"] for r in results]
    lat_vals = [r["avg_latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_h))
    ax1.barh(y_pos, fps_vals, color="#2b7bba", height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_title("Viteză Inferență (FPS — mai mare = mai bun)")
    ax1.set_xlabel("Cadre pe secundă (FPS)")
    ax1.set_xlim(0, max(fps_vals or [10]) * 1.18)
    for i, v in enumerate(fps_vals):
        ax1.text(v + max(fps_vals) * 0.02, i, f"{v} FPS", va="center", fontsize=8.5)

    ax2.barh(y_pos, lat_vals, color="#d95f02", height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("Latență Medie per Cadru (ms — mai mic = mai bun)")
    ax2.set_xlabel("Milisecunde (ms)")
    ax2.set_xlim(0, max(lat_vals or [10]) * 1.22)
    for i, v in enumerate(lat_vals):
        ax2.text(v + max(lat_vals) * 0.02, i, f"{v} ms", va="center", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(perf_chart_path, dpi=150)
    plt.close()

    players_vals = [r["avg_players_per_frame"] for r in results]
    ids_vals = [r["unique_track_ids"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_h))
    ax1.barh(y_pos, players_vals, color="#1b9e77", height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_title("Media Jucătorilor Detectați / Cadru")
    ax1.set_xlabel("Nr. jucători")
    ax1.set_xlim(0, max(players_vals or [5]) * 1.18)
    for i, v in enumerate(players_vals):
        ax1.text(v + max(players_vals) * 0.02, i, f"{v}", va="center", fontsize=8.5)

    ax2.barh(y_pos, ids_vals, color="#7570b3", height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("ID-uri Unice Generate (Mai mic = Tracking mai stabil)")
    ax2.set_xlabel("Total ID-uri")
    ax2.set_xlim(0, max(ids_vals or [5]) * 1.18)
    for i, v in enumerate(ids_vals):
        ax2.text(v + max(ids_vals) * 0.02, i, f"{v}", va="center", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(acc_chart_path, dpi=150)
    plt.close()

    return {
        "csv_path": csv_path,
        "perf_chart_path": perf_chart_path,
        "acc_chart_path": acc_chart_path
    }


def run_full_benchmark(
    video_path: str,
    models: list[str] = None,
    trackers: list[str] = None,
    devices: list[str] = None,
    max_frames: int = 120,
    progress_callback=None
) -> dict:
    if models is None:
        models = ["yolov8n.pt", "yolov8s.pt"]
    if trackers is None:
        trackers = ["bytetrack.yaml"]
    if devices is None:
        devices = ["mps"] if torch.backends.mps.is_available() else ["cpu"]

    configs = [(m, t, d) for m in models for t in trackers for d in devices]
    all_results = []

    for idx, (m, t, d) in enumerate(configs):
        if progress_callback:
            progress_callback(int((idx / len(configs)) * 90), f"Evaluare {m} + {t.replace('.yaml','')} pe {d.upper()}...")
        res = run_single_configuration(video_path, m, t, d, max_frames=max_frames)
        all_results.append(res)

    artifacts = generate_benchmark_artifacts(all_results)
    if progress_callback:
        progress_callback(100, "Evaluare finalizată!")

    return {
        "results": all_results,
        "artifacts": artifacts
    }


def evaluate_events_against_ground_truth(
    detected_events: list[dict],
    ground_truth_events: list[dict],
    tolerance_sec: float = 3.0
) -> dict:
    matched_gt = set()
    matched_det = set()
    timestamp_errors = []
    team_matches = 0
    flank_matches = 0

    for g_idx, gt in enumerate(ground_truth_events):
        gt_time = float(gt["timestamp_sec"])
        best_d_idx = None
        best_diff = float("inf")

        for d_idx, det in enumerate(detected_events):
            if d_idx in matched_det:
                continue
            diff = abs(float(det["timestamp_sec"]) - gt_time)
            if diff <= tolerance_sec and diff < best_diff:
                best_diff = diff
                best_d_idx = d_idx

        if best_d_idx is not None:
            matched_gt.add(g_idx)
            matched_det.add(best_d_idx)
            timestamp_errors.append(best_diff)

            det_ev = detected_events[best_d_idx]
            if gt.get("attacking_team") and det_ev.get("attacking_team"):
                if gt["attacking_team"].strip().lower() == det_ev["attacking_team"].strip().lower():
                    team_matches += 1
            if gt.get("flank") and det_ev.get("flank"):
                if gt["flank"].strip().lower() == det_ev["flank"].strip().lower():
                    flank_matches += 1

    tp = len(matched_gt)
    fp = max(0, len(detected_events) - tp)
    fn = max(0, len(ground_truth_events) - tp)

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_err = float(np.mean(timestamp_errors)) if timestamp_errors else 0.0
    team_acc = (team_matches / tp) * 100.0 if tp > 0 else 0.0
    flank_acc = (flank_matches / tp) * 100.0 if tp > 0 else 0.0

    summary = {
        "ground_truth_count": len(ground_truth_events),
        "detected_count": len(detected_events),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "tolerance_sec": tolerance_sec,
        "precision_pct": round(precision, 1),
        "recall_pct": round(recall, 1),
        "f1_score_pct": round(f1, 1),
        "mean_timestamp_error_sec": round(mean_err, 2),
        "team_attribution_acc_pct": round(team_acc, 1),
        "flank_classification_acc_pct": round(flank_acc, 1)
    }

    csv_path = os.path.join(BENCHMARK_DIR, "ground_truth_evaluation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return summary