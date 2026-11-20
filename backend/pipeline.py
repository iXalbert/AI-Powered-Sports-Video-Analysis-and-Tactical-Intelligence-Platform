import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from backend.team_classifier import TeamClassifier


class VideoAnalysisPipeline:
    def __init__(self, model_name: str = "yolov8n.pt"):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = YOLO(model_name)

    def _calibrate_teams(self, input_path: str, start_frame: int, end_frame: int, max_sample_frames: int = 45) -> TeamClassifier:
        classifier = TeamClassifier(outlier_threshold=68.0)
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        span = max(1, end_frame - start_frame)
        step = max(1, span // max_sample_frames)

        current_frame = start_frame
        sampled = 0
        while cap.isOpened() and current_frame < end_frame and sampled < max_sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if (current_frame - start_frame) % step == 0:
                results = self.model(frame, device=self.device, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.class_id == 0]
                for xyxy in detections.xyxy:
                    classifier.collect_sample(frame, xyxy)
                sampled += 1
            current_frame += 1

        cap.release()
        classifier.fit()
        return classifier

    def process_video(
        self,
        input_path: str,
        session_dir: str,
        team_names: tuple[str, str] = ("Team A", "Team B"),
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        progress_callback=None
    ) -> dict:
        if progress_callback:
            progress_callback(10)

        video_info = sv.VideoInfo.from_video_path(input_path)
        fps = video_info.fps or 25.0

        start_frame = int(max(0, start_sec * fps))
        if duration_sec and duration_sec > 0:
            end_frame = int(min(video_info.total_frames, start_frame + duration_sec * fps))
        else:
            end_frame = int(min(video_info.total_frames, start_frame + fps * 60 * 5))

        total_target_frames = max(1, end_frame - start_frame)

        team_classifier = self._calibrate_teams(input_path, start_frame, end_frame)
        if progress_callback:
            progress_callback(18)

        output_video_path = os.path.join(session_dir, "annotated_video.mp4")
        metadata_path = os.path.join(session_dir, "tracking_data.json")

        tracking_records = []
        track_team_votes = {}
        last_positions = {}

        team_metrics = {
            0: {"name": team_names[0], "zone_counts": {"left": 0, "center": 0, "right": 0}, "player_distances": {}, "x_positions": [], "attacking_frames": 0},
            1: {"name": team_names[1], "zone_counts": {"left": 0, "center": 0, "right": 0}, "player_distances": {}, "x_positions": [], "attacking_frames": 0}
        }
        timeline_events = []

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_info.width, video_info.height))

        frame_idx = start_frame
        processed_count = 0

        while cap.isOpened() and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", device=self.device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]

            frame_record = {
                "frame_index": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 2),
                "attacking_team": None,
                "defending_team": None,
                "players": []
            }

            team_frame_x = {0: [], 1: []}
            annotated = frame.copy()

            for i, xyxy in enumerate(detections.xyxy):
                tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
                raw_team, color_bgr = team_classifier.predict(frame, xyxy)

                if tid is not None:
                    votes = track_team_votes.setdefault(tid, {0: 0, 1: 0, -1: 0})
                    votes[raw_team] += 1
                    stable_team = max(votes, key=votes.get)
                    if stable_team in (0, 1) and team_classifier.is_calibrated:
                        color_bgr = team_classifier.team_colors_bgr[stable_team]
                    else:
                        color_bgr = (160, 160, 160)
                else:
                    stable_team = raw_team

                cx = float((xyxy[0] + xyxy[2]) / 2)
                by = float(xyxy[3])

                if stable_team in (0, 1):
                    team_frame_x[stable_team].append(cx)
                    team_metrics[stable_team]["x_positions"].append(cx)

                    if tid is not None:
                        if tid in last_positions:
                            dist = np.hypot(cx - last_positions[tid][0], by - last_positions[tid][1])
                            if dist < 100:
                                p_dists = team_metrics[stable_team]["player_distances"]
                                p_dists[tid] = p_dists.get(tid, 0.0) + dist
                        last_positions[tid] = (cx, by)

                team_tag = team_names[stable_team] if stable_team in (0, 1) else "Ref/GK"
                label_text = f"#{tid} {team_tag}" if tid is not None else team_tag

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(annotated, label_text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

                frame_record["players"].append({
                    "track_id": tid,
                    "team_id": stable_team,
                    "team_name": team_tag,
                    "bbox": [round(float(c), 2) for c in xyxy],
                    "position": [round(cx, 2), round(by, 2)]
                })

            avg_x_0 = sum(team_frame_x[0]) / len(team_frame_x[0]) if team_frame_x[0] else None
            avg_x_1 = sum(team_frame_x[1]) / len(team_frame_x[1]) if team_frame_x[1] else None

            if avg_x_0 is not None and avg_x_1 is not None:
                spread_0 = max(team_frame_x[0]) - min(team_frame_x[0])
                spread_1 = max(team_frame_x[1]) - min(team_frame_x[1])
                att_id = 0 if spread_0 >= spread_1 else 1
                def_id = 1 - att_id

                team_metrics[att_id]["attacking_frames"] += 1
                frame_record["attacking_team"] = team_names[att_id]
                frame_record["defending_team"] = team_names[def_id]

            third = video_info.width / 3
            for t_id in (0, 1):
                if team_frame_x[t_id]:
                    tx = sum(team_frame_x[t_id]) / len(team_frame_x[t_id])
                    if tx < third:
                        team_metrics[t_id]["zone_counts"]["left"] += 1
                    elif tx < 2 * third:
                        team_metrics[t_id]["zone_counts"]["center"] += 1
                    else:
                        team_metrics[t_id]["zone_counts"]["right"] += 1

            total_players_frame = len(team_frame_x[0]) + len(team_frame_x[1])
            if total_players_frame >= 6 and processed_count % int(fps * 2) == 0 and frame_record["attacking_team"]:
                timeline_events.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": frame_record["timestamp_sec"],
                    "attacking_team": frame_record["attacking_team"],
                    "defending_team": frame_record["defending_team"],
                    "players_count": total_players_frame
                })

            if frame_record["attacking_team"]:
                phase_banner = f"ATTACK: {frame_record['attacking_team']}  |  DEFENSE: {frame_record['defending_team']}"
                cv2.rectangle(annotated, (10, 10), (540, 42), (20, 20, 20), -1)
                cv2.putText(annotated, phase_banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            out.write(annotated)
            tracking_records.append(frame_record)

            frame_idx += 1
            processed_count += 1
            if progress_callback and processed_count % 15 == 0:
                progress_callback(18 + int((processed_count / total_target_frames) * 72))

        cap.release()
        out.release()

        total_phase_frames = (team_metrics[0]["attacking_frames"] + team_metrics[1]["attacking_frames"]) or 1
        teams_summary = {}

        for t_id in (0, 1):
            tm = team_metrics[t_id]
            z_total = sum(tm["zone_counts"].values()) or 1
            top_p = sorted(tm["player_distances"].items(), key=lambda x: x[1], reverse=True)[:5]
            avg_line_pct = round((np.mean(tm["x_positions"]) / video_info.width) * 100, 1) if tm["x_positions"] else 50.0
            bgr = team_classifier.team_colors_bgr.get(t_id, (200, 200, 200))
            hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

            teams_summary[str(t_id)] = {
                "name": tm["name"],
                "jersey_hex": hex_color,
                "attacking_share_pct": round((tm["attacking_frames"] / total_phase_frames) * 100, 1),
                "avg_line_height_pct": avg_line_pct,
                "attacking_distribution_pct": {
                    "left": round((tm["zone_counts"]["left"] / z_total) * 100, 1),
                    "center": round((tm["zone_counts"]["center"] / z_total) * 100, 1),
                    "right": round((tm["zone_counts"]["right"] / z_total) * 100, 1)
                },
                "top_active_players": [{"track_id": pid, "movement_score": round(dist, 1)} for pid, dist in top_p]
            }

        summary_stats = {
            "video_info": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": fps,
                "start_sec": round(start_frame / fps, 1),
                "end_sec": round(frame_idx / fps, 1),
                "analyzed_seconds": round(processed_count / fps, 1)
            },
            "teams": teams_summary,
            "timeline_events": timeline_events,
            "frames": tracking_records
        }

        with open(metadata_path, "w") as f:
            json.dump(summary_stats, f, indent=2)

        return summary_stats