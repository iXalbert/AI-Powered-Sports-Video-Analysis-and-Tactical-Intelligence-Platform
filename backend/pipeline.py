import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO, RTDETR
from backend.team_classifier import TeamClassifier
from backend.scoreboard_reader import ScoreboardReader
from backend.camera_classifier import TacticalCameraFilter
from backend.pitch_radar import (
    PitchHomographyMapper,
    draw_2d_pitch_canvas,
    render_live_radar_frame
)


def get_sport_event_labels(sport: str) -> dict:
    s = sport.lower()
    if "baschet" in s or "basketball" in s:
        return {"pass": "Pasă", "shot": "Aruncare la coș", "chance": "Pătrundere / Ocazie coș"}
    if "handbal" in s or "handball" in s:
        return {"pass": "Pasă", "shot": "Aruncare spre poartă", "chance": "Atac 6m / Ocazie gol"}
    if "volei" in s or "volleyball" in s or "tenis" in s or "badminton" in s:
        return {"pass": "Schimb / Preluare", "shot": "Lovitură atac / Smash", "chance": "Raliu intens"}
    if "hochei" in s or "hockey" in s:
        return {"pass": "Pasă puc", "shot": "Șut spre poartă", "chance": "Presiune în zona ofensivă"}
    if "rugby" in s:
        return {"pass": "Pasă", "shot": "Șut / Pătrundere", "chance": "Avans în 22m"}
    return {"pass": "Pasă", "shot": "Șut spre poartă", "chance": "Ocazie / Fază periculoasă"}


def resolve_sport_class_ids(model) -> tuple[list[int], list[int]]:
    """
    Detectează automat ID-urile claselor pentru jucători și minge din model.names,
    suportând atât modele generice COCO, cât și modele fine-tuned pe sport.
    """
    names = {int(k): str(v).lower().strip() for k, v in model.names.items()}
    player_ids = [
        cid for cid, name in names.items()
        if name in ("person", "player", "goalkeeper", "players")
    ]
    ball_ids = [
        cid for cid, name in names.items()
        if name in ("sports ball", "ball", "football", "soccer ball")
    ]
    if not player_ids:
        player_ids = [0]
    if not ball_ids:
        ball_ids = [32]
    return player_ids, ball_ids


class VideoAnalysisPipeline:
    def __init__(self, model_name: str = "yolov8n.pt"):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.current_model_name = model_name
        self.model = self._load_detector(model_name)
        self.scoreboard_reader = ScoreboardReader()

    def _load_detector(self, model_name: str):
        if "rtdetr" in model_name.lower():
            return RTDETR(model_name)
        return YOLO(model_name)

    def _calibrate_teams(
        self,
        input_path: str,
        start_frame: int,
        end_frame: int,
        cam_filter: TacticalCameraFilter,
        color_hint_0: str = "Auto-detect",
        color_hint_1: str = "Auto-detect",
        scoreboard_lab_0: np.ndarray | None = None,
        scoreboard_lab_1: np.ndarray | None = None,
        max_sample_frames: int = 50
    ) -> TeamClassifier:
        classifier = TeamClassifier(outlier_threshold=72.0)
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        span = max(1, end_frame - start_frame)
        step = max(1, span // max_sample_frames)
        player_class_ids, _ = resolve_sport_class_ids(self.model)

        current_frame = start_frame
        sampled = 0
        while cap.isOpened() and current_frame < end_frame and sampled < max_sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if (current_frame - start_frame) % step == 0:
                results = self.model(frame, classes=player_class_ids, device=self.device, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                is_tact, _ = cam_filter.classify_frame(frame, detections.xyxy)
                if is_tact:
                    for xyxy in detections.xyxy:
                        classifier.collect_sample(frame, xyxy)
                    sampled += 1
            current_frame += 1

        cap.release()
        classifier.fit(
            color_hint_0=color_hint_0,
            color_hint_1=color_hint_1,
            scoreboard_lab_0=scoreboard_lab_0,
            scoreboard_lab_1=scoreboard_lab_1
        )
        return classifier

    def process_video(
        self,
        input_path: str,
        session_dir: str,
        team_names: tuple[str, str] = ("Auto (Tabela Scor)", "Auto (Tabela Scor)"),
        color_hints: tuple[str, str] = ("Auto-detect", "Auto-detect"),
        sport: str = "Fotbal (Football)",
        homography_corners: dict | None = None,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        model_name: str = "yolov8n.pt",
        tracker_name: str = "bytetrack.yaml",
        progress_callback=None
    ) -> dict:
        if model_name != self.current_model_name:
            self.model = self._load_detector(model_name)
            self.current_model_name = model_name

        if progress_callback:
            progress_callback(6)

        video_info = sv.VideoInfo.from_video_path(input_path)
        fps = video_info.fps or 25.0

        start_frame = int(max(0, start_sec * fps))
        if duration_sec and duration_sec > 0:
            end_frame = int(min(video_info.total_frames, start_frame + duration_sec * fps))
        else:
            end_frame = int(min(video_info.total_frames, start_frame + fps * 60 * 15))

        total_target_frames = max(1, end_frame - start_frame)
        frame_stride = 2 if total_target_frames > int(fps * 120) else 1

        # 1. Citire automată a tabelei de scor (coduri 3 litere + culori echipamente)
        sb_info = self.scoreboard_reader.detect_from_video(input_path, start_frame, end_frame)
        t0_name, t1_name = team_names
        c0_hint, c1_hint = color_hints
        sb_lab0, sb_lab1 = None, None
        scoreboard_detected_meta = None

        if sb_info:
            scoreboard_detected_meta = {
                "source": sb_info["source"],
                "team_0_abbr": sb_info["team_0_abbr"],
                "team_1_abbr": sb_info["team_1_abbr"]
            }
            if t0_name.lower().startswith("auto") or t0_name == "Team A":
                t0_name = sb_info["team_0_name"]
            if t1_name.lower().startswith("auto") or t1_name == "Team B":
                t1_name = sb_info["team_1_name"]
            if c0_hint == "Auto-detect":
                c0_hint = sb_info.get("team_0_color_hint", "Auto-detect")
                sb_lab0 = sb_info.get("team_0_lab")
            if c1_hint == "Auto-detect":
                c1_hint = sb_info.get("team_1_color_hint", "Auto-detect")
                sb_lab1 = sb_info.get("team_1_lab")

        if t0_name.lower().startswith("auto"):
            t0_name = "Team A"
        if t1_name.lower().startswith("auto"):
            t1_name = "Team B"
        resolved_names = (t0_name, t1_name)

        if progress_callback:
            progress_callback(12)

        cam_filter = TacticalCameraFilter(sport=sport)
        team_classifier = self._calibrate_teams(
            input_path,
            start_frame,
            end_frame,
            cam_filter=cam_filter,
            color_hint_0=c0_hint,
            color_hint_1=c1_hint,
            scoreboard_lab_0=sb_lab0,
            scoreboard_lab_1=sb_lab1
        )
        if progress_callback:
            progress_callback(20)

        hc = homography_corners or {}
        mapper = PitchHomographyMapper(
            frame_w=video_info.width,
            frame_h=video_info.height,
            top_left_pct=(hc.get("tl_x", 18.0), hc.get("tl_y", 22.0)),
            top_right_pct=(hc.get("tr_x", 82.0), hc.get("tr_y", 22.0)),
            bottom_right_pct=(hc.get("br_x", 98.0), hc.get("br_y", 94.0)),
            bottom_left_pct=(hc.get("bl_x", 2.0), hc.get("bl_y", 94.0)),
        )

        output_video_path = os.path.join(session_dir, "annotated_video.mp4")
        metadata_path = os.path.join(session_dir, "tracking_data.json")

        radar_w = video_info.width
        radar_h = max(230, int(video_info.height * 0.42))
        base_pitch = draw_2d_pitch_canvas(width=radar_w, height=radar_h, sport=sport)
        combined_h = video_info.height + radar_h

        tracking_records = []
        track_team_votes = {}
        last_positions = {}
        ball_detections_count = 0
        tactical_frames_count = 0
        ignored_closeup_frames_count = 0

        last_ball_pos = None
        last_ball_norm = None
        ball_missing_streak = 0

        frozen_radar_players = []
        frozen_radar_ball = None
        frozen_attacking_team = None

        ev_labels = get_sport_event_labels(sport)
        last_ball_owner = None
        last_event_frame = {"pass": -999, "shot": -999, "chance": -999}
        prev_ball_norm = None

        team_metrics = {
            0: {"name": resolved_names[0], "zone_counts": {"left": 0, "center": 0, "right": 0}, "player_distances": {}, "norm_x_positions": [], "attacking_frames": 0, "passes": 0, "shots": 0},
            1: {"name": resolved_names[1], "zone_counts": {"left": 0, "center": 0, "right": 0}, "player_distances": {}, "norm_x_positions": [], "attacking_frames": 0, "passes": 0, "shots": 0}
        }
        timeline_events = []

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_fps = fps / frame_stride
        out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (video_info.width, combined_h))

        player_class_ids, ball_class_ids = resolve_sport_class_ids(self.model)
        target_classes = player_class_ids + ball_class_ids

        frame_idx = start_frame
        processed_count = 0

        while cap.isOpened() and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_stride != 0:
                frame_idx += 1
                continue

            results = self.model.track(
                frame,
                persist=True,
                tracker=tracker_name,
                classes=target_classes,
                conf=0.16,
                device=self.device,
                verbose=False
            )[0]
            all_dets = sv.Detections.from_ultralytics(results)

            player_dets = all_dets[np.isin(all_dets.class_id, player_class_ids)]
            ball_dets = all_dets[np.isin(all_dets.class_id, ball_class_ids)]

            is_tactical, cam_reason = cam_filter.classify_frame(frame, player_dets.xyxy)
            if is_tactical:
                tactical_frames_count += 1
            else:
                ignored_closeup_frames_count += 1

            current_ball_pos = None
            current_ball_norm = None
            if len(ball_dets) > 0:
                best_b_idx = int(np.argmax(ball_dets.confidence)) if ball_dets.confidence is not None else 0
                b_xyxy = ball_dets.xyxy[best_b_idx]
                bx = float((b_xyxy[0] + b_xyxy[2]) / 2)
                by = float((b_xyxy[1] + b_xyxy[3]) / 2)
                current_ball_pos = (bx, by)
                if is_tactical:
                    current_ball_norm = mapper.map_point(bx, by)
                    last_ball_pos = current_ball_pos
                    last_ball_norm = current_ball_norm
                    ball_missing_streak = 0
                    ball_detections_count += 1
            elif is_tactical and last_ball_pos is not None and ball_missing_streak < 8:
                current_ball_pos = last_ball_pos
                current_ball_norm = last_ball_norm
                ball_missing_streak += 1

            ts_sec = round(frame_idx / fps, 2)
            frame_record = {
                "frame_index": frame_idx,
                "timestamp_sec": ts_sec,
                "is_tactical": is_tactical,
                "camera_state": cam_reason,
                "attacking_team": None,
                "defending_team": None,
                "attack_flank": None,
                "ball_position": [round(current_ball_pos[0], 2), round(current_ball_pos[1], 2)] if current_ball_pos else None,
                "ball_norm": [round(current_ball_norm[0], 4), round(current_ball_norm[1], 4)] if current_ball_norm else None,
                "players": []
            }

            team_norm_x = {0: [], 1: []}
            team_norm_y = {0: [], 1: []}
            team_pixel_players = {0: [], 1: []}
            annotated = frame.copy()

            for i, xyxy in enumerate(player_dets.xyxy):
                tid = int(player_dets.tracker_id[i]) if player_dets.tracker_id is not None else None
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
                nx, ny = mapper.map_point(cx, by)

                if is_tactical and stable_team in (0, 1):
                    team_norm_x[stable_team].append(nx)
                    team_norm_y[stable_team].append(ny)
                    team_pixel_players[stable_team].append((tid, cx, by, nx, ny))
                    team_metrics[stable_team]["norm_x_positions"].append(nx)

                    if tid is not None:
                        if tid in last_positions:
                            dist = np.hypot(nx - last_positions[tid][0], ny - last_positions[tid][1]) * 100.0
                            if dist < 12.0:
                                p_dists = team_metrics[stable_team]["player_distances"]
                                p_dists[tid] = p_dists.get(tid, 0.0) + dist
                        last_positions[tid] = (nx, ny)

                team_tag = resolved_names[stable_team] if stable_team in (0, 1) else "Ref/GK"
                label_text = f"#{tid} {team_tag}" if tid is not None else team_tag

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(annotated, label_text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

                frame_record["players"].append({
                    "track_id": tid,
                    "team_id": stable_team,
                    "team_name": team_tag,
                    "bbox": [round(float(c), 2) for c in xyxy],
                    "position": [round(cx, 2), round(by, 2)],
                    "pitch_norm": [round(nx, 4), round(ny, 4)]
                })

            if current_ball_pos is not None:
                bx_i, by_i = int(current_ball_pos[0]), int(current_ball_pos[1])
                cv2.circle(annotated, (bx_i, by_i), 10, (0, 230, 255), 2)
                cv2.putText(annotated, "BALL", (bx_i + 10, by_i - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 2)

            if is_tactical:
                att_id = None
                if current_ball_pos is not None and (team_pixel_players[0] or team_pixel_players[1]):
                    candidates = []
                    for t_id in (0, 1):
                        for tid, px, py, nx, ny in team_pixel_players[t_id]:
                            d_px = np.hypot(px - current_ball_pos[0], py - current_ball_pos[1])
                            candidates.append((d_px, t_id, tid, nx, ny))
                    candidates.sort(key=lambda item: item[0])
                    best_d, best_team, best_tid, b_nx, b_ny = candidates[0]
                    att_id = best_team

                    if best_d < 75.0 and best_tid is not None:
                        if last_ball_owner is not None:
                            prev_team, prev_tid, prev_f = last_ball_owner
                            if prev_team == best_team and prev_tid != best_tid and (frame_idx - prev_f) >= int(fps * 0.4):
                                team_metrics[best_team]["passes"] += 1
                                if (frame_idx - last_event_frame["pass"]) >= int(fps * 4):
                                    flk = "Stânga" if b_ny < 0.33 else ("Centru" if b_ny < 0.67 else "Dreapta")
                                    timeline_events.append({
                                        "event_id": len(timeline_events) + 1,
                                        "event_type": ev_labels["pass"],
                                        "frame_index": frame_idx,
                                        "timestamp_sec": ts_sec,
                                        "attacking_team": resolved_names[best_team],
                                        "defending_team": resolved_names[1 - best_team],
                                        "flank": flk,
                                        "players_count": len(team_norm_x[0]) + len(team_norm_x[1]),
                                        "label": f"{ev_labels['pass']} ({ts_sec}s) — {resolved_names[best_team]} pe {flk}"
                                    })
                                    last_event_frame["pass"] = frame_idx
                        last_ball_owner = (best_team, best_tid, frame_idx)

                    if prev_ball_norm is not None and current_ball_norm is not None:
                        ball_speed = np.hypot(current_ball_norm[0] - prev_ball_norm[0], current_ball_norm[1] - prev_ball_norm[1])
                        in_goal_zone = (current_ball_norm[0] < 0.16 or current_ball_norm[0] > 0.84)
                        if ball_speed > 0.035 and in_goal_zone and (frame_idx - last_event_frame["shot"]) >= int(fps * 5):
                            team_metrics[att_id]["shots"] += 1
                            flk = "Stânga" if current_ball_norm[1] < 0.33 else ("Centru" if current_ball_norm[1] < 0.67 else "Dreapta")
                            timeline_events.append({
                                "event_id": len(timeline_events) + 1,
                                "event_type": ev_labels["shot"],
                                "frame_index": frame_idx,
                                "timestamp_sec": ts_sec,
                                "attacking_team": resolved_names[att_id],
                                "defending_team": resolved_names[1 - att_id],
                                "flank": flk,
                                "players_count": len(team_norm_x[0]) + len(team_norm_x[1]),
                                "label": f"🔥 {ev_labels['shot']} ({ts_sec}s) — {resolved_names[att_id]} ({flk})"
                            })
                            last_event_frame["shot"] = frame_idx

                elif team_norm_x[0] and team_norm_x[1]:
                    spread_0 = max(team_norm_x[0]) - min(team_norm_x[0])
                    spread_1 = max(team_norm_x[1]) - min(team_norm_x[1])
                    att_id = 0 if spread_0 >= spread_1 else 1

                prev_ball_norm = current_ball_norm

                if att_id is not None:
                    def_id = 1 - att_id
                    team_metrics[att_id]["attacking_frames"] += 1
                    frame_record["attacking_team"] = resolved_names[att_id]
                    frame_record["defending_team"] = resolved_names[def_id]

                ref_ny = current_ball_norm[1] if current_ball_norm is not None else (
                    float(np.mean(team_norm_y[att_id])) if (att_id is not None and team_norm_y[att_id]) else None
                )
                if ref_ny is not None:
                    if ref_ny < 0.33:
                        frame_record["attack_flank"] = "Stânga"
                    elif ref_ny < 0.67:
                        frame_record["attack_flank"] = "Centru"
                    else:
                        frame_record["attack_flank"] = "Dreapta"

                for t_id in (0, 1):
                    if team_norm_y[t_id]:
                        ty = float(np.mean(team_norm_y[t_id]))
                        if ty < 0.33:
                            team_metrics[t_id]["zone_counts"]["left"] += 1
                        elif ty < 0.67:
                            team_metrics[t_id]["zone_counts"]["center"] += 1
                        else:
                            team_metrics[t_id]["zone_counts"]["right"] += 1

                total_players_frame = len(team_norm_x[0]) + len(team_norm_x[1])
                if total_players_frame >= 4 and (frame_idx - last_event_frame["chance"]) >= int(fps * 4) and frame_record["attacking_team"]:
                    flank_str = frame_record["attack_flank"] or "Centru"
                    timeline_events.append({
                        "event_id": len(timeline_events) + 1,
                        "event_type": ev_labels["chance"],
                        "frame_index": frame_idx,
                        "timestamp_sec": ts_sec,
                        "attacking_team": frame_record["attacking_team"],
                        "defending_team": frame_record["defending_team"],
                        "flank": flank_str,
                        "players_count": total_players_frame,
                        "label": f"{ev_labels['chance']} ({ts_sec}s) — {frame_record['attacking_team']} pe {flank_str}"
                    })
                    last_event_frame["chance"] = frame_idx

                frozen_radar_players = frame_record["players"]
                frozen_radar_ball = current_ball_norm
                frozen_attacking_team = frame_record["attacking_team"]

                if frame_record["attacking_team"]:
                    phase_banner = f"[{sport.split(' ')[0].upper()}] ATAC: {frame_record['attacking_team']} ({frame_record['attack_flank'] or '-'})"
                    cv2.rectangle(annotated, (10, 10), (590, 42), (20, 20, 20), -1)
                    cv2.putText(annotated, phase_banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            else:
                last_positions.clear()
                prev_ball_norm = None
                cv2.rectangle(annotated, (10, 10), (460, 42), (30, 30, 140), -1)
                cv2.putText(annotated, f"CADRU NON-TACTIC: {cam_reason}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            radar_frame = render_live_radar_frame(
                base_pitch=base_pitch,
                players=frozen_radar_players,
                ball_norm_pos=frozen_radar_ball,
                team_colors_bgr=team_classifier.team_colors_bgr,
                team_names=resolved_names,
                attacking_team=frozen_attacking_team,
                is_tactical=is_tactical,
                pause_reason=cam_reason
            )
            combined_frame = np.vstack((annotated, radar_frame))
            out.write(combined_frame)

            tracking_records.append(frame_record)
            frame_idx += 1
            processed_count += 1
            if progress_callback and processed_count % 15 == 0:
                progress_callback(20 + int(((frame_idx - start_frame) / total_target_frames) * 70))

        cap.release()
        out.release()

        total_phase_frames = (team_metrics[0]["attacking_frames"] + team_metrics[1]["attacking_frames"]) or 1
        teams_summary = {}

        for t_id in (0, 1):
            tm = team_metrics[t_id]
            z_total = sum(tm["zone_counts"].values()) or 1
            top_p = sorted(tm["player_distances"].items(), key=lambda x: x[1], reverse=True)[:5]
            avg_line_pct = round(float(np.mean(tm["norm_x_positions"])) * 100.0, 1) if tm["norm_x_positions"] else 50.0
            bgr = team_classifier.team_colors_bgr.get(t_id, (200, 200, 200))
            hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

            teams_summary[str(t_id)] = {
                "name": tm["name"],
                "jersey_hex": hex_color,
                "bgr": list(bgr),
                "attacking_share_pct": round((tm["attacking_frames"] / total_phase_frames) * 100, 1),
                "avg_line_height_pct": avg_line_pct,
                "estimated_passes": tm["passes"],
                "estimated_shots": tm["shots"],
                "attacking_distribution_pct": {
                    "left": round((tm["zone_counts"]["left"] / z_total) * 100, 1),
                    "center": round((tm["zone_counts"]["center"] / z_total) * 100, 1),
                    "right": round((tm["zone_counts"]["right"] / z_total) * 100, 1)
                },
                "top_active_players": [{"track_id": pid, "movement_score": round(dist, 1)} for pid, dist in top_p]
            }

        summary_stats = {
            "sport": sport,
            "scoreboard_detected": scoreboard_detected_meta,
            "event_labels": ev_labels,
            "video_info": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": fps,
                "frame_stride": frame_stride,
                "start_sec": round(start_frame / fps, 1),
                "end_sec": round(frame_idx / fps, 1),
                "analyzed_seconds": round((frame_idx - start_frame) / fps, 1),
                "ball_detected_frames": ball_detections_count,
                "tactical_frames": tactical_frames_count,
                "filtered_closeup_frames": ignored_closeup_frames_count
            },
            "teams": teams_summary,
            "timeline_events": timeline_events,
            "frames": tracking_records
        }

        with open(metadata_path, "w") as f:
            json.dump(summary_stats, f, indent=2)

        return summary_stats


def rerender_swapped_video(input_path: str, session_dir: str) -> dict:
    """
    Inversează Echipa 0 cu Echipa 1 în tracking_data.json și re-randează instant
    annotated_video.mp4 + radarul 2D fără a rula din nou detecția YOLO.
    """
    metadata_path = os.path.join(session_dir, "tracking_data.json")
    output_video_path = os.path.join(session_dir, "annotated_video.mp4")

    with open(metadata_path, "r") as f:
        stats = json.load(f)

    sport = stats.get("sport", "Fotbal (Football)")
    v_info = stats["video_info"]
    fps = v_info["fps"]
    frame_stride = v_info.get("frame_stride", 1)

    t0 = stats["teams"]["0"]
    t1 = stats["teams"]["1"]
    stats["teams"]["0"], stats["teams"]["1"] = t1, t0

    new_names = (stats["teams"]["0"]["name"], stats["teams"]["1"]["name"])
    new_colors_bgr = {
        0: tuple(stats["teams"]["0"].get("bgr", (255, 140, 0))),
        1: tuple(stats["teams"]["1"].get("bgr", (0, 215, 255)))
    }

    old_n0, old_n1 = t0["name"], t1["name"]

    for ev in stats.get("timeline_events", []):
        if ev.get("attacking_team") == old_n0:
            ev["attacking_team"] = old_n1
            ev["defending_team"] = old_n0
        elif ev.get("attacking_team") == old_n1:
            ev["attacking_team"] = old_n0
            ev["defending_team"] = old_n1
        if "label" in ev:
            ev["label"] = (
                ev["label"]
                .replace(old_n0, "__TMP__")
                .replace(old_n1, old_n0)
                .replace("__TMP__", old_n1)
            )

    radar_w = v_info["width"]
    radar_h = max(230, int(v_info["height"] * 0.42))
    base_pitch = draw_2d_pitch_canvas(width=radar_w, height=radar_h, sport=sport)
    combined_h = v_info["height"] + radar_h

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_fps = fps / frame_stride
    out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (v_info["width"], combined_h))

    frozen_players = []
    frozen_ball = None
    frozen_att = None

    for rec in stats["frames"]:
        f_idx = rec["frame_index"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        is_tactical = rec.get("is_tactical", True)
        cam_reason = rec.get("camera_state", "TACTICAL_WIDE")

        if rec.get("attacking_team") == old_n0:
            rec["attacking_team"] = old_n1
            rec["defending_team"] = old_n0
        elif rec.get("attacking_team") == old_n1:
            rec["attacking_team"] = old_n0
            rec["defending_team"] = old_n1

        annotated = frame.copy()
        for p in rec["players"]:
            tid = p.get("team_id", -1)
            if tid in (0, 1):
                new_tid = 1 - tid
                p["team_id"] = new_tid
                p["team_name"] = new_names[new_tid]
                color_bgr = new_colors_bgr[new_tid]
            else:
                color_bgr = (160, 160, 160)

            x1, y1, x2, y2 = map(int, p["bbox"])
            lbl = f"#{p['track_id']} {p['team_name']}" if p.get("track_id") is not None else p["team_name"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(annotated, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

        b_pos = rec.get("ball_position")
        b_norm = tuple(rec["ball_norm"]) if rec.get("ball_norm") else None
        if b_pos:
            cv2.circle(annotated, (int(b_pos[0]), int(b_pos[1])), 10, (0, 230, 255), 2)
            cv2.putText(annotated, "BALL", (int(b_pos[0]) + 10, int(b_pos[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 2)

        if is_tactical:
            frozen_players = rec["players"]
            frozen_ball = b_norm
            frozen_att = rec.get("attacking_team")
            if rec.get("attacking_team"):
                phase_banner = f"[{sport.split(' ')[0].upper()}] ATAC: {rec['attacking_team']} ({rec.get('attack_flank') or '-'})"
                cv2.rectangle(annotated, (10, 10), (590, 42), (20, 20, 20), -1)
                cv2.putText(annotated, phase_banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (10, 10), (460, 42), (30, 30, 140), -1)
            cv2.putText(annotated, f"CADRU NON-TACTIC: {cam_reason}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        radar_frame = render_live_radar_frame(
            base_pitch=base_pitch,
            players=frozen_players,
            ball_norm_pos=frozen_ball,
            team_colors_bgr=new_colors_bgr,
            team_names=new_names,
            attacking_team=frozen_att,
            is_tactical=is_tactical,
            pause_reason=cam_reason
        )
        out.write(np.vstack((annotated, radar_frame)))

    cap.release()
    out.release()

    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats