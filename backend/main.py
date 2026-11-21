import os
import uuid
import json
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend.pipeline import VideoAnalysisPipeline
from backend.analytics import generate_visual_analytics, extract_10s_clip
from backend.llm_service import generate_tactical_report
from backend.youtube_service import download_youtube_video
from backend.benchmark import run_full_benchmark, BENCHMARK_DIR

load_dotenv()

app = FastAPI(title="AI Sports Tactical Analysis API")
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")
app.mount("/benchmarks", StaticFiles(directory=BENCHMARK_DIR), name="benchmarks")

jobs_status = {}
benchmark_jobs = {}
pipeline = VideoAnalysisPipeline(model_name="yolov8n.pt")


def run_analysis_job(
    session_id: str,
    input_path: str,
    session_dir: str,
    team_a: str,
    team_b: str,
    youtube_url: Optional[str],
    video_title: str,
    start_sec: float,
    duration_sec: Optional[float]
):
    try:
        if youtube_url:
            jobs_status[session_id] = {"status": "downloading_youtube", "progress": 3}
            video_title = download_youtube_video(youtube_url, input_path)

        jobs_status[session_id] = {"status": "processing", "progress": 8}

        def update_progress(pct: int):
            jobs_status[session_id]["progress"] = pct

        stats = pipeline.process_video(
            input_path,
            session_dir,
            team_names=(team_a, team_b),
            start_sec=start_sec,
            duration_sec=duration_sec,
            progress_callback=update_progress
        )
        jobs_status[session_id]["progress"] = 90

        visuals = generate_visual_analytics(stats, input_path, session_dir)
        jobs_status[session_id]["progress"] = 95

        report = generate_tactical_report(stats, visuals["key_frames"], session_dir)

        result_payload = {
            "session_id": session_id,
            "video_title": video_title,
            "stats": {k: v for k, v in stats.items() if k != "frames"},
            "visuals": {
                "video_url": f"/files/{session_id}/annotated_video.mp4",
                "heatmap_team_0_url": f"/files/{session_id}/{visuals['heatmap_team_0']}",
                "heatmap_team_1_url": f"/files/{session_id}/{visuals['heatmap_team_1']}",
                "timeline_url": f"/files/{session_id}/timeline.png",
                "key_frames": [
                    {
                        "url": f"/files/{session_id}/{kf['filename']}",
                        "clip_url": f"/files/{session_id}/{kf['clip_filename']}" if kf.get("clip_filename") else None,
                        "timestamp_sec": kf["timestamp_sec"],
                        "attacking_team": kf["attacking_team"],
                        "flank": kf.get("flank", "Centru"),
                        "players_count": kf["players_count"]
                    } for kf in visuals["key_frames"]
                ]
            },
            "tactical_report": report
        }

        with open(os.path.join(session_dir, "final_result.json"), "w") as f:
            json.dump(result_payload, f, indent=2)

        jobs_status[session_id] = {"status": "completed", "progress": 100}
    except Exception as e:
        jobs_status[session_id] = {"status": "failed", "error": str(e)}


def execute_benchmark_task(bench_id: str, video_path: str, models: list[str], devices: list[str], max_frames: int):
    try:
        benchmark_jobs[bench_id] = {"status": "running", "progress": 5, "message": "Inițializare benchmark..."}

        def on_prog(pct: int, msg: str):
            benchmark_jobs[bench_id] = {"status": "running", "progress": pct, "message": msg}

        output = run_full_benchmark(video_path, models=models, devices=devices, max_frames=max_frames, progress_callback=on_prog)
        benchmark_jobs[bench_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Evaluare finalizată!",
            "results": output["results"],
            "urls": {
                "csv": "/benchmarks/benchmark_results.csv",
                "perf_chart": "/benchmarks/latency_fps_comparison.png",
                "acc_chart": "/benchmarks/detection_stability_comparison.png"
            }
        }
    except Exception as e:
        benchmark_jobs[bench_id] = {"status": "failed", "error": str(e)}


@app.post("/api/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    team_a: str = Form("Team A"),
    team_b: str = Form("Team B"),
    start_sec: float = Form(0.0),
    duration_sec: float = Form(0.0)
):
    if not file and not (youtube_url and youtube_url.strip()):
        raise HTTPException(status_code=400, detail="Încarcă un fișier video sau introdu un link YouTube.")

    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "input.mp4")
    video_title = "Uploaded Clip"

    if file and file.filename:
        video_title = file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        clean_url = None
    else:
        clean_url = youtube_url.strip()

    dur = duration_sec if duration_sec > 0 else None

    jobs_status[session_id] = {"status": "queued", "progress": 0}
    background_tasks.add_task(
        run_analysis_job,
        session_id,
        input_path,
        session_dir,
        team_a.strip() or "Team A",
        team_b.strip() or "Team B",
        clean_url,
        video_title,
        max(0.0, start_sec),
        dur
    )

    return {"session_id": session_id, "message": "Analysis started"}


@app.get("/api/sessions")
def list_saved_sessions():
    """Returnează toate analizele salvate în storage/ pentru reîncărcare instantă."""
    sessions = []
    if not os.path.exists(STORAGE_DIR):
        return sessions
    for name in sorted(os.listdir(STORAGE_DIR), reverse=True):
        res_file = os.path.join(STORAGE_DIR, name, "final_result.json")
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    d = json.load(f)
                t0 = d["stats"]["teams"]["0"]["name"]
                t1 = d["stats"]["teams"]["1"]["name"]
                sessions.append({
                    "session_id": name,
                    "label": f"{d.get('video_title', 'Clip')} — {t0} vs {t1} ({name})"
                })
            except Exception:
                pass
    return sessions


@app.post("/api/extract_phase/{session_id}")
def extract_custom_phase_clip(session_id: str, timestamp_sec: float = Form(...)):
    """
    Decupează la cerere o fază de 10 secunde (5s înainte + 5s după timestamp_sec)
    din videoclipul adnotat (cu tot cu radarul 2D live).
    """
    session_dir = os.path.join(STORAGE_DIR, session_id)
    annotated_video = os.path.join(session_dir, "annotated_video.mp4")
    result_file = os.path.join(session_dir, "final_result.json")

    if not os.path.exists(annotated_video) or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    with open(result_file, "r") as f:
        res_data = json.load(f)

    start_offset = res_data["stats"]["video_info"].get("start_sec", 0.0)
    rel_sec = max(0.0, timestamp_sec - start_offset)

    clip_name = f"phase_{int(timestamp_sec * 10)}.mp4"
    clip_path = os.path.join(session_dir, clip_name)

    if not os.path.exists(clip_path):
        ok = extract_10s_clip(annotated_video, rel_sec, clip_path, window_half=5.0)
        if not ok:
            raise HTTPException(status_code=400, detail="Nu s-a putut extrage secvența video.")

    return {
        "clip_url": f"/files/{session_id}/{clip_name}",
        "window_start": round(max(start_offset, timestamp_sec - 5.0), 1),
        "window_end": round(timestamp_sec + 5.0, 1)
    }


@app.get("/api/status/{session_id}")
def get_status(session_id: str):
    if session_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Session not found")
    return jobs_status[session_id]


@app.get("/api/results/{session_id}")
def get_results(session_id: str):
    result_file = os.path.join(STORAGE_DIR, session_id, "final_result.json")
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Results not ready")
    with open(result_file, "r") as f:
        return json.load(f)


@app.post("/api/benchmark")
async def start_benchmark(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    models: str = Form("yolov8n.pt,yolov8s.pt"),
    devices: str = Form("mps,cpu"),
    max_frames: int = Form(150)
):
    bench_id = str(uuid.uuid4())[:8]
    video_path = None

    if file and file.filename:
        temp_dir = os.path.join(STORAGE_DIR, f"bench_{bench_id}")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "bench_input.mp4")
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    elif session_id:
        candidate = os.path.join(STORAGE_DIR, session_id.strip(), "input.mp4")
        if os.path.exists(candidate):
            video_path = candidate

    if not video_path:
        raise HTTPException(status_code=400, detail="Încarcă un videoclip sau rulează mai întâi o analiză de meci.")

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    device_list = [d.strip().lower() for d in devices.split(",") if d.strip()]

    benchmark_jobs[bench_id] = {"status": "queued", "progress": 0, "message": "În așteptare..."}
    background_tasks.add_task(execute_benchmark_task, bench_id, video_path, model_list, device_list, max_frames)
    return {"bench_id": bench_id}


@app.get("/api/benchmark/status/{bench_id}")
def get_benchmark_status(bench_id: str):
    if bench_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Benchmark job not found")
    return benchmark_jobs[bench_id]