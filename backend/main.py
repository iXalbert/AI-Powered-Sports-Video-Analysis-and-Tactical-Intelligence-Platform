import os
import uuid
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend.pipeline import VideoAnalysisPipeline
from backend.analytics import generate_visual_analytics
from backend.llm_service import generate_tactical_report

load_dotenv()

app = FastAPI(title="AI Sports Tactical Analysis API")
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

jobs_status = {}
pipeline = VideoAnalysisPipeline(model_name="yolov8n.pt")


def run_analysis_job(session_id: str, input_path: str, session_dir: str, team_a: str, team_b: str):
    try:
        jobs_status[session_id]["status"] = "processing"

        def update_progress(pct: int):
            jobs_status[session_id]["progress"] = pct

        stats = pipeline.process_video(
            input_path,
            session_dir,
            team_names=(team_a, team_b),
            max_minutes=5,
            progress_callback=update_progress
        )
        jobs_status[session_id]["progress"] = 90

        visuals = generate_visual_analytics(stats, input_path, session_dir)
        jobs_status[session_id]["progress"] = 95

        report = generate_tactical_report(stats, visuals["key_frames"], session_dir)

        result_payload = {
            "session_id": session_id,
            "stats": {k: v for k, v in stats.items() if k != "frames"},
            "visuals": {
                "video_url": f"/files/{session_id}/annotated_video.mp4",
                "heatmap_team_0_url": f"/files/{session_id}/{visuals['heatmap_team_0']}",
                "heatmap_team_1_url": f"/files/{session_id}/{visuals['heatmap_team_1']}",
                "timeline_url": f"/files/{session_id}/timeline.png",
                "key_frames": [
                    {
                        "url": f"/files/{session_id}/{kf['filename']}",
                        "timestamp_sec": kf["timestamp_sec"],
                        "attacking_team": kf["attacking_team"],
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


@app.post("/api/analyze")
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    team_a: str = Form("Team A"),
    team_b: str = Form("Team B")
):
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "input.mp4")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs_status[session_id] = {"status": "queued", "progress": 0}
    background_tasks.add_task(run_analysis_job, session_id, input_path, session_dir, team_a.strip() or "Team A", team_b.strip() or "Team B")

    return {"session_id": session_id, "message": "Analysis started"}


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