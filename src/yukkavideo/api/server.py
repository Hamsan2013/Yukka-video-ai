"""
FastAPI Server for Yukka Video AI

Provides REST API endpoints for video generation.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import uuid
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""
    prompt: str = Field(..., description="Text description of the video")
    negative_prompt: Optional[str] = Field(None, description="What to avoid")
    num_inference_steps: int = Field(50, ge=1, le=200)
    height: int = Field(480, ge=64, le=1024)
    width: int = Field(720, ge=64, le=1024)
    fps: int = Field(8, ge=1, le=30)
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0)
    seed: Optional[int] = Field(42, description="Random seed")


class VideoGenerationResponse(BaseModel):
    """Response model for video generation."""
    task_id: str
    status: str
    message: str
    video_path: Optional[str] = None


class TaskStatus(BaseModel):
    """Task status response."""
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    video_path: Optional[str] = None


class JobCreateRequest(BaseModel):
    """Request model for durable sequential long-video jobs."""
    prompt: str = Field(..., description="Video description")
    duration_seconds: float = Field(10.0, gt=0,
                                    description="Total duration (no hard max)")
    fps: int = Field(8, ge=1, le=60)
    height: int = Field(480, ge=64, le=4096)
    width: int = Field(720, ge=64, le=4096)
    model: str = Field("THUDM/CogVideoX-2b")
    segment_seconds: float = Field(5.0, gt=0)
    num_inference_steps: int = Field(50, ge=1, le=200)
    guidance_scale: float = Field(6.0, ge=0.0, le=30.0)
    negative_prompt: Optional[str] = None
    seed: Optional[int] = 42
    style: Optional[str] = None
    start_image: Optional[str] = Field(
        None, description="Optional first-frame path; used only if the "
                          "selected model supports image-to-video")
    transition_seconds: float = Field(0.0, ge=0.0, le=2.0)


# Global state for tasks
_tasks: Dict[str, Dict[str, Any]] = {}
_pipeline = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Yukka Video AI API",
        description="REST API for text-to-video generation",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup_event():
        """Initialize the video pipeline on startup."""
        global _pipeline
        try:
            from ..inference.pipeline import YukkaVideoPipeline
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            _pipeline = YukkaVideoPipeline.from_pretrained(
                model_name="THUDM/CogVideoX-2b",
                device=device,
                dtype=dtype,
            )

            if device == "cuda":
                _pipeline.enable_model_cpu_offload()
                _pipeline.enable_vae_slicing()

            print("Pipeline initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize pipeline: {e}")

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Yukka Video AI API",
            "version": "0.1.0",
            "endpoints": [
                "POST /generate - Generate a video",
                "GET /status/{task_id} - Get task status",
                "GET /video/{task_id} - Download generated video",
                "GET /health - Health check",
            ],
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint (never loads models as a side effect)."""
        return {
            "status": "healthy",
            "pipeline_loaded": _pipeline is not None,
            "shared_adapter_loaded": _shared_adapter is not None,
        }

    @app.post("/generate", response_model=VideoGenerationResponse)
    async def generate_video(
        request: VideoGenerationRequest,
        background_tasks: BackgroundTasks,
    ):
        """
        Generate a video from a text prompt.

        This endpoint starts a background task for video generation
        and returns immediately with a task ID.
        """
        task_id = str(uuid.uuid4())

        _tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Task queued",
            "video_path": None,
            "request": request.dict(),
        }

        background_tasks.add_task(process_generation_task, task_id, request)

        return VideoGenerationResponse(
            task_id=task_id,
            status="pending",
            message="Video generation task started",
        )

    @app.get("/status/{task_id}", response_model=TaskStatus)
    async def get_status(task_id: str):
        """Get the status of a video generation task."""
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = _tasks[task_id]
        return TaskStatus(
            task_id=task_id,
            status=task["status"],
            progress=task.get("progress"),
            message=task.get("message"),
            video_path=task.get("video_path"),
        )

    @app.get("/video/{task_id}")
    async def get_video(task_id: str):
        """Download the generated video file."""
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = _tasks[task_id]
        if task["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Video not ready yet",
            )

        video_path = task.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=Path(video_path).name,
        )

    # ------------------------------------------------------------------
    # Durable long-video jobs (sequential generation controller)
    #
    # These endpoints use the SAME GenerationRunner / JobStore used by the
    # CLI and the Colab notebook. Jobs survive server restarts because all
    # state lives in outputs/jobs/<job_id>/manifest.json. The server only
    # enqueues; a worker process (yukkavideo worker / python -m
    # yukkavideo.jobs.worker) performs GPU work under a file-based lock so
    # only one job uses the GPU at a time.
    # ------------------------------------------------------------------
    def _jobs_dir() -> Path:
        return Path(os.environ.get("YUKKA_JOBS_DIR", "outputs/jobs"))

    @app.post("/jobs")
    async def create_job(req: JobCreateRequest,
                         background_tasks: BackgroundTasks,
                         run_now: bool = False):
        """Create a sequential long-video job.

        By default the job is only *created* (status 'queued') — no model
        weights are downloaded and nothing is loaded into memory here. A
        worker process (``yukkavideo worker``) or an explicit
        ``run_now=true`` background task then executes it.
        """
        from ..jobs.manifest import JobStore
        from ..jobs.runner import GenerationRunner, RunnerOptions
        from ..models import ADAPTER_REGISTRY, detect_model_type

        store = JobStore(_jobs_dir())
        try:
            key = detect_model_type(req.model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        # Adapter is constructed WITHOUT loading weights (no downloads).
        adapter = ADAPTER_REGISTRY[key](model_name=req.model)
        options = RunnerOptions(
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            negative_prompt=req.negative_prompt or "",
            style=req.style or "",
            transition_seconds=req.transition_seconds,
        )
        runner = GenerationRunner(adapter, store, options=options)
        try:
            manifest = runner.create_job(
                prompt=req.prompt,
                duration_seconds=req.duration_seconds,
                fps=req.fps,
                height=req.height,
                width=req.width,
                seed=req.seed,
                start_image=req.start_image,
                segment_seconds=req.segment_seconds,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if run_now:
            background_tasks.add_task(_run_job_in_process, manifest.job_id)
        return {"job_id": manifest.job_id, "status": manifest.status,
                "segments": len(manifest.segments),
                "total_frames": manifest.total_frames}

    @app.get("/jobs")
    async def list_jobs():
        from ..jobs.manifest import JobStore
        store = JobStore(_jobs_dir())
        return [
            {"job_id": m.job_id, "status": m.status, "progress": m.progress,
             "prompt": m.prompt[:80], "final_video_path": m.final_video_path}
            for m in store.list_jobs()
        ]

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        from ..jobs.manifest import JobStore
        store = JobStore(_jobs_dir())
        try:
            m = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found")
        return m.to_dict()

    @app.post("/jobs/{job_id}/pause")
    async def pause_job(job_id: str):
        from ..jobs.control import JobController
        from ..jobs.manifest import JobStore
        store = JobStore(_jobs_dir())
        try:
            store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found")
        JobController(store, job_id).request_pause()
        return {"job_id": job_id, "pause_requested": True}

    @app.post("/jobs/{job_id}/resume")
    async def resume_job(job_id: str, background_tasks: BackgroundTasks):
        """Ask the worker loop to run/resume this job now (in-process).

        For robustness across restarts prefer running a separate worker
        process; this endpoint is a convenience for single-machine setups.
        """
        from ..jobs.control import JobController
        from ..jobs.manifest import JobStore
        store = JobStore(_jobs_dir())
        try:
            m = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found")
        JobController(store, job_id).clear_signals()
        if m.status == "completed":
            return {"job_id": job_id, "status": "completed"}
        background_tasks.add_task(_run_job_in_process, job_id)
        return {"job_id": job_id, "status": "resuming"}

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        from ..jobs.control import JobController
        from ..jobs.manifest import JobStore
        store = JobStore(_jobs_dir())
        try:
            store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found")
        JobController(store, job_id).request_cancel()
        return {"job_id": job_id, "cancel_requested": True}

    @app.get("/jobs/{job_id}/video")
    async def job_video(job_id: str):
        from ..jobs.manifest import JobStore
        store = JobStore(_jobs_dir())
        try:
            m = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found")
        if not m.final_video_path or not Path(m.final_video_path).exists():
            raise HTTPException(status_code=409,
                                detail="Final video not assembled yet")
        return FileResponse(m.final_video_path, media_type="video/mp4",
                            filename=Path(m.final_video_path).name)

    return app


# ----------------------------------------------------------------------
# Server-side job helpers (lazy shared adapter; never loads on import)
# ----------------------------------------------------------------------
_shared_adapter = None
_shared_adapter_name = None


def _get_shared_adapter(model_name: str):
    """Instantiate + load an adapter once per process, reusing it across jobs."""
    global _shared_adapter, _shared_adapter_name
    if _shared_adapter is not None and _shared_adapter_name == model_name:
        return _shared_adapter
    import torch
    from ..models import ADAPTER_REGISTRY, detect_model_type

    key = detect_model_type(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    adapter = ADAPTER_REGISTRY[key](model_name=model_name, device=device,
                                    dtype=dtype)
    if device == "cuda":
        adapter.enable_cpu_offload()
    adapter.load_model()
    _shared_adapter, _shared_adapter_name = adapter, model_name
    return adapter


def _run_job_in_process(job_id: str) -> None:
    """Resume a job inside the API process (guarded by the GPU file lock)."""
    import os as _os
    from pathlib import Path as _P
    from ..jobs.manifest import JobStore
    from ..jobs.worker import runner_from_manifest

    store = JobStore(_P(_os.environ.get("YUKKA_JOBS_DIR", "outputs/jobs")))
    manifest = store.load(job_id)
    runner = runner_from_manifest(manifest, store)
    runner.run(job_id)


def process_generation_task(task_id: str, request: VideoGenerationRequest):
    """Process a video generation task in the background."""
    global _pipeline

    if task_id not in _tasks:
        return

    _tasks[task_id]["status"] = "processing"
    _tasks[task_id]["progress"] = 0.1
    _tasks[task_id]["message"] = "Loading pipeline..."

    try:
        if _pipeline is None:
            from ..inference.pipeline import YukkaVideoPipeline
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            _pipeline = YukkaVideoPipeline.from_pretrained(
                model_name="THUDM/CogVideoX-2b",
                device=device,
                dtype=dtype,
            )

            if device == "cuda":
                _pipeline.enable_model_cpu_offload()
                _pipeline.enable_vae_slicing()

        _tasks[task_id]["progress"] = 0.3
        _tasks[task_id]["message"] = "Generating video frames..."

        # Generate video
        output = _pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width,
            fps=request.fps,
            guidance_scale=request.guidance_scale,
        )

        _tasks[task_id]["progress"] = 0.8
        _tasks[task_id]["message"] = "Exporting to MP4..."

        # Export video
        output_dir = Path("./outputs/api")
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / f"{task_id}.mp4"
        _pipeline.export_to_video(
            frames=output.frames,
            output_path=video_path,
            fps=request.fps,
        )

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 1.0
        _tasks[task_id]["message"] = "Video generation completed"
        _tasks[task_id]["video_path"] = str(video_path)

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["message"] = f"Error: {str(e)}"
        print(f"Task {task_id} failed: {e}")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
