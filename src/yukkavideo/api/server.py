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
        """Health check endpoint."""
        return {
            "status": "healthy",
            "pipeline_loaded": _pipeline is not None,
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

    return app


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
