"""
Durable job manifests for long video generation.

A JobManifest is the single source of truth for a generation job, stored on
disk as JSON with atomic writes (tmp file + os.replace) so an interrupted
process cannot leave a half-written manifest behind.

Job states: queued -> running -> (paused | completed | failed | cancelled)
Resume moves paused/failed/queued back to running; cancel/pause are honoured
by the worker at segment boundaries.
"""

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"queued", "running", "paused", "completed", "failed", "cancelled"}

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class InvalidTransitionError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class SegmentRecord:
    index: int
    status: str = "pending"  # pending | running | done | failed
    requested_frames: int = 0
    generated_frames: int = 0
    output_path: Optional[str] = None
    reference_frame_path: Optional[str] = None
    prompt: str = ""
    seed: Optional[int] = None
    model_name: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobManifest:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    prompt: str
    duration_seconds: float
    fps: float
    height: int
    width: int
    output_format: str = "mp4"
    style: str = ""
    model_name: str = ""
    model_type: str = ""
    segment_seconds: float = 5.0
    total_frames: int = 0
    seed: Optional[int] = None
    start_image: Optional[str] = None
    character_references: List[str] = field(default_factory=list)
    scene_plan: Dict[str, Any] = field(default_factory=dict)
    segments: List[SegmentRecord] = field(default_factory=list)
    final_video_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    progress: float = 0.0
    # Free-form job options; must remain JSON-serializable.
    # Never store secrets (API keys/tokens) here.
    options: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------
    def set_status(self, status: str) -> None:
        if status not in VALID_STATUSES:
            raise InvalidTransitionError(f"Unknown status: {status}")
        if self.status in ("completed", "cancelled") and status != self.status:
            raise InvalidTransitionError(
                f"Cannot move job from '{self.status}' to '{status}': terminal state."
            )
        self.status = status
        self.updated_at = _now()

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def recompute_progress(self) -> None:
        if not self.segments:
            self.progress = 1.0 if self.status == "completed" else 0.0
            return
        done = sum(1 for s in self.segments if s.status == "done")
        self.progress = done / len(self.segments)

    @property
    def next_incomplete_segment(self) -> Optional[SegmentRecord]:
        for s in self.segments:
            if s.status != "done":
                return s
        return None

    @property
    def last_done_reference(self) -> Optional[str]:
        ref = None
        for s in self.segments:
            if s.status == "done" and s.reference_frame_path:
                ref = s.reference_frame_path
        return ref

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobManifest":
        segs = [
            s if isinstance(s, SegmentRecord) else SegmentRecord(**s)
            for s in d.get("segments", [])
        ]
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in d.items() if k in known}
        clean["segments"] = segs
        return cls(**clean)


class JobStore:
    """Filesystem-backed store for job manifests."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # paths -------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / self.MANIFEST_NAME

    def segments_dir(self, job_id: str) -> Path:
        p = self.job_dir(job_id) / "segments"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def refs_dir(self, job_id: str) -> Path:
        p = self.job_dir(job_id) / "references"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # create ------------------------------------------------------------
    @staticmethod
    def new_job_id() -> str:
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def create(self, manifest: JobManifest) -> JobManifest:
        self.job_dir(manifest.job_id).mkdir(parents=True, exist_ok=True)
        self.save(manifest)
        return manifest

    # save (atomic) -----------------------------------------------------
    def save(self, manifest: JobManifest) -> None:
        manifest.updated_at = _now()
        path = self.manifest_path(manifest.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(manifest.to_dict(), indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # atomic on POSIX & Windows
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # load ---------------------------------------------------------------
    def load(self, job_id: str) -> JobManifest:
        path = self.manifest_path(job_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No job manifest for job id '{job_id}' at {path}"
            )
        return JobManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, job_id: str) -> bool:
        return self.manifest_path(job_id).exists()

    # listing / recovery --------------------------------------------------
    def list_jobs(self) -> List[JobManifest]:
        jobs: List[JobManifest] = []
        if not self.root.exists():
            return jobs
        for child in sorted(self.root.iterdir()):
            mp = child / self.MANIFEST_NAME
            if child.is_dir() and mp.exists():
                try:
                    jobs.append(
                        JobManifest.from_dict(json.loads(mp.read_text(encoding="utf-8")))
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # skip corrupt manifests rather than crash listing
        return jobs

    def unfinished_jobs(self) -> List[JobManifest]:
        return [j for j in self.list_jobs() if j.status not in TERMINAL_STATUSES]
