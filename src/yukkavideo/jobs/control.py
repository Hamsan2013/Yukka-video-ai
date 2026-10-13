"""
Job control: pause / resume / cancel signalling and a single-worker GPU lock.

Control is file-based so an external process (CLI or API in another request)
can signal a running worker, and so state survives even if the signalling
process dies. The worker checks for control files at segment boundaries —
this gives safe checkpointing without killing mid-inference.
"""

import json
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from .manifest import JobManifest, JobStore


class ControlSignal(str, Enum):
    NONE = "none"
    PAUSE = "pause"
    CANCEL = "cancel"


class JobController:
    """Coordinates control signals and manifest status for one job."""

    def __init__(self, store: JobStore, job_id: str):
        self.store = store
        self.job_id = job_id
        self._dir = store.job_dir(job_id)

    # ------------------------------------------------------------------
    def _signal_path(self, name: str) -> Path:
        return self._dir / f"{name}.signal"

    def request_pause(self) -> None:
        self._signal_path("pause").write_text(json.dumps({"requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))

    def request_cancel(self) -> None:
        self._signal_path("cancel").write_text(json.dumps({"requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))

    def clear_signals(self) -> None:
        for sig in ("pause", "cancel"):
            p = self._signal_path(sig)
            if p.exists():
                p.unlink()

    def pending_signal(self) -> ControlSignal:
        # Cancel wins over pause.
        if self._signal_path("cancel").exists():
            return ControlSignal.CANCEL
        if self._signal_path("pause").exists():
            return ControlSignal.PAUSE
        return ControlSignal.NONE

    # ------------------------------------------------------------------
    def check(self, manifest: JobManifest) -> ControlSignal:
        """Inspect signals; caller decides how to react at safe points."""
        return self.pending_signal()

    def apply_pause(self, manifest: JobManifest) -> None:
        manifest.set_status("paused")
        self.store.save(manifest)
        self.clear_signals()

    def apply_cancel(self, manifest: JobManifest) -> None:
        manifest.set_status("cancelled")
        self.store.save(manifest)
        self.clear_signals()

    def prepare_resume(self, manifest: JobManifest) -> JobManifest:
        """Validate a job can be resumed and reset transient state."""
        if manifest.status == "completed":
            raise RuntimeError(f"Job {self.job_id} is already completed.")
        if manifest.status == "cancelled":
            raise RuntimeError(
                f"Job {self.job_id} was cancelled; start a new job instead."
            )
        # A 'running' manifest on restart means the previous worker died —
        # it is safe to take over because of the worker lock below.
        self.clear_signals()
        # Segments marked 'running' were interrupted mid-generation; revert
        # them to pending so they regenerate (their outputs are validated).
        for seg in manifest.segments:
            if seg.status == "running":
                seg.status = "pending"
        manifest.set_status("running")
        self.store.save(manifest)
        return manifest


class GPULock:
    """Process-local lock preventing concurrent GPU-heavy jobs.

    For multi-process deployments use the file lock variant via
    :meth:`acquire_file_lock`.
    """

    def __init__(self, max_concurrent: int = 1):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))

    def acquire(self, timeout: Optional[float] = None) -> bool:
        return self._sem.acquire(timeout=timeout if timeout is not None else -1) \
            if timeout is not None else self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    @staticmethod
    def acquire_file_lock(lock_path: Path, timeout: float = 0.0) -> Optional[Path]:
        """Best-effort cross-process lock using O_EXCL file creation.

        Returns the lock path on success, None if unavailable. Caller must
        delete the file when done. Stale locks older than ``stale_after``
        are reclaimed.
        """
        stale_after = 24 * 3600  # 24h safety valve
        deadline = time.time() + timeout
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = open(lock_path, "x")
                fd.write(json.dumps({"pid": __import__("os").getpid(),
                                     "at": time.time()}))
                fd.close()
                return lock_path
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > stale_after:
                        lock_path.unlink()
                        continue
                except OSError:
                    pass
                if time.time() >= deadline:
                    return None
                time.sleep(0.5)

    @staticmethod
    def release_file_lock(lock_path: Path) -> None:
        try:
            lock_path.unlink()
        except OSError:
            pass
