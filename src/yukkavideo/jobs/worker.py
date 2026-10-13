"""
Background job worker: runs generation jobs in a separate process.

The FastAPI server enqueues jobs and supervises this worker; the worker owns
the GPU lock and processes queued/paused/failed jobs sequentially. Because
state lives in durable manifests, killing/restarting the worker (or the whole
Colab session) never loses completed segments.

Usage:
    python -m yukkavideo.jobs.worker --jobs-dir outputs/jobs [--once]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..models import ADAPTER_REGISTRY, detect_model_type
from .control import GPULock
from .manifest import JobManifest
from .runner import GenerationRunner, RunnerOptions
from .manifest import JobStore

logger = logging.getLogger(__name__)


def build_adapter(manifest: JobManifest, cache_dir: Optional[str] = None):
    """Instantiate the adapter recorded in the manifest."""
    model_name = manifest.model_name
    try:
        key = detect_model_type(model_name)
    except ValueError:
        mt = (manifest.model_type or "").lower()
        key = "cogvideo" if "cogvideo" in mt else "ltx" if "ltx" in mt \
            else "wan" if "wan" in mt else None
    if key is None or key not in ADAPTER_REGISTRY:
        raise RuntimeError(
            f"No adapter available for model '{model_name}' "
            f"(type '{manifest.model_type}')."
        )
    return ADAPTER_REGISTRY[key](model_name=model_name, cache_dir=cache_dir)


def runner_from_manifest(manifest: JobManifest, store: JobStore,
                         cache_dir: Optional[str] = None) -> GenerationRunner:
    adapter = build_adapter(manifest, cache_dir=cache_dir)
    opts_dict = {k: v for k, v in (manifest.options or {}).items()
                 if k in RunnerOptions.__dataclass_fields__}
    options = RunnerOptions(**opts_dict)
    return GenerationRunner(adapter, store, options=options)


def process_pending(store: JobStore, cache_dir: Optional[str] = None,
                    max_jobs: int = 1) -> int:
    """Run up to ``max_jobs`` unfinished jobs oldest-first. Returns count run."""
    ran = 0
    for job in store.unfinished_jobs():
        if ran >= max_jobs:
            break
        if job.status == "queued" and not any(
            s.status != "done" and False for s in job.segments
        ):
            pass  # fallthrough: process every unfinished job
        logger.info("Worker: starting job %s (status=%s, progress=%.0f%%)",
                    job.job_id, job.status, job.progress * 100)
        try:
            runner = runner_from_manifest(job, store, cache_dir=cache_dir)
            if not runner.adapter.is_loaded:
                runner.adapter.load_model()
            result = runner.run(job.job_id)
            ran += 1
            logger.info("Worker: job %s -> %s (%.0f%%)",
                        job.job_id, result.status, result.progress * 100)
        except Exception as exc:  # noqa: BLE001 - keep worker alive
            ran += 1
            logger.exception("Worker: job %s crashed: %s", job.job_id, exc)
            try:
                m = store.load(job.job_id)
                if m.status not in ("completed", "cancelled"):
                    m.set_status("failed")
                    m.errors.append(f"Worker crash: {type(exc).__name__}: {exc}")
                    store.save(m)
            except Exception:
                pass
    return ran


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="yukkavideo-worker")
    parser.add_argument("--jobs-dir", default="outputs/jobs")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--once", action="store_true",
                        help="Process currently pending jobs then exit.")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = JobStore(Path(args.jobs_dir))
    if args.once:
        n = process_pending(store, args.cache_dir)
        print(f"Processed {n} job(s).")
        return 0
    import time
    while True:
        try:
            process_pending(store, args.cache_dir)
        except KeyboardInterrupt:
            return 0
        except Exception:  # noqa: BLE001
            logger.exception("Worker loop error")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
