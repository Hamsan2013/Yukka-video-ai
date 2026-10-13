"""
Sequential long-video generation controller.

Orchestrates: scene planning -> segment frame-budget planning -> per-segment
generation (T2V or genuine I2V continuation) -> clip validation -> final-frame
reference extraction -> manifest checkpointing -> assembly.

The loop is resumable: completed segments are never regenerated, and pause /
cancel signals are honoured at segment boundaries. No hard-coded maximum
duration — the frame budget comes from the user's request.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..continuity.frames import (
    ClipValidationError,
    extract_final_frame,
    validate_clip_file,
)
from ..inference.planning import build_plan
from ..inference.scene_planner import create_scene_plan
from ..inference.video_exporter import VideoExporter
from ..models.base import BaseModelAdapter
from .control import ControlSignal, GPULock, JobController
from .manifest import JobManifest, JobStore, SegmentRecord

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class RunnerOptions:
    """Generation options for a job (JSON-serializable; never store secrets)."""

    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    negative_prompt: str = ""
    use_reference_continuation: bool = True   # only used if model supports I2V
    max_segment_retries: int = 1
    retry_on_oom_downscale: bool = False      # opt-in automatic smaller retry
    transition_seconds: float = 0.0           # optional crossfade at assembly
    cleanup_intermediates: bool = False       # opt-in destructive cleanup
    style: str = ""
    characters: List[str] = field(default_factory=list)
    location: str = ""
    use_llm_planner: bool = False             # never required


class OutOfMemoryError(RuntimeError):
    """Raised when CUDA OOM is detected and no configured remedy applies."""


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text


class GenerationRunner:
    """Real sequential clip-generation controller bound to one adapter."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        store: JobStore,
        options: Optional[RunnerOptions] = None,
        gpu_lock: Optional[GPULock] = None,
        progress_callback: Optional[Callable[[JobManifest], None]] = None,
    ):
        self.adapter = adapter
        self.store = store
        self.options = options or RunnerOptions()
        self.gpu_lock = gpu_lock or GPULock(max_concurrent=1)
        self.progress_callback = progress_callback
        self.exporter = VideoExporter()

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------
    def create_job(
        self,
        prompt: str,
        duration_seconds: float,
        fps: float,
        height: int,
        width: int,
        output_format: str = "mp4",
        seed: Optional[int] = 42,
        start_image: Optional[Union[str, Path]] = None,
        character_references: Optional[List[str]] = None,
        segment_seconds: float = 5.0,
        model_name: Optional[str] = None,
    ) -> JobManifest:
        caps = self.adapter.get_capabilities()
        if not caps.supports_dimensions(height, width):
            raise ValueError(
                f"Model '{self.adapter.model_name}' cannot produce {height}x{width}. "
                f"Constraints: divisor={caps.dimension_divisor}, "
                f"heights={caps.supported_heights}, widths={caps.supported_widths}."
            )
        if start_image and not Path(start_image).exists():
            raise FileNotFoundError(f"Start image not found: {start_image}")
        if start_image and not caps.supports_image_to_video:
            logger.warning(
                "Start image provided but %s does not support image-to-video; "
                "the first segment will be generated from text only.",
                self.adapter.model_name,
            )

        plan = build_plan(duration_seconds, fps, segment_seconds,
                          adjust_num_frames=self.adapter.adjust_num_frames)
        scene_plan = create_scene_plan(
            prompt=prompt,
            num_scenes=max(1, len(plan.segments)),
            total_duration=duration_seconds,
            style=self.options.style,
            characters=self.options.characters,
            location=self.options.location,
            use_llm=self.options.use_llm_planner,
        )

        manifest = JobManifest(
            job_id=JobStore.new_job_id(),
            status="queued",
            created_at=_now_iso(),
            updated_at=_now_iso(),
            prompt=prompt,
            duration_seconds=float(duration_seconds),
            fps=float(fps),
            height=int(height),
            width=int(width),
            output_format=output_format,
            style=self.options.style,
            model_name=model_name or self.adapter.model_name,
            model_type=type(self.adapter).__name__,
            segment_seconds=segment_seconds,
            total_frames=plan.total_frames,
            seed=seed,
            start_image=str(start_image) if start_image else None,
            character_references=[str(p) for p in (character_references or [])],
            scene_plan=scene_plan.to_dict(),
            segments=[
                SegmentRecord(
                    index=s.index,
                    requested_frames=s.num_frames,
                    prompt=scene_plan.segment_prompt(s.index),
                    seed=seed,
                    model_name=self.adapter.model_name,
                )
                for s in plan.segments
            ],
            options={
                **{k: v for k, v in self.options.__dict__.items()},
                "planned_actual_duration": round(plan.actual_duration_seconds, 4),
                "frame_count_constraint": caps.frame_count_constraint,
            },
        )
        self.store.create(manifest)
        logger.info(
            "Created job %s: %d segments, %d frames total (~%.2fs planned)",
            manifest.job_id, len(manifest.segments), manifest.total_frames,
            plan.actual_duration_seconds,
        )
        return manifest

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, job_id: str) -> JobManifest:
        """Run (or resume) a job to completion / pause / cancel.

        Completed segments are validated and skipped — never regenerated.
        """
        manifest = self.store.load(job_id)
        ctrl = JobController(self.store, job_id)
        if manifest.status == "completed":
            logger.info("Job %s already completed.", job_id)
            return manifest
        if manifest.status == "cancelled":
            raise RuntimeError(f"Job {job_id} is cancelled; create a new job.")
        if manifest.status == "running":
            logger.warning(
                "Job %s marked 'running' — assuming previous worker died; "
                "taking over after lock acquisition.", job_id,
            )
        ctrl.prepare_resume(manifest)

        if not self.gpu_lock.acquire(timeout=300):
            manifest.set_status("queued")
            manifest.errors.append("GPU busy: another job holds the lock.")
            self.store.save(manifest)
            raise RuntimeError(
                "Could not acquire GPU lock within 300s. Another job may be "
                "running; retry later or raise the lock timeout."
            )
        try:
            self._run_segments(manifest, ctrl)
        finally:
            self.gpu_lock.release()
        return self.store.load(job_id)

    # ------------------------------------------------------------------
    def _run_segments(self, manifest: JobManifest, ctrl: JobController) -> None:
        opts = self.options
        total = len(manifest.segments)
        while True:
            signal = ctrl.check(manifest)
            if signal == ControlSignal.CANCEL:
                ctrl.apply_cancel(manifest)
                logger.info("Job %s cancelled at segment boundary.", manifest.job_id)
                return
            if signal == ControlSignal.PAUSE:
                ctrl.apply_pause(manifest)
                logger.info("Job %s paused at segment boundary.", manifest.job_id)
                return

            seg = manifest.next_incomplete_segment
            if seg is None:
                break

            # Resume path: skip + validate already-done segments.
            if seg.status == "done":
                if seg.output_path and self._clip_ok(seg):
                    continue
                logger.warning(
                    "Segment %d marked done but file invalid -> regenerating.",
                    seg.index,
                )
                seg.status = "pending"

            seg.status = "running"
            seg.started_at = _now_iso()
            self.store.save(manifest)

            attempt = 0
            height, width = manifest.height, manifest.width
            while True:
                try:
                    self._generate_one(manifest, seg, height, width)
                    break
                except Exception as exc:  # noqa: BLE001 - recorded & re-raised
                    seg.retries += 1
                    attempt += 1
                    oom = _is_oom(exc)
                    recoverable = (
                        oom
                        and opts.retry_on_oom_downscale
                        and attempt <= opts.max_segment_retries
                        and min(height, width) > 256
                    )
                    if recoverable:
                        new_h = max(256, height // 2)
                        new_w = max(256, width // 2)
                        new_h -= new_h % 8
                        new_w -= new_w % 8
                        if (new_h, new_w) == (height, width) or \
                                not self.adapter.get_capabilities().supports_dimensions(new_h, new_w):
                            recoverable = False
                        else:
                            logger.warning(
                                "CUDA OOM on segment %d. Retrying at reduced "
                                "resolution %dx%d (was %dx%d) because "
                                "retry_on_oom_downscale is enabled.",
                                seg.index, new_h, new_w, height, width,
                            )
                            manifest.errors.append(
                                f"Segment {seg.index}: OOM at {height}x{width}; "
                                f"auto-retried at {new_h}x{new_w} (user-approved "
                                f"via retry_on_oom_downscale)."
                            )
                            height, width = new_h, new_w
                            self.adapter.cleanup()
                            continue
                    seg.status = "failed"
                    seg.error = f"{type(exc).__name__}: {exc}"
                    seg.finished_at = _now_iso()
                    manifest.set_status("failed")
                    manifest.errors.append(
                        f"Segment {seg.index} failed: {seg.error}"
                        + (" [OOM]" if oom else "")
                    )
                    self.store.save(manifest)
                    if oom:
                        raise OutOfMemoryError(
                            f"CUDA out-of-memory generating segment {seg.index} at "
                            f"{height}x{width}. Remedies: enable CPU offload + VAE "
                            f"slicing, lower resolution/fps/steps, free VRAM, or set "
                            f"retry_on_oom_downscale=true. Use a lighter model "
                            f"(e.g. CogVideoX-2b or LTX-Video on T4-class GPUs)."
                        ) from exc
                    raise

            # Checkpoint succeeded: keep the clip even if later segments fail.
            seg.status = "done"
            seg.finished_at = _now_iso()
            manifest.recompute_progress()
            self.store.save(manifest)
            if self.progress_callback:
                try:
                    self.progress_callback(self.store.load(manifest.job_id))
                except Exception:  # pragma: no cover - callback must not kill job
                    pass
            logger.info(
                "Job %s: segment %d/%d complete (%.0f%%)",
                manifest.job_id, seg.index + 1, total, manifest.progress * 100,
            )

        # All segments done -> assemble.
        manifest.recompute_progress()
        if manifest.next_incomplete_segment is None:
            try:
                self.assemble_job(manifest)
                manifest.set_status("completed")
            except Exception as exc:  # assembly failure keeps clips intact
                manifest.errors.append(
                    f"Assembly failed: {type(exc).__name__}: {exc}"
                )
                manifest.set_status("failed")
            self.store.save(manifest)

    # ------------------------------------------------------------------
    def _generate_one(
        self,
        manifest: JobManifest,
        seg: SegmentRecord,
        height: int,
        width: int,
    ) -> str:
        opts = self.options
        seg_dir = self.store.segments_dir(manifest.job_id)
        refs_dir = self.store.refs_dir(manifest.job_id)
        clip_path = seg_dir / f"segment_{seg.index:04d}.mp4"
        ref_path = refs_dir / f"final_{seg.index:04d}.png"

        reference_image = self._continuation_reference(manifest, seg)

        gen_kwargs = dict(
            num_frames=seg.requested_frames,
            height=height,
            width=width,
            seed=(None if seg.seed is None else int(seg.seed) + seg.index),
            negative_prompt=opts.negative_prompt,
            num_inference_steps=opts.num_inference_steps,
            guidance_scale=opts.guidance_scale,
        )
        if reference_image is not None:
            output = self.adapter.generate_from_image(
                image_path=reference_image, prompt=seg.prompt, **gen_kwargs
            )
            seg.model_name = self.adapter.model_name + " (i2v)"
        else:
            output = self.adapter.generate_clip(prompt=seg.prompt, **gen_kwargs)

        frames = getattr(output, "frames", output)
        self.exporter.export(frames, clip_path, fps=int(round(manifest.fps)))

        # Validate real output before recording success.
        actual_frames = validate_clip_file(clip_path, min_frames=1)
        seg.generated_frames = actual_frames
        seg.output_path = str(clip_path)

        # Extract final frame for continuity. Failures degrade continuity but
        # do not invalidate the produced clip.
        try:
            saved = extract_final_frame(frames, ref_path)
            seg.reference_frame_path = saved
        except ClipValidationError as exc:
            logger.warning("Continuity degraded for segment %d: %s", seg.index, exc)
            seg.reference_frame_path = None
        return str(clip_path)

    def _continuation_reference(
        self, manifest: JobManifest, seg: SegmentRecord
    ) -> Optional[str]:
        """Return a reference image path ONLY if the model genuinely accepts it."""
        caps = self.adapter.get_capabilities()
        if not self.options.use_reference_continuation:
            return None
        if not caps.supports_image_to_video:
            return None
        if seg.index == 0:
            if manifest.start_image and Path(manifest.start_image).exists():
                return manifest.start_image
            return None
        prev = manifest.segments[seg.index - 1]
        if prev.status == "done" and prev.reference_frame_path \
                and Path(prev.reference_frame_path).exists():
            return prev.reference_frame_path
        return None

    def _clip_ok(self, seg: SegmentRecord) -> bool:
        try:
            n = validate_clip_file(seg.output_path, min_frames=1)
            seg.generated_frames = n
            return True
        except ClipValidationError:
            return False

    # ------------------------------------------------------------------
    def assemble_job(
        self, manifest: JobManifest, output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Concatenate all done segments into the final video."""
        from ..assembly.assembler import VideoAssembler

        done = [s for s in manifest.segments if s.status == "done"]
        if not done:
            raise RuntimeError("No completed segments to assemble.")
        missing = [s.index for s in done
                   if not (s.output_path and Path(s.output_path).exists())]
        if missing:
            raise RuntimeError(f"Segment files missing for indices: {missing}")
        paths = sorted(done, key=lambda s: s.index)
        out = Path(output_path) if output_path else (
            self.store.job_dir(manifest.job_id)
            / f"final_{manifest.job_id}.{manifest.output_format}"
        )
        assembler = VideoAssembler()
        result = assembler.assemble(
            segment_paths=[s.output_path for s in paths],
            output_path=out,
            target_fps=manifest.fps,
            target_size=(manifest.width, manifest.height),
            transition_seconds=float(
                manifest.options.get("transition_seconds", 0.0) or 0.0
            ),
            requested_duration=manifest.duration_seconds,
            cleanup_intermediates=bool(
                manifest.options.get("cleanup_intermediates", False)
            ),
        )
        manifest.final_video_path = result.output_path
        manifest.options["assembly"] = result.to_dict()
        self.store.save(manifest)
        logger.info(
            "Assembled %s (%s, %.2fs actual vs %.2fs requested)",
            result.output_path, result.method, result.actual_duration,
            manifest.duration_seconds,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    def preflight(self, height: int, width: int) -> List[str]:
        """Warn about likely hardware/model incompatibilities BEFORE a job."""
        warnings: List[str] = []
        caps = self.adapter.get_capabilities()
        try:
            import torch

            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                need = caps.approx_vram_gb_fp16
                if need and free_gb < need:
                    warnings.append(
                        f"GPU has ~{free_gb:.1f} GB free but '{self.adapter.model_name}' "
                        f"needs ~{need:.1f} GB fp16 with offload. Expect OOM or use a "
                        f"smaller model/resolution."
                    )
            elif self.adapter.device == "cpu":
                warnings.append(
                    "Running on CPU: supported but extremely slow for video models."
                )
        except ImportError:  # pragma: no cover
            pass
        if not caps.supports_dimensions(height, width):
            warnings.append(f"{height}x{width} violates model dimension constraints.")
        return warnings
