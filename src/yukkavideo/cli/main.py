"""
Command-Line Interface for Yukka Video AI

Provides a CLI for generating videos from text prompts.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="yukkavideo",
        description="Yukka Video AI - Text-to-Video Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a video with default settings
  yukkavideo generate "A robot walking in the rain"

  # Generate with custom settings
  yukkavideo generate "A cat playing piano" --steps 100 --height 720 --width 1280

  # Use a specific model
  yukkavideo generate "Sunset over mountains" --model Lightricks/LTX-Video

  # Show hardware info
  yukkavideo info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a video from a text prompt",
    )
    gen_parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the video to generate",
    )
    gen_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.mp4",
        help="Output video file path (default: output.mp4)",
    )
    gen_parser.add_argument(
        "-m", "--model",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Model name or path (default: THUDM/CogVideoX-2b)",
    )
    gen_parser.add_argument(
        "-t", "--model-type",
        type=str,
        choices=["cogvideo", "ltx"],
        default=None,
        help="Model type (auto-detected if not specified)",
    )
    gen_parser.add_argument(
        "-s", "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    gen_parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 480)",
    )
    gen_parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Video width in pixels (default: 720)",
    )
    gen_parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second (default: 8)",
    )
    gen_parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames (auto-selected based on model)",
    )
    gen_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale (default: 6.0)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    gen_parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Disable CPU offloading (uses more VRAM)",
    )
    gen_parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (what to avoid in the output)",
    )
    gen_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show hardware and system information",
    )

    # ------------------------------------------------------------------
    # Long-video job commands (sequential generation controller)
    # ------------------------------------------------------------------
    default_jobs_dir = "outputs/jobs"

    start_parser = subparsers.add_parser(
        "job-start",
        help="Create a sequential long-video generation job and run it",
    )
    start_parser.add_argument("prompt", type=str, help="Video description")
    start_parser.add_argument("--duration", type=float, default=10.0,
                              help="Total duration in seconds (no hard maximum)")
    start_parser.add_argument("--fps", type=int, default=8)
    start_parser.add_argument("--height", type=int, default=480)
    start_parser.add_argument("--width", type=int, default=720)
    start_parser.add_argument("-m", "--model", type=str, default="THUDM/CogVideoX-2b")
    start_parser.add_argument("--segment-seconds", type=float, default=5.0)
    start_parser.add_argument("--steps", type=int, default=50)
    start_parser.add_argument("--guidance-scale", type=float, default=6.0)
    start_parser.add_argument("--seed", type=int, default=42)
    start_parser.add_argument("--style", type=str, default="")
    start_parser.add_argument("--start-image", type=str, default=None,
                              help="Optional first frame (only used if model supports I2V)")
    start_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)
    start_parser.add_argument("--cache-dir", type=str, default=None)
    start_parser.add_argument("--no-offload", action="store_true")
    start_parser.add_argument("--run-now", action="store_true", default=True,
                              help="Run immediately (default)")
    start_parser.add_argument("--enqueue-only", action="store_true",
                              help="Only create the job; run later via `yukkavideo job-run`")

    list_parser = subparsers.add_parser("job-list", help="List generation jobs")
    list_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)

    status_parser = subparsers.add_parser("job-status",
                                          help="Show detailed job status")
    status_parser.add_argument("job_id")
    status_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)

    resume_parser = subparsers.add_parser(
        "job-run", help="Run/resume a job (completed segments are skipped)")
    resume_parser.add_argument("job_id")
    resume_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)
    resume_parser.add_argument("--cache-dir", type=str, default=None)
    resume_parser.add_argument("--no-offload", action="store_true")

    pause_parser = subparsers.add_parser("job-pause",
                                         help="Request pause at next segment boundary")
    pause_parser.add_argument("job_id")
    pause_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)

    cancel_parser = subparsers.add_parser("job-cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id")
    cancel_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)

    assemble_parser = subparsers.add_parser(
        "job-assemble", help="Assemble completed segments into the final video")
    assemble_parser.add_argument("job_id")
    assemble_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)
    assemble_parser.add_argument("-o", "--output", type=str, default=None)

    worker_parser = subparsers.add_parser(
        "worker", help="Run the background job worker loop")
    worker_parser.add_argument("--jobs-dir", type=str, default=default_jobs_dir)
    worker_parser.add_argument("--cache-dir", type=str, default=None)
    worker_parser.add_argument("--once", action="store_true")

    return parser


def cli_generate(args: argparse.Namespace) -> int:
    """
    Execute the generate command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    import torch
    from ..inference.pipeline import YukkaVideoPipeline
    from ..utils.hardware import print_hardware_summary
    from ..utils.logging_config import setup_logging

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level.upper())

    print("=" * 50)
    print("Yukka Video AI - Video Generation")
    print("=" * 50)

    # Print hardware info
    print_hardware_summary()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load pipeline
    print(f"\nLoading model: {args.model}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipeline = YukkaVideoPipeline.from_pretrained(
            model_name=args.model,
            model_type=args.model_type,
            device=device,
            dtype=dtype,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Enable optimizations
    if not args.no_offload and device == "cuda":
        print("Enabling CPU offload...")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()

    # Generate video
    print(f"\nGenerating video for prompt: '{args.prompt}'")
    print(f"Settings: {args.steps} steps, {args.height}x{args.width}, {args.fps} fps")

    try:
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_frames=args.frames,
            guidance_scale=args.guidance_scale,
        )

        # Export to video
        output_path = Path(args.output)
        print(f"\nExporting video to: {output_path}")

        pipeline.export_to_video(
            frames=output.frames,
            output_path=output_path,
            fps=args.fps,
        )

        print("\n" + "=" * 50)
        print(f"✓ Video generated successfully: {output_path}")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\nError during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cli_info(args: argparse.Namespace) -> int:
    """
    Execute the info command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from ..utils.hardware import print_hardware_summary, get_hardware_info

    print("=" * 50)
    print("Yukka Video AI - System Information")
    print("=" * 50)

    print_hardware_summary()

    info = get_hardware_info()
    print("\nDetailed Info:")
    print(f"  Platform: {info['platform']}")
    print(f"  GPU Available: {info['gpu_available']}")
    if info["gpu_name"]:
        print(f"  GPU Name: {info['gpu_name']}")
    print(f"  CPU Cores: {info['cpu_count']}")

    return 0


def _job_store(args):
    from pathlib import Path as _P
    from ..jobs.manifest import JobStore
    return JobStore(_P(args.jobs_dir))


def cli_job_start(args) -> int:
    """Create (and by default run) a sequential long-video generation job."""
    import torch
    from ..jobs.runner import GenerationRunner, RunnerOptions
    from ..utils.logging_config import setup_logging

    setup_logging("DEBUG" if getattr(args, "verbose", False) else "INFO")
    store = _job_store(args)
    device = "cuda" if torch.cuda.is_available() and not args.no_offload else None
    adapter = build_adapter_obj(args.model, args.cache_dir, device)
    options = RunnerOptions(num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            style=args.style)
    runner = GenerationRunner(adapter, store, options=options)
    for w in runner.preflight(args.height, args.width):
        print(f"WARNING: {w}")
    manifest = runner.create_job(
        prompt=args.prompt,
        duration_seconds=args.duration,
        fps=args.fps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        start_image=args.start_image,
        segment_seconds=args.segment_seconds,
    )
    print(f"Job created: {manifest.job_id} ({len(manifest.segments)} segments)")
    if args.enqueue_only:
        print("Enqueued only. Run later with: yukkavideo job-run "
              f"{manifest.job_id} --jobs-dir {args.jobs_dir}")
        return 0
    result = runner.run(manifest.job_id)
    return _print_job_result(result)


def build_adapter_obj(model_name: str, cache_dir, device):
    """Create + load the adapter for a model name (shared helper)."""
    import torch
    from ..models import ADAPTER_REGISTRY, detect_model_type

    key = detect_model_type(model_name)
    dtype = torch.float16 if (device or "cuda") == "cuda" and torch.cuda.is_available() \
        else torch.float32
    adapter = ADAPTER_REGISTRY[key](model_name=model_name, device=device,
                                    dtype=dtype, cache_dir=cache_dir)
    adapter.load_model()
    if device != "cpu" and hasattr(adapter.pipeline, "enable_vae_slicing"):
        adapter.pipeline.enable_vae_slicing()
    return adapter


def cli_job_list(args) -> int:
    store = _job_store(args)
    jobs = store.list_jobs()
    if not jobs:
        print("No jobs found.")
        return 0
    print(f"{'job_id':<32} {'status':<10} {'progress':>8} {'segments':>9}  prompt")
    for m in jobs:
        done = sum(1 for s in m.segments if s.status == "done")
        print(f"{m.job_id:<32} {m.status:<10} {m.progress*100:>7.0f}% "
              f"{done:>4}/{len(m.segments)}  {m.prompt[:48]}")
    return 0


def cli_job_status(args) -> int:
    store = _job_store(args)
    m = store.load(args.job_id)
    print(f"Job {m.job_id}: status={m.status} progress={m.progress:.0%}")
    print(f"  prompt: {m.prompt}")
    print(f"  model: {m.model_name} | {m.height}x{m.width} @ {m.fps} fps | "
          f"{m.duration_seconds}s requested")
    for s in m.segments:
        print(f"  seg {s.index:02d}: {s.status:<8} req={s.requested_frames} "
              f"gen={s.generated_frames} out={s.output_path or '-'}"
              + (f" error={s.error}" if s.error else ""))
    if m.errors:
        print("  errors:")
        for e in m.errors:
            print(f"    - {e}")
    if m.final_video_path:
        print(f"  final video: {m.final_video_path}")
    return 0


def cli_job_run(args) -> int:
    """Run/resume a job; completed segments are validated and skipped."""
    from ..jobs.runner import GenerationRunner, RunnerOptions
    from ..jobs.worker import runner_from_manifest
    from ..utils.logging_config import setup_logging

    setup_logging("INFO")
    store = _job_store(args)
    manifest = store.load(args.job_id)
    runner = runner_from_manifest(manifest, store, cache_dir=args.cache_dir)
    if not runner.adapter.is_loaded:
        device = "cuda" if __import__("torch").cuda.is_available() \
            and not args.no_offload else None
        if device:
            runner.adapter.enable_cpu_offload()
        runner.adapter.load_model()
    result = runner.run(args.job_id)
    return _print_job_result(result)


def cli_job_pause(args) -> int:
    from ..jobs.control import JobController
    store = _job_store(args)
    JobController(store, args.job_id).request_pause()
    print(f"Pause requested for {args.job_id} (takes effect at next "
          f"segment boundary).")
    return 0


def cli_job_cancel(args) -> int:
    from ..jobs.control import JobController
    store = _job_store(args)
    JobController(store, args.job_id).request_cancel()
    print(f"Cancel requested for {args.job_id}. If the job is running, it "
          f"will stop at the next segment boundary; otherwise run "
          f"'yukkavideo job-status {args.job_id}' to confirm state.")
    return 0


def cli_job_assemble(args) -> int:
    from ..jobs.worker import runner_from_manifest
    store = _job_store(args)
    manifest = store.load(args.job_id)
    runner = runner_from_manifest(manifest, store, cache_dir=None)
    info = runner.assemble_job(manifest, output_path=args.output)
    print("Assembly complete:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


def cli_worker(args) -> int:
    from ..jobs.worker import main as worker_main
    argv = ["--jobs-dir", args.jobs_dir]
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    if args.once:
        argv.append("--once")
    return worker_main(argv)


def _print_job_result(m) -> int:
    if m.status == "completed":
        print(f"\nJob {m.job_id} COMPLETED.")
        print(f"Final video: {m.final_video_path}")
        asm = m.options.get("assembly", {})
        if asm:
            print(f"Actual duration: {asm.get('actual_duration_seconds')}s "
                  f"(requested {m.duration_seconds}s), method={asm.get('method')}")
        return 0
    print(f"\nJob {m.job_id} status: {m.status}, progress {m.progress:.0%}")
    for e in m.errors[-5:]:
        print(f"  error: {e}")
    if m.status in ("paused", "failed"):
        print(f"Resume with: yukkavideo job-run {m.job_id}")
    return 1


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "generate": cli_generate,
        "info": cli_info,
        "job-start": cli_job_start,
        "job-list": cli_job_list,
        "job-status": cli_job_status,
        "job-run": cli_job_run,
        "job-pause": cli_job_pause,
        "job-cancel": cli_job_cancel,
        "job-assemble": cli_job_assemble,
        "worker": cli_worker,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
