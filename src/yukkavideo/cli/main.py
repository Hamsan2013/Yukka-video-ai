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

    if args.command == "generate":
        return cli_generate(args)
    elif args.command == "info":
        return cli_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
