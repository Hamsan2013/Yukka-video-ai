"""
Main Video Generation Pipeline for Yukka Video AI

This module provides the core pipeline for text-to-video generation using
open-source models from Hugging Face diffusers. It supports both CogVideoX-2b
and LTX-Video models, with memory optimizations for running on Google Colab T4 GPUs.
"""

from typing import Any, Dict, List, Optional, Union
import torch
from pathlib import Path

from ..models.cogvideo_adapter import CogVideoAdapter
from ..models.ltx_adapter import LTXAdapter
from .video_exporter import VideoExporter


class YukkaVideoPipeline:
    """
    Main pipeline class for text-to-video generation.

    This class orchestrates the video generation process, handling model loading,
    inference, and video export. It is designed to work efficiently on limited
    GPU memory by utilizing CPU offloading and VAE optimization techniques.

    Example:
        >>> pipeline = YukkaVideoPipeline.from_pretrained("THUDM/CogVideoX-2b")
        >>> pipeline.enable_model_cpu_offload()
        >>> output = pipeline("A robot walking in the rain")
        >>> pipeline.export_to_video(output.frames, "output.mp4")
    """

    SUPPORTED_MODELS = {
        "cogvideo": "THUDM/CogVideoX-2b",
        "ltx": "Lightricks/LTX-Video",
    }

    def __init__(
        self,
        adapter: Union[CogVideoAdapter, LTXAdapter],
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the YukkaVideoPipeline.

        Args:
            adapter: Model adapter instance (CogVideoAdapter or LTXAdapter).
            device: Device to run inference on.
            dtype: Data type for model weights.
        """
        self.adapter = adapter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.video_exporter = VideoExporter()
        self._pipeline_loaded = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "YukkaVideoPipeline":
        """
        Create a pipeline from a pre-trained model.

        Args:
            model_name: Name or path of the model to load.
            model_type: Type of model ('cogvideo' or 'ltx'). If None, auto-detected.
            device: Device to run the model on.
            dtype: Data type for model weights.

        Returns:
            Initialized YukkaVideoPipeline instance.

        Example:
            >>> pipeline = YukkaVideoPipeline.from_pretrained("THUDM/CogVideoX-2b")
        """
        # Auto-detect model type based on model name
        if model_type is None:
            if "cogvideo" in model_name.lower():
                model_type = "cogvideo"
            elif "ltx" in model_name.lower():
                model_type = "ltx"
            else:
                # Default to CogVideoX-2b for unknown models
                model_type = "cogvideo"

        # Create appropriate adapter
        if model_type == "cogvideo":
            adapter = CogVideoAdapter(model_name=model_name, device=device, dtype=dtype)
        elif model_type == "ltx":
            adapter = LTXAdapter(model_name=model_name, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls(adapter=adapter, device=device, dtype=dtype)

    def enable_model_cpu_offload(self) -> None:
        """
        Enable CPU offloading for the model.

        This moves model components to CPU when not in use, significantly
        reducing GPU memory usage at the cost of slightly slower inference.
        Essential for running on Colab T4 GPUs with 16GB VRAM.
        """
        self.adapter.enable_cpu_offload()
        print("Model CPU offload enabled")

    def enable_vae_slicing(self) -> None:
        """
        Enable VAE slicing for memory efficiency.

        Processes video frames in slices to reduce peak memory usage.
        """
        self.adapter.enable_vae_slicing()
        print("VAE slicing enabled")

    def enable_vae_tiling(self) -> None:
        """
        Enable VAE tiling for memory efficiency.

        Processes video frames in tiles to reduce peak memory usage.
        Can be combined with slicing for maximum memory savings.
        """
        self.adapter.enable_vae_tiling()
        print("VAE tiling enabled")

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_videos: int = 1,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: int = 8,
        num_frames: Optional[int] = None,
        guidance_scale: float = 6.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Any:
        """
        Generate video from a text prompt.

        Args:
            prompt: Text description of the video to generate.
            negative_prompt: Text description of what to avoid in the output.
            num_videos: Number of videos to generate.
            num_inference_steps: Number of denoising steps (more = better quality, slower).
            height: Height of output video in pixels (default from model config).
            width: Width of output video in pixels (default from model config).
            fps: Frames per second for the output video.
            num_frames: Total number of frames to generate (default from model config).
            guidance_scale: Classifier-free guidance scale (higher = more prompt adherence).
            generator: Random number generator for reproducibility.
            **kwargs: Additional arguments passed to the model's generate method.

        Returns:
            Pipeline output containing generated frames.

        Example:
            >>> output = pipeline(
            ...     "A cinematic shot of a futuristic robot in the rain",
            ...     num_inference_steps=50,
            ...     guidance_scale=6.0,
            ... )
            >>> frames = output.frames
        """
        # Get default config and override with provided values
        config = self.adapter.get_default_config()

        if height is None:
            height = config.get("height", 480)
        if width is None:
            width = config.get("width", 720)
        if num_frames is None:
            num_frames = config.get("num_frames", 49)

        # Set seed for reproducibility if not provided
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(42)

        # Generate video frames
        output = self.adapter.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            fps=fps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
            num_videos_per_prompt=num_videos,
            **kwargs,
        )

        self._pipeline_loaded = True
        return output

    def export_to_video(
        self,
        frames: Any,
        output_path: Union[str, Path],
        fps: int = 8,
        codec: str = "libx264",
        pixel_format: str = "yuv420p",
    ) -> str:
        """
        Export generated frames to an MP4 video file.

        Args:
            frames: Video frames from pipeline output.
            output_path: Path to save the output video file.
            fps: Frames per second for the output video.
            codec: Video codec to use (default: libx264).
            pixel_format: Pixel format for the output video.

        Returns:
            Path to the saved video file.

        Example:
            >>> output = pipeline("A cat playing piano")
            >>> video_path = pipeline.export_to_video(output.frames, "cat_piano.mp4")
        """
        return self.video_exporter.export(
            frames=frames,
            output_path=output_path,
            fps=fps,
            codec=codec,
            pixel_format=pixel_format,
        )

    def generate_and_save(
        self,
        prompt: str,
        output_path: Union[str, Path],
        **kwargs,
    ) -> str:
        """
        Generate video from prompt and save directly to file.

        Convenience method that combines generation and export.

        Args:
            prompt: Text description of the video to generate.
            output_path: Path to save the output video file.
            **kwargs: Arguments passed to the generation method.

        Returns:
            Path to the saved video file.

        Example:
            >>> video_path = pipeline.generate_and_save(
            ...     "A robot walking in the rain",
            ...     "robot_rain.mp4",
            ...     num_inference_steps=50,
            ... )
        """
        output = self(prompt, **kwargs)
        return self.export_to_video(output.frames, output_path)

    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded and ready."""
        return self.adapter.is_loaded and self._pipeline_loaded
