"""
LTX Video Adapter for Yukka Video AI
Supports Lightricks/LTX-Video model
"""

from typing import Any, Dict, Optional
import torch

from .base import BaseModelAdapter


class LTXAdapter(BaseModelAdapter):
    """
    Adapter for Lightricks LTX-Video model.
    """

    def __init__(
        self,
        model_name: str = "Lightricks/LTX-Video",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype)

    def load_model(self) -> Any:
        """Load the LTX-Video pipeline."""
        try:
            from diffusers import LTXPipeline
        except ImportError:
            raise ImportError(
                "Please install diffusers with: pip install diffusers[torch]"
            )

        self.pipeline = LTXPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        self._model_loaded = True
        return self.pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        num_frames: int = 81,
        **kwargs,
    ) -> Any:
        """
        Generate video frames from a text prompt using LTX-Video.

        Args:
            prompt: Text description of the video to generate.
            negative_prompt: Text description of what to avoid.
            num_inference_steps: Number of denoising steps.
            height: Height of the output video in pixels.
            width: Width of the output video in pixels.
            fps: Frames per second.
            num_frames: Total number of frames to generate.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            Generated video frames.
        """
        if not self._model_loaded:
            self.load_model()

        if negative_prompt is None:
            negative_prompt = ""

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            fps=fps,
            num_frames=num_frames,
            **kwargs,
        )

        return output

    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for LTX-Video."""
        return {
            "model_name": "Lightricks/LTX-Video",
            "dtype": "float16",
            "num_inference_steps": 50,
            "height": 480,
            "width": 720,
            "fps": 8,
            "num_frames": 81,
        }
