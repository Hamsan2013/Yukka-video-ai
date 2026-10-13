"""
LTX-Video Adapter for Yukka Video AI

Supports Lightricks/LTX-Video via diffusers' LTXPipeline (text-to-video)
and LTXImageToVideoPipeline (image-to-video). The I2V pipeline genuinely
accepts a starting image, so this adapter can be used for reference-guided
continuation of sequential clips.

Frame-count note: LTX-Video in diffusers requires num_frames of the form
8k + 1 (default 41; up to 96+1 style counts depending on version). We snap
requests to the nearest valid 8k+1 value.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .base import BaseModelAdapter, MemoryEstimate
from .capabilities import ModelCapabilities


class LTXAdapter(BaseModelAdapter):
    """Adapter for Lightricks LTX-Video (T2V + I2V)."""

    def __init__(
        self,
        model_name: str = "Lightricks/LTX-Video",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name, device=device, dtype=dtype, cache_dir=cache_dir
        )
        self._i2v_pipeline = None

    # ------------------------------------------------------------------
    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_text_to_video=True,
            supports_image_to_video=True,  # via LTXImageToVideoPipeline
            supports_reference_conditioning=False,  # start frame only
            supports_seed=True,
            dimension_divisor=32,
            frame_count_constraint="num_frames must be 8k+1 (e.g. 41, 49, 81, 97)",
            min_frames=9,
            max_frames=300,
            preferred_fps=25.0,
            approx_vram_gb_fp16=10.0,
            license="LTX-Video Community License (research/non-commercial tiers apply)",
            notes=(
                "Image-to-video uses the LAST FRAME of the previous clip as the "
                "first frame of the next one. Continuity improves but identity "
                "preservation is not guaranteed."
            ),
        )

    def adjust_num_frames(self, num_frames: int) -> int:
        n = max(9, int(num_frames))
        k = round((n - 1) / 8)
        return max(1, k) * 8 + 1

    # ------------------------------------------------------------------
    def load_model(self) -> Any:
        try:
            from diffusers import LTXPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "diffusers with LTX-Video support is required: "
                "pip install 'diffusers[torch]>=0.32'"
            ) from exc

        kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self.pipeline = LTXPipeline.from_pretrained(self.model_name, **kwargs)
        self._apply_pending_optimizations()
        if self.pipeline is not None and not getattr(self, "_offload_ever_requested", False):
            self.pipeline = self.pipeline.to(self.device)
        self._model_loaded = True
        return self.pipeline

    def _load_i2v_pipeline(self):
        """Lazily construct the image-to-video pipeline (shares cached weights)."""
        if self._i2v_pipeline is not None:
            return self._i2v_pipeline
        try:
            from diffusers import LTXImageToVideoPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError("diffusers >= 0.32 required for LTX image-to-video") from exc

        kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._i2v_pipeline = LTXImageToVideoPipeline.from_pretrained(
            self.model_name, **kwargs
        )
        self._apply_pending_optimizations_on(self._i2v_pipeline)
        if not getattr(self, "_offload_ever_requested", False):
            self._i2v_pipeline = self._i2v_pipeline.to(self.device)
        return self._i2v_pipeline

    def _apply_pending_optimizations_on(self, pipe) -> None:
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if getattr(self, "_offload_ever_requested", False) and hasattr(
            pipe, "enable_model_cpu_offload"
        ):
            try:
                pipe.enable_model_cpu_offload()
            except ValueError:
                pipe.to("cpu")

    def enable_cpu_offload(self) -> None:
        self._offload_ever_requested = True
        super().enable_cpu_offload()
        if self._i2v_pipeline is not None:
            self._apply_pending_optimizations_on(self._i2v_pipeline)

    # ------------------------------------------------------------------
    def generate_clip(
        self,
        prompt: str,
        num_frames: int = 41,
        height: int = 512,
        width: int = 768,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        fps: Optional[float] = None,
        **kwargs,
    ) -> Any:
        if not self._model_loaded:
            self.load_model()

        caps = self.get_capabilities()
        if not caps.supports_dimensions(height, width):
            raise ValueError(
                f"LTX-Video requires height/width divisible by {caps.dimension_divisor}; "
                f"got {height}x{width}."
            )
        num_frames = self.adjust_num_frames(num_frames)

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            frame_rate=fps if fps else self.get_default_config()["fps"],
            generator=self._make_generator(seed),
            **kwargs,
        )
        return output

    def generate_from_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        num_frames: int = 41,
        height: int = 512,
        width: int = 768,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        fps: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Generate a clip continuing from a starting image (genuine I2V)."""
        caps = self.get_capabilities()
        if not caps.supports_image_to_video:
            from .base import UnsupportedFeatureError

            raise UnsupportedFeatureError("LTX adapter unexpectedly lost I2V support")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        pipe = self._load_i2v_pipeline()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        num_frames = self.adjust_num_frames(num_frames)

        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            frame_rate=fps if fps else self.get_default_config()["fps"],
            generator=self._make_generator(seed),
            **kwargs,
        )
        return output

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        super().unload_model()
        self._i2v_pipeline = None

    def get_memory_requirements(self) -> MemoryEstimate:
        return MemoryEstimate(
            approx_model_gb=6.0,
            approx_peak_inference_gb=10.0,
            notes="LTX-Video 2B runs at ~768x512 on a 12GB GPU with offloading.",
        )

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dtype": "float16",
            "num_inference_steps": 50,
            "height": 512,
            "width": 768,
            "fps": 25,
            "num_frames": 41,
            "guidance_scale": 3.0,
        }
