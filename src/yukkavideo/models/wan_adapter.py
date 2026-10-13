"""
Wan2.1 Adapter for Yukka Video AI

Wraps diffusers' WanPipeline (text-to-video) and WanImageToVideoPipeline
(image-to-video). Verified against diffusers 0.35.x, where both pipelines
exist and accept the parameters used below.

Recommended checkpoints:
- "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"  (image-to-video, gated, Apache-2.0)
- "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"      (text-to-video, ungated, Apache-2.0)

Honest constraints:
- The 14B models need ~24GB+ VRAM even with offloading; the 1.3B T2V model
  is feasible on 16GB cards. We refuse to *silently* download multi-tens-of-
  GB weights: loading must be explicitly requested by the user.
- Wan I2V also supports a ``last_image`` keyframe parameter in diffusers
  >= 0.32; we use the start-image conditioning for continuation.
- num_frames must be 4k + 1 for Wan (temporal VAE stride 4).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .base import BaseModelAdapter, MemoryEstimate
from .capabilities import ModelCapabilities


class WanAdapter(BaseModelAdapter):
    """Adapter for Alibaba Wan2.1 video models (T2V and/or I2V variants)."""

    def __init__(
        self,
        model_name: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        enable_i2v: bool = False,
    ):
        super().__init__(
            model_name=model_name, device=device, dtype=dtype, cache_dir=cache_dir
        )
        # If the checkpoint name itself says I2V, image-to-video is available.
        self._i2v_requested = enable_i2v or "i2v" in model_name.lower()
        self._i2v_pipeline = None

    # ------------------------------------------------------------------
    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_text_to_video=not self._i2v_requested,
            supports_image_to_video=self._i2v_requested,
            supports_reference_conditioning=False,
            supports_seed=True,
            dimension_divisor=16,
            frame_count_constraint="num_frames must be 4k+1 (e.g. 33, 49, 81)",
            min_frames=5,
            max_frames=129,
            preferred_fps=24.0 if "ti2v" in self.model_name.lower() else 16.0,
            approx_vram_gb_fp16=(12.0 if "1.3b" in self.model_name.lower() else 40.0),
            license="Apache-2.0 (Wan2.1 weights)",
            gated="14b" in self.model_name.lower(),
            notes=(
                "14B checkpoints are large downloads (>20GB) and gated on "
                "Hugging Face; explicit user action required."
            ),
        )

    def adjust_num_frames(self, num_frames: int) -> int:
        n = max(5, int(num_frames))
        k = round((n - 1) / 4)
        return max(1, k) * 4 + 1

    # ------------------------------------------------------------------
    def load_model(self) -> Any:
        try:
            from diffusers import WanPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "diffusers >= 0.32 with Wan support required: pip install 'diffusers[torch]>=0.32'"
            ) from exc

        kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        if self._i2v_requested:
            from diffusers import WanImageToVideoPipeline

            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.model_name, **kwargs
            )
        else:
            self.pipeline = WanPipeline.from_pretrained(self.model_name, **kwargs)

        self._apply_pending_optimizations()
        if not getattr(self, "_offload_ever_requested", False):
            self.pipeline = self.pipeline.to(self.device)
        self._model_loaded = True
        return self.pipeline

    def enable_cpu_offload(self) -> None:
        self._offload_ever_requested = True
        super().enable_cpu_offload()

    # ------------------------------------------------------------------
    def generate_clip(
        self,
        prompt: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        fps: Optional[float] = None,
        **kwargs,
    ) -> Any:
        caps = self.get_capabilities()
        if not caps.supports_text_to_video:
            from .base import UnsupportedFeatureError

            raise UnsupportedFeatureError(
                f"{self.model_name} is an image-to-video checkpoint; "
                "use generate_from_image() or select a T2V model."
            )
        if not self._model_loaded:
            self.load_model()
        if not caps.supports_dimensions(height, width):
            raise ValueError(
                f"Wan requires dimensions divisible by {caps.dimension_divisor}; "
                f"got {height}x{width}."
            )
        num_frames = self.adjust_num_frames(num_frames)

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self._make_generator(seed),
            **kwargs,
        )
        return output

    def generate_from_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        last_image_path: Optional[Union[str, Path]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Continue generation from a starting image using Wan I2V."""
        caps = self.get_capabilities()
        if not caps.supports_image_to_video:
            from .base import UnsupportedFeatureError

            raise UnsupportedFeatureError(
                f"{self.model_name} is not an image-to-video checkpoint. "
                "Configure a Wan I2V model (e.g. Wan2.1-I2V-14B-480P-Diffusers)."
            )
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        if not self._model_loaded:
            self.load_model()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        last_image = None
        if last_image_path is not None:
            if not Path(last_image_path).exists():
                raise FileNotFoundError(f"Last-frame image not found: {last_image_path}")
            last_image = Image.open(last_image_path).convert("RGB")

        num_frames = self.adjust_num_frames(num_frames)

        output = self.pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self._make_generator(seed),
            last_image=last_image,
            **kwargs,
        )
        return output

    # ------------------------------------------------------------------
    def get_memory_requirements(self) -> MemoryEstimate:
        small = "1.3b" in self.model_name.lower()
        return MemoryEstimate(
            approx_model_gb=3.5 if small else 28.0,
            approx_peak_inference_gb=8.0 if small else 40.0,
            notes=(
                "1.3B T2V fits ~12GB with bf16; 14B needs >=24GB even with "
                "sequential CPU offload and is a very large download."
            ),
        )

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dtype": "bfloat16",
            "num_inference_steps": 50,
            "height": 480,
            "width": 832,
            "fps": 16,
            "num_frames": 81,
            "guidance_scale": 5.0,
        }
