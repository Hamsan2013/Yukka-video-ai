"""
CogVideoX Adapter for Yukka Video AI

Supports THUDM/CogVideoX-2b / CogVideoX-5b via diffusers' CogVideoXPipeline.

Important honest limitations:
- The standard CogVideoXPipeline in diffusers 0.35.x is TEXT-TO-VIDEO ONLY.
  It does not accept a starting image, so this adapter declares
  ``supports_image_to_video = False`` and refuses reference frames.
- CogVideoX requires num_frames of the form 8k + 1 (e.g. 49, 81).
"""

from typing import Any, Dict, Optional

import torch

from .base import BaseModelAdapter, MemoryEstimate
from .capabilities import ModelCapabilities


class CogVideoAdapter(BaseModelAdapter):
    """Adapter for THUDM CogVideoX-2b / CogVideoX-5b (text-to-video)."""

    def __init__(
        self,
        model_name: str = "THUDM/CogVideoX-2b",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name, device=device, dtype=dtype, cache_dir=cache_dir
        )

    # ------------------------------------------------------------------
    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_text_to_video=True,
            supports_image_to_video=False,  # CogVideoXPipeline has no image input
            supports_reference_conditioning=False,
            supports_seed=True,
            dimension_divisor=8,
            frame_count_constraint="num_frames must be 8k+1 (e.g. 49 or 81)",
            min_frames=9,
            max_frames=81,
            preferred_fps=8.0,
            approx_vram_gb_fp16=10.0,
            license="apache-2.0 (model code); weights per THUDM agreement",
            notes=(
                "Text-to-video only. Cannot continue a clip from a reference "
                "frame; sequential continuity is limited to prompt-level style."
            ),
        )

    def adjust_num_frames(self, num_frames: int) -> int:
        """Snap to nearest valid 8k+1 value within [9, 81]."""
        n = max(9, int(num_frames))
        k = round((n - 1) / 8)
        snapped = max(1, k) * 8 + 1
        return min(snapped, 81)

    # ------------------------------------------------------------------
    def load_model(self) -> Any:
        try:
            from diffusers import CogVideoXPipeline
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "diffusers with CogVideoX support is required: "
                "pip install 'diffusers[torch]>=0.31'"
            ) from exc

        kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self.pipeline = CogVideoXPipeline.from_pretrained(self.model_name, **kwargs)

        # Apply optimizations that were requested before the pipeline existed.
        # If CPU offloading was requested, enable_model_cpu_offload manages
        # device placement itself; otherwise move the pipeline to the target
        # device explicitly.
        self._apply_pending_optimizations()
        if self.pipeline is not None and not self._offload_ever_requested:
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
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        **kwargs,
    ) -> Any:
        """Generate a text-to-video clip with CogVideoX."""
        if not self._model_loaded:
            self.load_model()

        caps = self.get_capabilities()
        if not caps.supports_dimensions(height, width):
            raise ValueError(
                f"CogVideoX cannot produce {height}x{width}; dimensions must be "
                f"multiples of {caps.dimension_divisor}."
            )
        num_frames = self.adjust_num_frames(num_frames)

        kwargs.pop("fps", None)  # CogVideoX ignores fps at inference time

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=self._make_generator(seed),
            **kwargs,
        )
        return output

    # ------------------------------------------------------------------
    def get_memory_requirements(self) -> MemoryEstimate:
        return MemoryEstimate(
            approx_model_gb=9.0 if "2b" in self.model_name.lower() else 19.0,
            approx_peak_inference_gb=12.0 if "2b" in self.model_name.lower() else 24.0,
            notes="With enable_model_cpu_offload, ~10-12GB VRAM suffices for 2b @480p.",
        )

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dtype": "float16",
            "num_inference_steps": 50,
            "height": 480,
            "width": 720,
            "fps": 8,
            "num_frames": 49,
            "guidance_scale": 6.0,
        }
