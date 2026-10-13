"""
Base Model Adapter for Yukka Video AI

Defines the common interface every model integration must implement:
loading/unloading, clip generation (text-to-video and, where genuinely
supported, image-to-video), capability reporting, memory estimation and
cleanup.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path

import torch

from .capabilities import ModelCapabilities


class MemoryEstimate:
    """Simple container for adapter memory requirements."""

    def __init__(
        self,
        approx_model_gb: float,
        approx_peak_inference_gb: float,
        notes: str = "",
    ):
        self.approx_model_gb = approx_model_gb
        self.approx_peak_inference_gb = approx_peak_inference_gb
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approx_model_gb": self.approx_model_gb,
            "approx_peak_inference_gb": self.approx_peak_inference_gb,
            "notes": self.notes,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MemoryEstimate(model≈{self.approx_model_gb:.1f}GB, "
            f"peak≈{self.approx_peak_inference_gb:.1f}GB)"
        )


class UnsupportedFeatureError(RuntimeError):
    """Raised when a caller requests an operation the model cannot do."""


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters.

    Subclasses wrap a concrete diffusers pipeline (or equivalent) and must
    honestly declare their capabilities through ``get_capabilities()``.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or self._get_default_device()
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.pipeline = None
        self._model_loaded = False
        # Deferred optimizations: requested before load_model() created the
        # pipeline; applied automatically right after initialization.
        self._pending_offload = False
        self._pending_vae_slicing = False
        self._pending_vae_tiling = False

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------
    @abstractmethod
    def load_model(self) -> Any:
        """Load the underlying pipeline and apply deferred optimizations."""

    @abstractmethod
    def generate_clip(
        self,
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        **kwargs,
    ) -> Any:
        """Generate one text-to-video clip. Returns pipeline output."""

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Return the explicit capability report for this model."""

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for this model."""

    # ------------------------------------------------------------------
    # Optional interface with safe defaults
    # ------------------------------------------------------------------
    def generate_from_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Generate a clip conditioned on a starting/reference image.

        The default implementation refuses unless the adapter declares
        ``supports_image_to_video`` — we never pretend a T2V-only model
        accepts image conditioning.
        """
        caps = self.get_capabilities()
        if not caps.supports_image_to_video:
            raise UnsupportedFeatureError(
                f"{self.model_name} does not support image-to-video generation."
            )
        raise NotImplementedError(
            "Adapters supporting image-to-video must override generate_from_image."
        )

    def adjust_num_frames(self, num_frames: int) -> int:
        """Snap a requested frame count to one the model can produce.

        Default: identity. Adapters with modular constraints (e.g. CogVideoX
        8k+1) override this.
        """
        return max(1, int(num_frames))

    def get_memory_requirements(self) -> Optional[MemoryEstimate]:
        """Estimate memory needs, if known for this model."""
        return None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _apply_pending_optimizations(self) -> None:
        """Apply optimizations requested before the pipeline existed."""
        if self._pending_offload:
            self.enable_cpu_offload()
        if self._pending_vae_slicing:
            self.enable_vae_slicing()
        if self._pending_vae_tiling:
            self.enable_vae_tiling()

    def enable_cpu_offload(self) -> None:
        """Enable model CPU offloading (deferred until pipeline exists)."""
        if self.pipeline is None:
            self._pending_offload = True
            return
        if hasattr(self.pipeline, "enable_model_cpu_offload"):
            try:
                self.pipeline.enable_model_cpu_offload()
                self._pending_offload = False
            except ValueError:
                # e.g. accelerate hooks require CUDA; fall back to plain CPU move
                self.pipeline.to("cpu")
                self._pending_offload = False
        else:
            self.pipeline.to("cpu")
            self._pending_offload = False

    def enable_vae_slicing(self) -> None:
        if self.pipeline is None:
            self._pending_vae_slicing = True
            return
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
        self._pending_vae_slicing = False

    def enable_vae_tiling(self) -> None:
        if self.pipeline is None:
            self._pending_vae_tiling = True
            return
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
        self._pending_vae_tiling = False

    def unload_model(self) -> None:
        """Release the pipeline and free device memory."""
        if self.pipeline is not None:
            try:
                from diffusers.hooks import remove_all_hooks  # type: ignore

                remove_all_hooks(self.pipeline)
            except Exception:
                pass
        self.pipeline = None
        self._model_loaded = False
        self.cleanup()

    def cleanup(self) -> None:
        """Free temporary tensors / caches."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _make_generator(self, seed: Optional[int]):
        """Build a seeded torch.Generator on the adapter device (if supported)."""
        if not self.get_capabilities().supports_seed:
            return None
        gen_device = self.device if self.device != "cpu" else "cpu"
        try:
            generator = torch.Generator(device=gen_device)
        except RuntimeError:
            # Device may be unavailable in mocked/test environments.
            generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        return generator

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    # Backwards-compatible alias used by the original codebase/tests.
    def generate(self, prompt: str, **kwargs) -> Any:
        """Legacy alias mapping onto :meth:`generate_clip`."""
        config = self.get_default_config()
        num_frames = kwargs.pop("num_frames", config.get("num_frames", 49))
        height = kwargs.pop("height", config.get("height", 480))
        width = kwargs.pop("width", config.get("width", 720))
        seed = kwargs.pop("seed", None)
        return self.generate_clip(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            **kwargs,
        )
