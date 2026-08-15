"""
VAE Optimizer for Yukka Video AI

Provides VAE slicing and tiling optimizations to reduce memory usage
during video generation.
"""

from typing import Optional, Any


class VAEOptimizer:
    """
    Optimizes VAE (Variational Autoencoder) memory usage.

    This class provides methods to enable VAE slicing and tiling,
    which process video frames in smaller chunks to reduce peak
    memory consumption during encoding/decoding.
    """

    def __init__(self, pipeline: Optional[Any] = None):
        """
        Initialize the VAE Optimizer.

        Args:
            pipeline: The diffusers pipeline to optimize.
        """
        self.pipeline = pipeline
        self._slicing_enabled = False
        self._tiling_enabled = False

    def enable_slicing(self) -> None:
        """
        Enable VAE slicing.

        Slicing processes frames one at a time or in small batches,
        significantly reducing memory usage at the cost of some speed.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline set for optimization")

        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
            self._slicing_enabled = True
            print("VAE slicing enabled")
        else:
            print("VAE slicing not supported by this pipeline")

    def disable_slicing(self) -> None:
        """Disable VAE slicing."""
        if self.pipeline is None:
            return

        if hasattr(self.pipeline, "disable_vae_slicing"):
            self.pipeline.disable_vae_slicing()
            self._slicing_enabled = False
            print("VAE slicing disabled")

    def enable_tiling(self) -> None:
        """
        Enable VAE tiling.

        Tiling processes each frame in spatial tiles, reducing memory
        usage for high-resolution videos. Can be combined with slicing.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline set for optimization")

        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            self._tiling_enabled = True
            print("VAE tiling enabled")
        else:
            print("VAE tiling not supported by this pipeline")

    def disable_tiling(self) -> None:
        """Disable VAE tiling."""
        if self.pipeline is None:
            return

        if hasattr(self.pipeline, "disable_vae_tiling"):
            self.pipeline.disable_vae_tiling()
            self._tiling_enabled = False
            print("VAE tiling disabled")

    def enable_all_optimizations(self) -> None:
        """Enable both slicing and tiling for maximum memory savings."""
        self.enable_slicing()
        self.enable_tiling()
        print("All VAE optimizations enabled")

    def disable_all_optimizations(self) -> None:
        """Disable all VAE optimizations."""
        self.disable_slicing()
        self.disable_tiling()
        print("All VAE optimizations disabled")

    @property
    def is_slicing_enabled(self) -> bool:
        """Check if slicing is enabled."""
        return self._slicing_enabled

    @property
    def is_tiling_enabled(self) -> bool:
        """Check if tiling is enabled."""
        return self._tiling_enabled

    def get_optimization_status(self) -> dict:
        """
        Get current optimization status.

        Returns:
            Dictionary containing optimization settings.
        """
        return {
            "slicing_enabled": self._slicing_enabled,
            "tiling_enabled": self._tiling_enabled,
        }
