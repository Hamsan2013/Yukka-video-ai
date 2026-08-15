"""
Base Model Adapter for Yukka Video AI
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    All model-specific adapters must inherit from this class.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, dtype: torch.dtype = torch.float16):
        """
        Initialize the base model adapter.

        Args:
            model_name: Name or path of the model to load.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
            dtype: Data type for model weights.
        """
        self.model_name = model_name
        self.device = device or self._get_default_device()
        self.dtype = dtype
        self.pipeline = None
        self._model_loaded = False

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the model and return the pipeline.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate video frames from a text prompt.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this model.
        Must be implemented by subclasses.
        """
        pass

    def _get_default_device(self) -> str:
        """Get the default device based on available hardware."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def enable_cpu_offload(self) -> None:
        """Enable CPU offloading for memory efficiency."""
        if self.pipeline is not None and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
            print(f"CPU offload enabled for {self.model_name}")

    def enable_vae_slicing(self) -> None:
        """Enable VAE slicing for memory efficiency."""
        if self.pipeline is not None and hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
            print(f"VAE slicing enabled for {self.model_name}")

    def enable_vae_tiling(self) -> None:
        """Enable VAE tiling for memory efficiency."""
        if self.pipeline is not None and hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            print(f"VAE tiling enabled for {self.model_name}")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
