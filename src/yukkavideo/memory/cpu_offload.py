"""
CPU Offload Manager for Yukka Video AI

Provides utilities for managing CPU offloading of model components
to reduce GPU memory usage.
"""

import torch
from typing import Optional, Dict, Any


class CPUOffloadManager:
    """
    Manages CPU offloading for model components.

    This class provides methods to move model components between CPU and GPU
    dynamically, allowing for efficient memory usage on GPUs with limited VRAM.
    """

    def __init__(self, pipeline: Optional[Any] = None):
        """
        Initialize the CPU Offload Manager.

        Args:
            pipeline: The diffusers pipeline to manage.
        """
        self.pipeline = pipeline
        self._offloaded_components: Dict[str, torch.device] = {}

    def offload_model_to_cpu(self) -> None:
        """
        Offload the entire model to CPU.

        This moves all model weights to CPU while keeping the execution
        device configuration intact. Useful for freeing GPU memory when
        the model is not actively generating.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline set for offloading")

        if hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
            print("Model offloaded to CPU successfully")
        else:
            # Manual offloading for pipelines without built-in support
            self._manual_cpu_offload()

    def _manual_cpu_offload(self) -> None:
        """Manually offload model components to CPU."""
        if self.pipeline is None:
            return

        # Offload main model components
        components_to_offload = [
            "text_encoder",
            "transformer",
            "unet",
            "vae",
        ]

        for component_name in components_to_offload:
            if hasattr(self.pipeline, component_name):
                component = getattr(self.pipeline, component_name)
                if hasattr(component, "to"):
                    component.to("cpu")
                    self._offloaded_components[component_name] = torch.device("cpu")
                    print(f"Offloaded {component_name} to CPU")

    def bring_component_to_gpu(
        self,
        component_name: str,
        gpu_id: int = 0,
    ) -> None:
        """
        Bring a specific component back to GPU.

        Args:
            component_name: Name of the component to move.
            gpu_id: GPU device ID to move to.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline set")

        if not hasattr(self.pipeline, component_name):
            raise ValueError(f"Component {component_name} not found")

        component = getattr(self.pipeline, component_name)
        device = torch.device(f"cuda:{gpu_id}")

        if hasattr(component, "to"):
            component.to(device)
            self._offloaded_components[component_name] = device
            print(f"Moved {component_name} to GPU {gpu_id}")

    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status.

        Returns:
            Dictionary containing memory information.
        """
        status = {
            "offloaded_components": list(self._offloaded_components.keys()),
            "gpu_memory_allocated": 0,
            "gpu_memory_reserved": 0,
        }

        if torch.cuda.is_available():
            status["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            status["gpu_memory_reserved"] = torch.cuda.memory_reserved()

        return status

    def clear_cache(self) -> None:
        """Clear CUDA cache to free unused memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("MPS cache cleared")
