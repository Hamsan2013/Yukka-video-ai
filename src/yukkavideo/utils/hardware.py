"""
Hardware Detection Utilities for Yukka Video AI

Provides functions to detect and report available hardware (GPU, CPU, etc.)
for optimal model configuration.
"""

import torch
from typing import Dict, Any, Optional


def detect_hardware() -> str:
    """
    Detect the best available hardware for inference.

    Returns:
        Device string ('cuda', 'mps', or 'cpu').

    Example:
        >>> device = detect_hardware()
        >>> print(f"Using device: {device}")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_hardware_info() -> Dict[str, Any]:
    """
    Get detailed information about available hardware.

    Returns:
        Dictionary containing hardware information.

    Example:
        >>> info = get_hardware_info()
        >>> print(f"GPU: {info['gpu_name']}")
        >>> print(f"VRAM: {info['gpu_memory_gb']:.2f} GB")
    """
    info: Dict[str, Any] = {
        "platform": "unknown",
        "gpu_available": False,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_memory_gb": 0.0,
        "gpu_memory_allocated_gb": 0.0,
        "cpu_count": 0,
    }

    # Detect CUDA GPU
    if torch.cuda.is_available():
        info["platform"] = "cuda"
        info["gpu_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
        )
        info["gpu_memory_allocated_gb"] = (
            torch.cuda.memory_allocated(0) / (1024**3)
        )

    # Detect Apple Silicon MPS
    elif torch.backends.mps.is_available():
        info["platform"] = "mps"
        info["gpu_available"] = True
        info["gpu_name"] = "Apple Silicon"
        # MPS doesn't expose memory info directly

    # CPU only
    else:
        info["platform"] = "cpu"
        info["gpu_available"] = False

    # CPU info
    try:
        import os
        info["cpu_count"] = os.cpu_count() or 0
    except Exception:
        info["cpu_count"] = 0

    return info


def check_colab_environment() -> bool:
    """
    Check if running in Google Colab.

    Returns:
        True if running in Colab, False otherwise.
    """
    try:
        import colab  # noqa: F401
        return True
    except ImportError:
        pass

    # Alternative detection method
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
            if "Google Cloud" in content or "Colab" in content:
                return True
    except Exception:
        pass

    return False


def get_recommended_batch_size(device: Optional[str] = None) -> int:
    """
    Get recommended batch size based on available hardware.

    Args:
        device: Device to check (auto-detected if None).

    Returns:
        Recommended batch size for video generation.
    """
    if device is None:
        device = detect_hardware()

    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem >= 40:  # A100 or similar
            return 4
        elif gpu_mem >= 24:  # RTX 3090/4090
            return 2
        elif gpu_mem >= 16:  # T4, RTX 3080
            return 1
        else:  # < 16GB
            return 1

    elif device == "mps":
        return 1

    else:  # CPU
        return 1


def print_hardware_summary() -> None:
    """Print a summary of available hardware."""
    info = get_hardware_info()

    print("=" * 50)
    print("Hardware Summary")
    print("=" * 50)
    print(f"Platform: {info['platform'].upper()}")
    print(f"GPU Available: {info['gpu_available']}")

    if info["gpu_available"]:
        print(f"GPU Name: {info['gpu_name']}")
        print(f"GPU Count: {info['gpu_count']}")
        print(f"Total VRAM: {info['gpu_memory_gb']:.2f} GB")
        if info["gpu_memory_allocated_gb"] > 0:
            print(f"Allocated VRAM: {info['gpu_memory_allocated_gb']:.2f} GB")

    print(f"CPU Cores: {info['cpu_count']}")

    if check_colab_environment():
        print("Running in: Google Colab")

    print("=" * 50)
