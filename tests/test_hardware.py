"""
Test hardware detection utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yukkavideo.utils.hardware import (
    detect_hardware,
    get_hardware_info,
    check_colab_environment,
    get_recommended_batch_size,
)


class TestHardwareDetection:
    """Tests for hardware detection functions."""

    def test_detect_hardware_returns_valid_device(self):
        """Test that detect_hardware returns a valid device string."""
        device = detect_hardware()
        assert device in ["cuda", "mps", "cpu"]

    def test_get_hardware_info_returns_dict(self):
        """Test that get_hardware_info returns a dictionary."""
        info = get_hardware_info()
        assert isinstance(info, dict)
        
        # Check required keys
        required_keys = [
            "platform",
            "gpu_available",
            "gpu_name",
            "gpu_count",
            "gpu_memory_gb",
            "cpu_count",
        ]
        for key in required_keys:
            assert key in info

    def test_get_hardware_info_platform_matches_device(self):
        """Test that platform matches detected device."""
        device = detect_hardware()
        info = get_hardware_info()
        
        if device == "cuda":
            assert info["platform"] == "cuda"
            assert info["gpu_available"] is True
        elif device == "mps":
            assert info["platform"] == "mps"
            assert info["gpu_available"] is True
        else:
            assert info["platform"] == "cpu"

    def test_check_colab_environment_returns_bool(self):
        """Test that check_colab_environment returns a boolean."""
        result = check_colab_environment()
        assert isinstance(result, bool)

    def test_get_recommended_batch_size_returns_positive_int(self):
        """Test that get_recommended_batch_size returns a positive integer."""
        batch_size = get_recommended_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size >= 1

    def test_get_recommended_batch_size_for_cuda(self):
        """Test batch size recommendation for CUDA devices."""
        batch_size = get_recommended_batch_size("cuda")
        assert isinstance(batch_size, int)
        assert batch_size >= 1

    def test_get_recommended_batch_size_for_cpu(self):
        """Test batch size recommendation for CPU."""
        batch_size = get_recommended_batch_size("cpu")
        assert batch_size == 1

    def test_get_recommended_batch_size_for_mps(self):
        """Test batch size recommendation for MPS."""
        batch_size = get_recommended_batch_size("mps")
        assert batch_size == 1
