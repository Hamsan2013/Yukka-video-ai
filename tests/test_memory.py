"""
Test memory management utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_memory_tracker_initialization(self):
        """Test MemoryTracker initialization."""
        from yukkavideo.memory.memory_tracker import MemoryTracker
        
        tracker = MemoryTracker()
        assert tracker is not None
        assert tracker._tracking_enabled is True

    def test_get_gpu_memory_returns_dict(self):
        """Test that get_gpu_memory returns a dictionary."""
        from yukkavideo.memory.memory_tracker import MemoryTracker
        
        tracker = MemoryTracker()
        gpu_mem = tracker.get_gpu_memory()
        
        assert isinstance(gpu_mem, dict)
        assert "allocated_mb" in gpu_mem

    def test_take_snapshot_returns_snapshot(self):
        """Test taking a memory snapshot."""
        from yukkavideo.memory.memory_tracker import MemoryTracker
        
        tracker = MemoryTracker()
        snapshot = tracker.take_snapshot()
        
        assert snapshot is not None
        assert hasattr(snapshot, "gpu_allocated_mb")

    def test_get_snapshots_returns_list(self):
        """Test getting all snapshots."""
        from yukkavideo.memory.memory_tracker import MemoryTracker
        
        tracker = MemoryTracker()
        tracker.take_snapshot()
        
        snapshots = tracker.get_snapshots()
        assert isinstance(snapshots, list)
        assert len(snapshots) >= 1


class TestVAEOptimizer:
    """Tests for VAEOptimizer class."""

    def test_vae_optimizer_initialization(self):
        """Test VAEOptimizer initialization."""
        from yukkavideo.memory.vae_optimizer import VAEOptimizer
        
        optimizer = VAEOptimizer()
        assert optimizer is not None
        assert optimizer._slicing_enabled is False
        assert optimizer._tiling_enabled is False

    def test_get_optimization_status(self):
        """Test getting optimization status."""
        from yukkavideo.memory.vae_optimizer import VAEOptimizer
        
        optimizer = VAEOptimizer()
        status = optimizer.get_optimization_status()
        
        assert isinstance(status, dict)
        assert "slicing_enabled" in status
        assert "tiling_enabled" in status


class TestCPUOffloadManager:
    """Tests for CPUOffloadManager class."""

    def test_cpu_offload_manager_initialization(self):
        """Test CPUOffloadManager initialization."""
        from yukkavideo.memory.cpu_offload import CPUOffloadManager
        
        manager = CPUOffloadManager()
        assert manager is not None
        assert manager.pipeline is None

    def test_get_memory_status_without_pipeline(self):
        """Test getting memory status without pipeline."""
        from yukkavideo.memory.cpu_offload import CPUOffloadManager
        
        manager = CPUOffloadManager()
        status = manager.get_memory_status()
        
        assert isinstance(status, dict)
        assert "offloaded_components" in status
