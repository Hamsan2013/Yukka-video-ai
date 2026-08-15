"""
Memory Tracker for Yukka Video AI

Provides utilities for monitoring GPU and CPU memory usage during inference.
"""

import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot."""
    timestamp: str
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_max_allocated_mb: float
    cpu_used_mb: float


class MemoryTracker:
    """
    Tracks memory usage during video generation.

    This class provides methods to monitor GPU and CPU memory consumption,
    take snapshots, and log memory patterns for debugging and optimization.
    """

    def __init__(self):
        """Initialize the Memory Tracker."""
        self._snapshots: List[MemorySnapshot] = []
        self._tracking_enabled = True

    def get_gpu_memory(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary containing GPU memory statistics in MB.
        """
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
            }

        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }

    def get_cpu_memory(self) -> Dict[str, float]:
        """
        Get current CPU memory usage.

        Returns:
            Dictionary containing CPU memory statistics in MB.
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "used_mb": memory_info.rss / (1024 ** 2),
            }
        except ImportError:
            # Fallback without psutil
            return {"used_mb": 0.0}

    def take_snapshot(self, label: Optional[str] = None) -> MemorySnapshot:
        """
        Take a memory usage snapshot.

        Args:
            label: Optional label for the snapshot.

        Returns:
            MemorySnapshot object with current memory stats.
        """
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()

        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            gpu_allocated_mb=gpu_mem["allocated_mb"],
            gpu_reserved_mb=gpu_mem["reserved_mb"],
            gpu_max_allocated_mb=gpu_mem["max_allocated_mb"],
            cpu_used_mb=cpu_mem["used_mb"],
        )

        if self._tracking_enabled:
            self._snapshots.append(snapshot)

        return snapshot

    def log_snapshot(self, label: Optional[str] = None) -> None:
        """
        Take and print a memory snapshot.

        Args:
            label: Optional label for the snapshot.
        """
        snapshot = self.take_snapshot(label)
        print(f"[Memory @ {snapshot.timestamp}]")
        print(f"  GPU Allocated: {snapshot.gpu_allocated_mb:.1f} MB")
        print(f"  GPU Reserved:  {snapshot.gpu_reserved_mb:.1f} MB")
        print(f"  GPU Max Alloc: {snapshot.gpu_max_allocated_mb:.1f} MB")
        print(f"  CPU Used:      {snapshot.cpu_used_mb:.1f} MB")

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("GPU peak memory stats reset")

    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all recorded snapshots."""
        return self._snapshots.copy()

    def clear_snapshots(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()

    def enable_tracking(self) -> None:
        """Enable memory tracking."""
        self._tracking_enabled = True

    def disable_tracking(self) -> None:
        """Disable memory tracking."""
        self._tracking_enabled = False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of memory usage from all snapshots.

        Returns:
            Dictionary containing memory usage statistics.
        """
        if not self._snapshots:
            return {"error": "No snapshots available"}

        gpu_allocated = [s.gpu_allocated_mb for s in self._snapshots]
        gpu_reserved = [s.gpu_reserved_mb for s in self._snapshots]
        cpu_used = [s.cpu_used_mb for s in self._snapshots]

        return {
            "num_snapshots": len(self._snapshots),
            "gpu_allocated_avg_mb": sum(gpu_allocated) / len(gpu_allocated),
            "gpu_allocated_max_mb": max(gpu_allocated),
            "gpu_allocated_min_mb": min(gpu_allocated),
            "gpu_reserved_avg_mb": sum(gpu_reserved) / len(gpu_reserved),
            "gpu_reserved_max_mb": max(gpu_reserved),
            "cpu_used_avg_mb": sum(cpu_used) / len(cpu_used),
            "cpu_used_max_mb": max(cpu_used),
        }
