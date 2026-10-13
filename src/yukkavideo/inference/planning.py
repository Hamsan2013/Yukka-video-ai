"""
Duration / frame-budget planning for long video jobs.

Pure functions, no model or GPU dependencies — fully unit-testable.

Rounding policy (documented):
- total_frames = round(target_duration_seconds * fps)
- Each segment targets ``segment_seconds`` (default 5.0), rounded to whole
  frames. The final segment absorbs all remaining frames so the sum of
  segment frame counts equals total_frames exactly.
- Individual adapters may snap a segment's frame count to a model-valid
  value (e.g. CogVideoX 8k+1). The controller records both the requested
  and actually-generated counts; assembly uses real clip lengths, so the
  final duration can deviate slightly from the request. The deviation is
  reported, never hidden.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SegmentSpec:
    index: int                  # 0-based segment number
    num_frames: int             # frames requested for this segment
    scene_index: Optional[int] = None

    @property
    def to_dict(self) -> dict:
        return {"index": self.index, "num_frames": self.num_frames,
                "scene_index": self.scene_index}


def compute_frame_budget(duration_seconds: float, fps: float) -> int:
    """Total frames needed for a target duration at a given fps (>= 1)."""
    if duration_seconds <= 0:
        raise ValueError(f"duration must be > 0, got {duration_seconds}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return max(1, int(round(float(duration_seconds) * float(fps))))


def plan_segments(
    total_frames: int,
    fps: float,
    segment_seconds: float = 5.0,
    adjust_num_frames=None,
    min_segment_frames: int = 1,
) -> List[SegmentSpec]:
    """Split a frame budget into sequential segments.

    Args:
        total_frames: overall frame budget (from compute_frame_budget).
        fps: frames per second (used only for segment frame math).
        segment_seconds: nominal per-segment length in seconds.
        adjust_num_frames: optional callable(int)->int supplied by the model
            adapter to snap frame counts to model-valid values. Applied to
            every segment except that the last segment always receives the
            exact remainder first, then is snapped (and allowed to be short
            if snapping would drop below min_segment_frames).
        min_segment_frames: never emit a smaller final segment than this;
            instead merge it into the previous segment.

    Returns:
        Ordered list of SegmentSpec whose num_frames sum is >= total_frames
        before snapping and as close as possible after snapping.
    """
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    seg_target = max(1, int(round(segment_seconds * fps)))

    specs: List[SegmentSpec] = []
    remaining = total_frames
    idx = 0
    while remaining > 0:
        take = min(seg_target, remaining)
        specs.append(SegmentSpec(index=idx, num_frames=take))
        remaining -= take
        idx += 1

    # Merge an undersized final segment into its predecessor.
    if len(specs) > 1 and specs[-1].num_frames < min_segment_frames:
        specs[-2].num_frames += specs[-1].num_frames
        specs.pop()

    # Apply model-specific snapping.
    if adjust_num_frames is not None:
        for s in specs:
            s.num_frames = int(adjust_num_frames(s.num_frames))

    return specs


def planned_duration(segments: List[SegmentSpec], fps: float) -> float:
    """Actual duration produced by a plan."""
    return sum(s.num_frames for s in segments) / float(fps)


@dataclass
class GenerationPlan:
    duration_seconds: float
    fps: float
    total_frames: int
    segments: List[SegmentSpec] = field(default_factory=list)

    @property
    def actual_duration_seconds(self) -> float:
        return planned_duration(self.segments, self.fps)

    @property
    def duration_deviation_seconds(self) -> float:
        return self.actual_duration_seconds - self.duration_seconds

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "actual_duration_seconds": round(self.actual_duration_seconds, 4),
            "segments": [s.to_dict for s in self.segments],
        }


def build_plan(
    duration_seconds: float,
    fps: float,
    segment_seconds: float = 5.0,
    adjust_num_frames=None,
) -> GenerationPlan:
    """Convenience: frame budget + segmentation in one call."""
    total = compute_frame_budget(duration_seconds, fps)
    segments = plan_segments(total, fps, segment_seconds, adjust_num_frames)
    return GenerationPlan(
        duration_seconds=duration_seconds,
        fps=fps,
        total_frames=total,
        segments=segments,
    )
