"""
Continuity helpers: final-frame extraction and clip validation.

All functions here are model-agnostic and testable without a GPU: they work
on frame arrays (numpy) or on exported clip files via imageio/ffmpeg.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np


class ClipValidationError(RuntimeError):
    """Raised when a generated clip file is missing or unreadable."""


def frames_to_uint8(frames: Any) -> np.ndarray:
    """Normalize pipeline output frames to [N, H, W, 3] uint8 array."""
    if hasattr(frames, "frames"):
        frames = frames.frames
    if isinstance(frames, (list, tuple)):
        arrs = []
        for f in frames:
            if hasattr(f, "cpu"):
                f = f.cpu().numpy()
            elif hasattr(f, "array"):
                f = np.array(f)
            arrs.append(np.asarray(f))
        frames = np.stack(arrs) if arrs else np.zeros((0,), dtype=np.uint8)
    arr = np.asarray(frames.cpu().numpy() if hasattr(frames, "cpu") else frames)

    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ClipValidationError(f"Unexpected frame array shape {arr.shape}")

    # Move channel dim last if needed ([N, C, H, W] -> [N, H, W, C]).
    if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.dtype != np.uint8:
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx <= 1.5 and mn >= -0.1:
            arr = (np.clip(arr, 0, 1) * 255.0)
        elif mn < 0:  # e.g. [-1, 1] normalization
            arr = ((np.clip(arr, -1, 1) + 1.0) / 2.0 * 255.0)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def extract_final_frame(
    frames: Any,
    output_path: Union[str, Path],
) -> str:
    """Save the last frame of a clip as a PNG reference image.

    Raises ClipValidationError if there are no usable frames or the write
    fails — the caller must treat that as a continuity degradation, not a
    silent success.
    """
    out = Path(output_path)
    arr = frames_to_uint8(frames)
    if arr.shape[0] == 0:
        raise ClipValidationError("Cannot extract final frame: clip has zero frames")
    try:
        from PIL import Image

        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr[-1]).save(out, format="PNG")
    except Exception as exc:
        raise ClipValidationError(f"Failed to write reference frame {out}: {exc}") from exc
    if not out.exists() or out.stat().st_size == 0:
        raise ClipValidationError(f"Reference frame file invalid: {out}")
    return str(out)


def validate_clip_file(path: Union[str, Path], min_frames: int = 1) -> int:
    """Verify a clip file exists and contains readable frames.

    Returns the frame count. Raises ClipValidationError otherwise.
    """
    p = Path(path)
    if not p.exists():
        raise ClipValidationError(f"Clip file does not exist: {p}")
    if p.stat().st_size == 0:
        raise ClipValidationError(f"Clip file is empty: {p}")
    # Use imageio (ffmpeg backend) to decode and count frames.
    try:
        import imageio

        count = 0
        reader = None
        try:
            reader = imageio.get_reader(str(p))
            for _frame in reader:
                count += 1
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
    except Exception as exc:
        raise ClipValidationError(f"Clip file unreadable: {p} ({exc})") from exc
    if count < min_frames:
        raise ClipValidationError(
            f"Clip {p} has {count} readable frames, expected >= {min_frames}"
        )
    return count
