"""Shared fake model adapters for unit tests (no GPU, no downloads).

These subclass the REAL BaseModelAdapter so capability checks, frame snapping
and the generate_from_image refusal logic are exercised for real. Only the
diffusion call itself is faked with synthetic frames.
"""
from __future__ import annotations

import numpy as np

from yukkavideo.models.base import BaseModelAdapter
from yukkavideo.models.capabilities import ModelCapabilities


def _synthetic_frames(num_frames: int, height: int, width: int,
                      tint: tuple = (200, 60, 60)) -> list:
    """Deterministic tiny RGB frames (list of uint8 arrays)."""
    base = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for i in range(num_frames):
        base[i, :, :, 0] = (tint[0] + i * 3) % 256
        base[i, :, :, 1] = tint[1]
        base[i, :, :, 2] = (tint[2] + i * 5) % 256
        # Encode segment position so tests can verify continuity ordering.
        base[i, :8, :8] = i % 256
    return list(base)


class FakeT2VAdapter(BaseModelAdapter):
    """Text-to-video only adapter (like CogVideoX). Refuses image input."""

    def __init__(self, model_name="fake/t2v-only", **kw):
        super().__init__(model_name=model_name, **kw)
        self.t2v_calls = 0
        self.i2v_calls = 0
        self.load_calls = 0
        self.fail_times = set()          # segment seeds to fail on
        self.oom_times = set()           # seeds raising CUDA OOM text

    def load_model(self):
        self.load_calls += 1
        self._model_loaded = True
        return self

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_text_to_video=True,
            supports_image_to_video=False,
            dimension_divisor=8,
            frame_count_constraint="8k+1",
            min_frames=9, max_frames=241,
            preferred_fps=8.0,
            approx_vram_gb_fp16=10.0,
            license="test",
        )

    def adjust_num_frames(self, num_frames: int) -> int:
        k = max(1, round((num_frames - 1) / 8))
        return 8 * k + 1

    def generate_clip(self, prompt, num_frames=49, height=480, width=720,
                      seed=None, negative_prompt=None,
                      num_inference_steps=50, guidance_scale=6.0, **kw):
        self.t2v_calls += 1
        if seed in self.oom_times:
            raise RuntimeError("The following CUDA out of memory. "
                               "Tried to allocate 8.00 GiB")
        if seed in self.fail_times:
            raise ValueError("synthetic generation failure")
        n = self.adjust_num_frames(num_frames)
        return type("Out", (), {"frames": _synthetic_frames(n, height, width)})()

    def get_default_config(self):
        return {"model_name": self.model_name}


class FakeI2VAdapter(FakeT2VAdapter):
    """Adapter that genuinely supports image-to-video continuation (like LTX)."""

    def get_capabilities(self) -> ModelCapabilities:
        caps = super().get_capabilities()
        caps.supports_image_to_video = True
        return caps

    def generate_from_image(self, image_path, prompt, num_frames=49,
                            height=480, width=720, seed=None, **kw):
        from pathlib import Path
        p = Path(image_path)
        assert p.exists(), "reference image must exist"
        self.i2v_calls += 1
        n = self.adjust_num_frames(num_frames)
        return type("Out", (), {"frames": _synthetic_frames(
            n, height, width, tint=(60, 160, 60))})()
