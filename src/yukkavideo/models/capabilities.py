"""
Model capability descriptors for Yukka Video AI.

Every model adapter must explicitly declare what it can do so the
generation controller never feeds, e.g., a reference image into a
text-to-video-only model.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelCapabilities:
    """Explicit capability report for a video-generation model adapter.

    Attributes:
        supports_text_to_video: Model can generate clips from text alone.
        supports_image_to_video: Model accepts a starting/reference image.
        supports_reference_conditioning: Model can be conditioned on extra
            character/style reference images beyond the start frame.
        supports_seed: Pipeline accepts a torch.Generator for reproducibility.
        supported_heights / supported_widths: If set, allowed pixel dimensions.
        dimension_divisor: Height/width must be divisible by this value
            (VAE spatial stride). 0 means unconstrained.
        frame_count_constraint: Description of model-specific restrictions on
            ``num_frames`` (e.g. CogVideoX requires 8k+1 frames).
        min_frames / max_frames: Allowed per-clip frame counts, if known.
        preferred_fps: Native fps the model was trained at. The exporter is
            free to retime the output to a different container fps.
        approx_vram_gb_fp16: Rough minimum VRAM (GB) needed at fp16 with
            CPU offload enabled. None if unknown.
        license: Known license identifier(s) for the weights.
        gated: True if the checkpoint requires accepting terms / token on
            Hugging Face before download.
        notes: Free-form caveats (e.g. continuity limitations).
    """

    supports_text_to_video: bool = False
    supports_image_to_video: bool = False
    supports_reference_conditioning: bool = False
    supports_seed: bool = True
    supported_heights: Optional[List[int]] = None
    supported_widths: Optional[List[int]] = None
    dimension_divisor: int = 8
    frame_count_constraint: str = ""
    min_frames: int = 1
    max_frames: int = 0  # 0 => no known upper bound
    preferred_fps: float = 8.0
    approx_vram_gb_fp16: Optional[float] = None
    license: str = "unknown"
    gated: bool = False
    notes: str = ""

    def supports_dimensions(self, height: int, width: int) -> bool:
        """Return True if the requested pixel dimensions are acceptable."""
        if height <= 0 or width <= 0:
            return False
        if self.supported_heights is not None and height not in self.supported_heights:
            return False
        if self.supported_widths is not None and width not in self.supported_widths:
            return False
        if self.dimension_divisor > 0:
            if height % self.dimension_divisor != 0 or width % self.dimension_divisor != 0:
                return False
        return True

    def supports_frame_count(self, num_frames: int) -> bool:
        """Return True if num_frames satisfies declared constraints.

        Note: exact modular constraints described in ``frame_count_constraint``
        are enforced by each adapter via ``adjust_num_frames``; this method
        only checks bounds.
        """
        if num_frames < self.min_frames:
            return False
        if self.max_frames and num_frames > self.max_frames:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "supports_text_to_video": self.supports_text_to_video,
            "supports_image_to_video": self.supports_image_to_video,
            "supports_reference_conditioning": self.supports_reference_conditioning,
            "supports_seed": self.supports_seed,
            "supported_heights": self.supported_heights,
            "supported_widths": self.supported_widths,
            "dimension_divisor": self.dimension_divisor,
            "frame_count_constraint": self.frame_count_constraint,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
            "preferred_fps": self.preferred_fps,
            "approx_vram_gb_fp16": self.approx_vram_gb_fp16,
            "license": self.license,
            "gated": self.gated,
            "notes": self.notes,
        }
