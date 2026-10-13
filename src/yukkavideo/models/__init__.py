"""Model adapters for Yukka Video AI."""

from .base import BaseModelAdapter, MemoryEstimate, UnsupportedFeatureError
from .capabilities import ModelCapabilities
from .cogvideo_adapter import CogVideoAdapter
from .ltx_adapter import LTXAdapter
from .wan_adapter import WanAdapter

ADAPTER_REGISTRY = {
    "cogvideo": CogVideoAdapter,
    "ltx": LTXAdapter,
    "wan": WanAdapter,
}


def detect_model_type(model_name: str) -> str:
    """Infer adapter key from a checkpoint name."""
    lowered = model_name.lower()
    if "cogvideo" in lowered:
        return "cogvideo"
    if "ltx" in lowered:
        return "ltx"
    if "wan" in lowered:
        return "wan"
    raise ValueError(
        f"Cannot infer model type for '{model_name}'. "
        f"Known families: {sorted(ADAPTER_REGISTRY)}"
    )


__all__ = [
    "BaseModelAdapter",
    "MemoryEstimate",
    "ModelCapabilities",
    "UnsupportedFeatureError",
    "CogVideoAdapter",
    "LTXAdapter",
    "WanAdapter",
    "ADAPTER_REGISTRY",
    "detect_model_type",
]
