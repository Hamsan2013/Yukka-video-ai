"""
Yukka Video AI - Open Source Text-to-Video Generation System
"""

__version__ = "0.1.0"
__author__ = "Yukka Video AI Team"

from .inference.pipeline import YukkaVideoPipeline
from .models.base import BaseModelAdapter
from .utils.hardware import detect_hardware

__all__ = [
    "YukkaVideoPipeline",
    "BaseModelAdapter",
    "detect_hardware",
]
