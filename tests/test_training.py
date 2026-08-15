"""Test training utilities."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yukkavideo.training.lora_config import LoRAConfig, get_lora_config


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_lora_config(self):
        """Test default LoRA configuration values."""
        config = LoRAConfig()
        
        assert config.r == 8
        assert config.lora_alpha == 32
        assert config.learning_rate == 1e-4

    def test_lora_config_to_dict(self):
        """Test converting LoRAConfig to dictionary."""
        config = LoRAConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "r" in config_dict

    def test_get_lora_config_for_cogvideo(self):
        """Test getting LoRA config for CogVideo."""
        config = get_lora_config("cogvideo")
        
        assert config.target_modules is not None
        assert len(config.target_modules) > 0

    def test_get_lora_config_custom_rank(self):
        """Test getting LoRA config with custom rank."""
        config = get_lora_config("cogvideo", rank=16, alpha=64)
        
        assert config.r == 16
        assert config.lora_alpha == 64
