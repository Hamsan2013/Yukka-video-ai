"""
Test configuration loading utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yukkavideo.utils.config_loader import (
    Config,
    load_config,
    save_config,
    get_default_config,
    merge_configs,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_config_values(self):
        """Test that default config has expected values."""
        config = get_default_config()
        
        assert config.model_name == "THUDM/CogVideoX-2b"
        assert config.model_type == "cogvideo"
        assert config.num_inference_steps == 50
        assert config.height == 480
        assert config.width == 720
        assert config.fps == 8
        assert config.enable_cpu_offload is True

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = get_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "height" in config_dict

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model_name": "test_model",
            "height": 1080,
            "width": 1920,
        }
        config = Config.from_dict(data)
        
        assert config.model_name == "test_model"
        assert config.height == 1080
        assert config.width == 1920
        # Default values should be preserved
        assert config.fps == 8


class TestLoadConfig:
    """Tests for loading configuration files."""

    def test_load_default_config_file(self):
        """Test loading the default config file."""
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        
        if config_path.exists():
            config = load_config(config_path)
            assert isinstance(config, Config)
            assert config.model_name == "THUDM/CogVideoX-2b"
        else:
            pytest.skip("Config file not found")

    def test_load_colab_t4_config(self):
        """Test loading the Colab T4 config file."""
        config_path = Path(__file__).parent.parent / "configs" / "colab_t4.yaml"
        
        if config_path.exists():
            config = load_config(config_path)
            assert isinstance(config, Config)
            assert config.enable_cpu_offload is True
        else:
            pytest.skip("Config file not found")

    def test_load_nonexistent_config_raises_error(self):
        """Test that loading nonexistent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestMergeConfigs:
    """Tests for merging configurations."""

    def test_merge_configs_overrides_values(self):
        """Test that merge_configs properly overrides values."""
        base = get_default_config()
        override = {"height": 1080, "width": 1920}
        
        merged = merge_configs(base, override)
        
        assert merged.height == 1080
        assert merged.width == 1920
        # Non-overridden values should remain
        assert merged.fps == base.fps

    def test_merge_configs_empty_override(self):
        """Test merging with empty override dict."""
        base = get_default_config()
        merged = merge_configs(base, {})
        
        assert merged.height == base.height
        assert merged.width == base.width
