"""
Configuration Loader for Yukka Video AI

Provides utilities for loading and managing YAML configuration files.
"""

import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration data class."""
    model_name: str = "THUDM/CogVideoX-2b"
    model_type: str = "cogvideo"
    device: str = "auto"
    dtype: str = "float16"
    num_inference_steps: int = 50
    height: int = 480
    width: int = 720
    fps: int = 8
    num_frames: int = 49
    guidance_scale: float = 6.0
    enable_cpu_offload: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    seed: Optional[int] = 42
    output_dir: str = "./outputs"
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "dtype": self.dtype,
            "num_inference_steps": self.num_inference_steps,
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "guidance_scale": self.guidance_scale,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_vae_tiling": self.enable_vae_tiling,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with loaded settings.

    Example:
        >>> config = load_config("configs/colab_t4.yaml")
        >>> print(config.model_name)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("Please install PyYAML: pip install pyyaml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Handle nested config structure
    if data and "config" in data:
        data = data["config"]

    return Config.from_dict(data or {})


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Config object to save.
        config_path: Path to save the configuration file.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("Please install PyYAML: pip install pyyaml")

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump({"config": config.to_dict()}, f, default_flow_style=False)

    print(f"Config saved to: {config_path}")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def merge_configs(base: Config, override: Dict[str, Any]) -> Config:
    """
    Merge base config with override values.

    Args:
        base: Base Config object.
        override: Dictionary of values to override.

    Returns:
        New Config object with merged values.
    """
    merged = base.to_dict()
    merged.update(override)
    return Config.from_dict(merged)


def get_config_from_env(prefix: str = "YUKKA_") -> Config:
    """
    Load configuration from environment variables.

    Args:
        prefix: Prefix for environment variables.

    Returns:
        Config object populated from environment.
    """
    env_config: Dict[str, Any] = {}

    config_fields = Config.__dataclass_fields__

    for field_name in config_fields:
        env_var = f"{prefix}{field_name.upper()}"
        value = os.environ.get(env_var)

        if value is not None:
            field_type = config_fields[field_name].type

            # Type conversion
            if field_type == bool:
                env_config[field_name] = value.lower() in ("true", "1", "yes")
            elif field_type == int:
                env_config[field_name] = int(value)
            elif field_type == float:
                env_config[field_name] = float(value)
            elif field_type == Optional[int]:
                env_config[field_name] = int(value) if value else None
            else:
                env_config[field_name] = value

    base_config = get_default_config()
    return merge_configs(base_config, env_config)
