"""
Training Utilities for Yukka Video AI

Provides utility functions for preparing training data and optimizers.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def prepare_dataset(
    data_dir: str,
    video_extension: str = ".mp4",
    caption_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare a dataset for training from a directory of videos.

    Args:
        data_dir: Directory containing video files.
        video_extension: Extension of video files to look for.
        caption_file: Optional path to a file containing captions.

    Returns:
        List of dictionaries with video paths and captions.

    Note:
        This is a stub function for future implementation.
        Full training support will be added in a future release.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all video files
    video_files = list(data_path.glob(f"*{video_extension}"))

    if len(video_files) == 0:
        print(f"Warning: No video files found in {data_dir}")
        return []

    # Load captions if provided
    captions: Dict[str, str] = {}
    if caption_file:
        caption_path = Path(caption_file)
        if caption_path.exists():
            with open(caption_path, "r") as f:
                for line in f:
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        filename, caption = parts
                        captions[filename.strip()] = caption.strip()

    # Create dataset entries
    dataset = []
    for video_path in video_files:
        entry = {
            "video_path": str(video_path),
            "caption": captions.get(video_path.name, ""),
        }
        dataset.append(entry)

    print(f"Prepared {len(dataset)} video entries")
    return dataset


def create_optimizer(
    model_parameters: Any,
    learning_rate: float = 1e-4,
    optimizer_type: str = "adamw",
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-8,
):
    """
    Create an optimizer for training.

    Args:
        model_parameters: Model parameters to optimize.
        learning_rate: Learning rate.
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd').
        weight_decay: Weight decay for regularization.
        betas: Beta parameters for Adam-based optimizers.
        epsilon: Epsilon for numerical stability.

    Returns:
        Configured optimizer instance.

    Raises:
        ImportError: If required packages are not installed.
        ValueError: If unknown optimizer type is specified.

    Note:
        This function requires torch and optionally bitsandbytes.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for training utilities")

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=epsilon,
        )
    elif optimizer_type == "adam":
        from torch.optim import Adam
        optimizer = Adam(
            model_parameters,
            lr=learning_rate,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        from torch.optim import SGD
        optimizer = SGD(
            model_parameters,
            lr=learning_rate,
            momentum=betas[0],
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for 8-bit AdamW. "
                "Install with: pip install bitsandbytes"
            )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    print(f"Created {optimizer_type} optimizer with lr={learning_rate}")
    return optimizer


def get_lr_scheduler(
    optimizer: Any,
    num_warmup_steps: int = 500,
    num_training_steps: int = 10000,
    scheduler_type: str = "cosine",
):
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        scheduler_type: Type of scheduler ('linear', 'cosine', 'constant').

    Returns:
        Configured learning rate scheduler.

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import get_scheduler
    except ImportError:
        raise ImportError(
            "transformers is required for learning rate schedulers. "
            "Install with: pip install transformers"
        )

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"Created {scheduler_type} scheduler")
    return scheduler


def calculate_steps(
    num_samples: int,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
) -> Tuple[int, int]:
    """
    Calculate training steps.

    Args:
        num_samples: Number of training samples.
        num_epochs: Number of epochs.
        batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Tuple of (num_warmup_steps, num_training_steps).
    """
    num_training_steps = (
        num_samples * num_epochs // (batch_size * gradient_accumulation_steps)
    )
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

    return num_warmup_steps, num_training_steps
