"""
Video Exporter for Yukka Video AI

Handles exporting generated video frames to MP4 files using imageio-ffmpeg.
"""

from typing import Any, List, Union
from pathlib import Path
import numpy as np


class VideoExporter:
    """
    Utility class for exporting video frames to MP4 files.

    This class handles the conversion of generated frames (as numpy arrays
    or PIL images) into standard video formats suitable for playback.
    """

    def __init__(self, backend: str = "imageio"):
        """
        Initialize the VideoExporter.

        Args:
            backend: Backend to use for video export ('imageio' or 'opencv').
        """
        self.backend = backend

    def export(
        self,
        frames: Any,
        output_path: Union[str, Path],
        fps: int = 8,
        codec: str = "libx264",
        pixel_format: str = "yuv420p",
    ) -> str:
        """
        Export frames to a video file.

        Args:
            frames: Video frames as numpy array, list of arrays, or diffusers output.
            output_path: Path to save the output video.
            fps: Frames per second.
            codec: Video codec to use.
            pixel_format: Pixel format for the output.

        Returns:
            Path to the saved video file.
        """
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "imageio":
            return self._export_with_imageio(
                frames=frames,
                output_path=output_path,
                fps=fps,
                codec=codec,
                pixel_format=pixel_format,
            )
        elif self.backend == "opencv":
            return self._export_with_opencv(
                frames=frames,
                output_path=output_path,
                fps=fps,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _export_with_imageio(
        self,
        frames: Any,
        output_path: Path,
        fps: int = 8,
        codec: str = "libx264",
        pixel_format: str = "yuv420p",
    ) -> str:
        """
        Export frames using imageio-ffmpeg.

        Args:
            frames: Video frames to export.
            output_path: Path to save the video.
            fps: Frames per second.
            codec: Video codec.
            pixel_format: Pixel format.

        Returns:
            Path to the saved video file.
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "Please install imageio with ffmpeg support: pip install imageio[ffmpeg]"
            )

        # Convert frames to numpy array if needed
        frames_array = self._normalize_frames(frames)

        # Write video using imageio
        with imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=codec,
            macro_block_size=None,  # Avoid padding issues
            ffmpeg_params=["-pix_fmt", pixel_format],
        ) as writer:
            for frame in frames_array:
                writer.append_data(frame)

        print(f"Video saved to: {output_path}")
        return str(output_path)

    def _export_with_opencv(
        self,
        frames: Any,
        output_path: Path,
        fps: int = 8,
    ) -> str:
        """
        Export frames using OpenCV.

        Args:
            frames: Video frames to export.
            output_path: Path to save the video.
            fps: Frames per second.

        Returns:
            Path to the saved video file.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("Please install opencv-python: pip install opencv-python")

        frames_array = self._normalize_frames(frames)

        if len(frames_array) == 0:
            raise ValueError("No frames to export")

        height, width = frames_array[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames_array:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)

        out.release()
        print(f"Video saved to: {output_path}")
        return str(output_path)

    def _normalize_frames(self, frames: Any) -> List[np.ndarray]:
        """
        Normalize frames to a list of numpy arrays.

        Handles various input formats including:
        - numpy arrays (batched or unbatched)
        - lists of arrays
        - PIL images
        - diffusers output objects

        Args:
            frames: Input frames in various formats.

        Returns:
            List of numpy arrays representing frames.
        """
        # Handle diffusers BaseOutput objects
        if hasattr(frames, "frames"):
            frames = frames.frames

        # Handle single numpy array (batched)
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:  # Batch of frames [B, H, W, C]
                return [frames[i] for i in range(frames.shape[0])]
            elif frames.ndim == 3:  # Single frame [H, W, C]
                return [frames]
            else:
                raise ValueError(f"Unexpected array shape: {frames.shape}")

        # Handle torch tensors
        import torch
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
            if frames.ndim == 4:  # Batch of frames [B, C, H, W] or [B, H, W, C]
                if frames.shape[1] <= 4:  # Likely [B, C, H, W]
                    frames = np.transpose(frames, (0, 2, 3, 1))
                return [frames[i] for i in range(frames.shape[0])]
            elif frames.ndim == 3:  # Single frame
                return [frames]
            else:
                raise ValueError(f"Unexpected tensor shape: {frames.shape}")

        # Handle list of frames
        if isinstance(frames, (list, tuple)):
            normalized = []
            for frame in frames:
                if hasattr(frame, "numpy"):  # Torch tensor
                    frame = frame.cpu().numpy()
                elif hasattr(frame, "array"):  # PIL Image
                    frame = np.array(frame)
                normalized.append(frame)
            return normalized

        raise ValueError(f"Unsupported frames type: {type(frames)}")
