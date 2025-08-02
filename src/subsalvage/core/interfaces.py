"""Core interfaces for SubSalvage using Strategy pattern."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Extract text from an image.

        Args:
            image: Input image as numpy array

        Returns:
            List of text detections with bounding boxes and confidence scores
        """
        pass

    @abstractmethod
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        pass


class Upscaler(ABC):
    """Abstract base class for image upscaling engines."""

    @abstractmethod
    def upscale(self, image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """Upscale an image by the given factor.

        Args:
            image: Input image as numpy array
            scale_factor: Scaling factor (2x, 4x, etc.)

        Returns:
            Upscaled image as numpy array
        """
        pass

    @abstractmethod
    def get_supported_scales(self) -> list[int]:
        """Get list of supported scaling factors."""
        pass


class SubtitleExtractor(ABC):
    """Abstract base class for subtitle extraction pipelines."""

    @abstractmethod
    def extract(
        self,
        video_path: str,
        output_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> bool:
        """Extract subtitles from video and save as SRT.

        Args:
            video_path: Path to input video file
            output_path: Path to output SRT file
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: -1.0 for full video)

        Returns:
            True if extraction successful, False otherwise
        """
        pass
