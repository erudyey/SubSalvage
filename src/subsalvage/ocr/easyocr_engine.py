"""EasyOCR implementation of the OCR engine interface."""

import logging
from typing import Any

import cv2
import easyocr
import numpy as np
import torch

from ..core.interfaces import OCREngine

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """EasyOCR-based text recognition engine with GPU support."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        """Initialize EasyOCR engine.

        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU acceleration if available
        """
        if languages is None:
            languages = ["en"]

        self.languages = languages
        self.use_gpu = gpu and torch.cuda.is_available()

        try:
            self.reader = easyocr.Reader(
                lang_list=self.languages, gpu=self.use_gpu, verbose=False
            )
            logger.info(f"EasyOCR initialized with GPU: {self.use_gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def extract_text(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Extract text from image using EasyOCR.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of dictionaries containing:
                - text: Detected text string
                - confidence: Confidence score (0-1)
                - bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        try:
            # EasyOCR expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Perform OCR
            results = self.reader.readtext(image_rgb)

            # Convert to standard format
            detections = []
            for bbox_coords, text, confidence in results:
                detection = {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": bbox_coords,
                }
                detections.append(detection)

            logger.debug(f"Extracted {len(detections)} text detections")
            return detections

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available and being used."""
        return self.use_gpu and torch.cuda.is_available()

    def get_supported_languages(self) -> list[str]:
        """Get list of currently supported languages."""
        return self.languages.copy()

    def set_languages(self, languages: list[str]) -> None:
        """Update supported languages and reinitialize reader.

        Args:
            languages: New list of language codes
        """
        if languages != self.languages:
            self.languages = languages
            try:
                self.reader = easyocr.Reader(
                    lang_list=self.languages, gpu=self.use_gpu, verbose=False
                )
                logger.info(f"EasyOCR reinitialized with languages: {languages}")
            except Exception as e:
                logger.error(f"Failed to reinitialize EasyOCR: {e}")
                raise
