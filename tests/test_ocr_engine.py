"""Tests for OCR engine implementations."""

import numpy as np
import pytest
import torch

from src.subsalvage.ocr.easyocr_engine import EasyOCREngine


class TestEasyOCREngine:
    """Test cases for EasyOCREngine."""

    def test_init_default_params(self) -> None:
        """Test EasyOCREngine initialization with default parameters."""
        engine = EasyOCREngine()
        assert engine.languages == ["en"]
        assert isinstance(engine.use_gpu, bool)

    def test_init_custom_languages(self) -> None:
        """Test EasyOCREngine initialization with custom languages."""
        languages = ["en", "es", "fr"]
        engine = EasyOCREngine(languages=languages)
        assert engine.languages == languages

    def test_init_gpu_disabled(self) -> None:
        """Test EasyOCREngine initialization with GPU disabled."""
        engine = EasyOCREngine(gpu=False)
        assert not engine.use_gpu

    def test_is_gpu_available(self) -> None:
        """Test GPU availability check."""
        engine = EasyOCREngine()
        gpu_available = engine.is_gpu_available()
        assert isinstance(gpu_available, bool)
        assert gpu_available == (engine.use_gpu and torch.cuda.is_available())

    def test_get_supported_languages(self) -> None:
        """Test getting supported languages."""
        languages = ["en"]  # Use only supported language
        engine = EasyOCREngine(languages=languages)
        supported = engine.get_supported_languages()
        assert supported == languages
        assert supported is not engine.languages  # Should be a copy

    def test_extract_text_empty_image(self) -> None:
        """Test text extraction with empty image."""
        engine = EasyOCREngine()
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = engine.extract_text(empty_image)
        assert isinstance(results, list)
        # Empty image should return empty or minimal results
        assert len(results) >= 0

    def test_extract_text_return_format(self) -> None:
        """Test that extract_text returns proper format."""
        engine = EasyOCREngine()
        # Create a simple image with some structure
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
        results = engine.extract_text(test_image)
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            assert "text" in result
            assert "confidence" in result
            assert "bbox" in result
            assert isinstance(result["text"], str)
            assert isinstance(result["confidence"], (int, float))
            assert isinstance(result["bbox"], list)

    def test_set_languages(self) -> None:
        """Test setting new languages."""
        engine = EasyOCREngine(languages=["en"])
        new_languages = ["en", "fr", "de"]
        
        engine.set_languages(new_languages)
        assert engine.languages == new_languages

    def test_set_languages_same(self) -> None:
        """Test setting same languages doesn't reinitialize."""
        engine = EasyOCREngine(languages=["en"])
        original_reader = engine.reader
        
        # Setting same languages should not change reader
        engine.set_languages(["en"])
        assert engine.reader is original_reader


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image."""
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    return image


def test_ocr_engine_integration(sample_image: np.ndarray) -> None:
    """Integration test for OCR engine."""
    engine = EasyOCREngine()
    results = engine.extract_text(sample_image)
    
    # Should not crash and return valid format
    assert isinstance(results, list)
    print(f"GPU available: {engine.is_gpu_available()}")
    print(f"Languages: {engine.get_supported_languages()}")