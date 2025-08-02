"""Tests for CLI functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from main import app


class TestCLI:
    """Test cases for CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_extract_help(self) -> None:
        """Test extract command help."""
        result = self.runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract hardcoded subtitles" in result.stdout

    def test_validate_help(self) -> None:
        """Test validate command help."""
        result = self.runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate system requirements" in result.stdout

    def test_main_help(self) -> None:
        """Test main app help."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "SubSalvage" in result.stdout
        assert "extract" in result.stdout
        assert "validate" in result.stdout

    def test_extract_nonexistent_file(self) -> None:
        """Test extract command with non-existent file."""
        result = self.runner.invoke(app, ["extract", "nonexistent.mp4"])
        assert result.exit_code != 0

    def test_validate_command(self) -> None:
        """Test validate command execution."""
        result = self.runner.invoke(app, ["validate"])
        assert result.exit_code == 0
        assert "Validating SubSalvage installation" in result.stdout

    @pytest.mark.parametrize("gpu_flag", ["--gpu", "--no-gpu"])
    def test_extract_gpu_options(self, temp_video_file: Path, gpu_flag: str) -> None:
        """Test extract command with GPU options."""
        result = self.runner.invoke(app, ["extract", str(temp_video_file), gpu_flag])
        assert result.exit_code == 0
        
        if gpu_flag == "--gpu":
            assert "GPU: Enabled" in result.stdout
        else:
            assert "GPU: Disabled" in result.stdout

    def test_extract_time_range(self, temp_video_file: Path) -> None:
        """Test extract command with time range options."""
        result = self.runner.invoke(app, [
            "extract", str(temp_video_file), 
            "--start", "10.5", 
            "--end", "60.0"
        ])
        assert result.exit_code == 0
        assert "10.5s - 60.0s" in result.stdout

    def test_extract_custom_output(self, temp_video_file: Path) -> None:
        """Test extract command with custom output path."""
        result = self.runner.invoke(app, [
            "extract", str(temp_video_file), 
            "--output", "custom.srt"
        ])
        assert result.exit_code == 0
        assert "custom.srt" in result.stdout