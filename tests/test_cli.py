"""
Test CLI functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yukkavideo.cli.main import (
    create_parser,
    main,
)


class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_create_parser_returns_parser(self):
        """Test that create_parser returns an ArgumentParser."""
        parser = create_parser()
        
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_parser_has_generate_subcommand(self):
        """Test that parser has 'generate' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["generate", "test prompt"])
        
        assert args.command == "generate"
        assert args.prompt == "test prompt"

    def test_parser_has_info_subcommand(self):
        """Test that parser has 'info' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["info"])
        
        assert args.command == "info"

    def test_generate_with_custom_output(self):
        """Test generate command with custom output path."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "-o", "custom_output.mp4"
        ])
        
        assert args.output == "custom_output.mp4"

    def test_generate_with_custom_model(self):
        """Test generate command with custom model."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "-m", "THUDM/CogVideoX-2b"
        ])
        
        assert args.model == "THUDM/CogVideoX-2b"

    def test_generate_with_steps(self):
        """Test generate command with custom steps."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "-s", "100"
        ])
        
        assert args.steps == 100

    def test_generate_with_dimensions(self):
        """Test generate command with custom dimensions."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "--height", "720",
            "--width", "1280"
        ])
        
        assert args.height == 720
        assert args.width == 1280

    def test_generate_with_no_offload(self):
        """Test generate command with --no-offload flag."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "--no-offload"
        ])
        
        assert args.no_offload is True

    def test_generate_with_negative_prompt(self):
        """Test generate command with negative prompt."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "--negative-prompt", "blurry, low quality"
        ])
        
        assert args.negative_prompt == "blurry, low quality"

    def test_generate_with_verbose(self):
        """Test generate command with verbose flag."""
        parser = create_parser()
        args = parser.parse_args([
            "generate", "test prompt",
            "-v"
        ])
        
        assert args.verbose is True


class TestMainFunction:
    """Tests for main CLI function."""

    def test_main_with_no_args_shows_help(self, capsys):
        """Test that main with no args shows help."""
        result = main([])
        
        assert result == 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "Yukka" in captured.out

    def test_main_with_info_command(self, capsys):
        """Test main with info command."""
        result = main(["info"])
        
        # Should return 0 and print hardware info
        assert result == 0

    def test_main_with_invalid_command(self, capsys):
        """Test main with invalid command."""
        result = main(["invalid_command"])
        
        # Should show help and return non-zero
        assert result == 1
