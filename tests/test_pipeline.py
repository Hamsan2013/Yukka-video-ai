"""
Test pipeline imports and basic functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPipelineImports:
    """Tests for pipeline module imports."""

    def test_import_yukka_video_pipeline(self):
        """Test that YukkaVideoPipeline can be imported."""
        from yukkavideo.inference.pipeline import YukkaVideoPipeline
        assert YukkaVideoPipeline is not None

    def test_import_video_exporter(self):
        """Test that VideoExporter can be imported."""
        from yukkavideo.inference.video_exporter import VideoExporter
        assert VideoExporter is not None

    def test_import_cogvideo_adapter(self):
        """Test that CogVideoAdapter can be imported."""
        from yukkavideo.models.cogvideo_adapter import CogVideoAdapter
        assert CogVideoAdapter is not None

    def test_import_ltx_adapter(self):
        """Test that LTXAdapter can be imported."""
        from yukkavideo.models.ltx_adapter import LTXAdapter
        assert LTXAdapter is not None

    def test_import_base_model_adapter(self):
        """Test that BaseModelAdapter can be imported."""
        from yukkavideo.models.base import BaseModelAdapter
        assert BaseModelAdapter is not None


class TestModelAdapters:
    """Tests for model adapter classes."""

    def test_cogvideo_adapter_initialization(self):
        """Test CogVideoAdapter initialization."""
        from yukkavideo.models.cogvideo_adapter import CogVideoAdapter
        
        adapter = CogVideoAdapter()
        assert adapter.model_name == "THUDM/CogVideoX-2b"
        assert adapter.dtype is not None

    def test_ltx_adapter_initialization(self):
        """Test LTXAdapter initialization."""
        from yukkavideo.models.ltx_adapter import LTXAdapter
        
        adapter = LTXAdapter()
        assert adapter.model_name == "Lightricks/LTX-Video"
        assert adapter.dtype is not None

    def test_cogvideo_adapter_get_default_config(self):
        """Test CogVideoAdapter get_default_config method."""
        from yukkavideo.models.cogvideo_adapter import CogVideoAdapter
        
        adapter = CogVideoAdapter()
        config = adapter.get_default_config()
        
        assert isinstance(config, dict)
        assert "model_name" in config
        assert config["model_name"] == "THUDM/CogVideoX-2b"

    def test_ltx_adapter_get_default_config(self):
        """Test LTXAdapter get_default_config method."""
        from yukkavideo.models.ltx_adapter import LTXAdapter
        
        adapter = LTXAdapter()
        config = adapter.get_default_config()
        
        assert isinstance(config, dict)
        assert "model_name" in config
        assert config["model_name"] == "Lightricks/LTX-Video"


class TestVideoExporter:
    """Tests for VideoExporter class."""

    def test_video_exporter_initialization(self):
        """Test VideoExporter initialization."""
        from yukkavideo.inference.video_exporter import VideoExporter
        
        exporter = VideoExporter()
        assert exporter.backend == "imageio"

    def test_video_exporter_with_backend(self):
        """Test VideoExporter initialization with custom backend."""
        from yukkavideo.inference.video_exporter import VideoExporter
        
        exporter = VideoExporter(backend="opencv")
        assert exporter.backend == "opencv"


class TestPipelineClassStructure:
    """Tests for YukkaVideoPipeline class structure."""

    def test_pipeline_has_from_pretrained(self):
        """Test that YukkaVideoPipeline has from_pretrained method."""
        from yukkavideo.inference.pipeline import YukkaVideoPipeline
        
        assert hasattr(YukkaVideoPipeline, "from_pretrained")
        assert callable(getattr(YukkaVideoPipeline, "from_pretrained"))

    def test_pipeline_has_enable_cpu_offload(self):
        """Test that YukkaVideoPipeline has enable_model_cpu_offload method."""
        from yukkavideo.inference.pipeline import YukkaVideoPipeline
        
        # Check class has the method
        assert hasattr(YukkaVideoPipeline, "enable_model_cpu_offload")

    def test_pipeline_has_export_to_video(self):
        """Test that YukkaVideoPipeline has export_to_video method."""
        from yukkavideo.inference.pipeline import YukkaVideoPipeline
        
        assert hasattr(YukkaVideoPipeline, "export_to_video")

    def test_pipeline_supported_models(self):
        """Test that YukkaVideoPipeline has SUPPORTED_MODELS attribute."""
        from yukkavideo.inference.pipeline import YukkaVideoPipeline
        
        assert hasattr(YukkaVideoPipeline, "SUPPORTED_MODELS")
        models = YukkaVideoPipeline.SUPPORTED_MODELS
        assert "cogvideo" in models
        assert "ltx" in models
