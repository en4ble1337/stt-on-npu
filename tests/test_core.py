
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.stt_npu.core import Transcriber

@pytest.fixture
def mock_ov_genai():
    with patch("src.stt_npu.core.ov_genai") as mock:
        yield mock

def test_transcriber_initialization(mock_ov_genai):
    """Test that Transcriber initializes the pipeline correctly."""
    # Setup
    model_path = "models/whisper-tiny"
    device = "NPU"
    
    # Act
    transcriber = Transcriber(model_path=model_path, device=device)
    
    # Assert
    mock_ov_genai.WhisperPipeline.assert_called_once_with(model_path, device=device)
    assert transcriber.pipeline == mock_ov_genai.WhisperPipeline.return_value

def test_transcribe_chunk(mock_ov_genai):
    """Test that transcribe method calls pipeline.generate."""
    # Setup
    transcriber = Transcriber(model_path="dummy", device="CPU")
    mock_pipeline = mock_ov_genai.WhisperPipeline.return_value
    mock_pipeline.generate.return_value = "Hello World"
    
    dummy_audio = np.zeros(16000, dtype=np.float32) # 1 sec silent audio
    
    # Act
    result = transcriber.transcribe(dummy_audio)
    
    # Assert
    mock_pipeline.generate.assert_called_once()
    assert result == "Hello World"

def test_transcriber_initialization_defaults(mock_ov_genai):
    """Test default initialization values."""
    transcriber = Transcriber(model_path="dummy")
    # Verify default device is NPU as per ARCH.md
    mock_ov_genai.WhisperPipeline.assert_called_with("dummy", device="NPU")
