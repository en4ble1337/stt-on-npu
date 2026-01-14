
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from src.stt_npu.vad import VoiceActivityDetector

@pytest.fixture
def mock_torch_hub():
    with patch("torch.hub.load") as mock:
        yield mock

def test_vad_initialization(mock_torch_hub):
    """Test VAD detector initialization loads model."""
    mock_model = MagicMock()
    mock_utils = (MagicMock(), MagicMock(), MagicMock())
    mock_torch_hub.return_value = (mock_model, mock_utils)
    
    vad = VoiceActivityDetector(threshold=0.5)
    
    assert vad.threshold == 0.5
    mock_torch_hub.assert_called_once()
    assert vad.model == mock_model

def test_is_speech_detected(mock_torch_hub):
    """Test is_speech returns True when model prediction > threshold."""
    mock_model = MagicMock()
    # Mock model output: Tensor([0.8])
    mock_model.return_value = torch.tensor([0.8])
    mock_torch_hub.return_value = (mock_model, (MagicMock(), MagicMock(), MagicMock()))
    
    vad = VoiceActivityDetector(threshold=0.5)
    
    dummy_audio = np.zeros(512, dtype=np.float32)
    sample_rate = 16000
    
    result = vad.is_speech(dummy_audio, sample_rate)
    
    assert result is True
    mock_model.assert_called_once()

def test_is_speech_not_detected(mock_torch_hub):
    """Test is_speech returns False when model prediction < threshold."""
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([0.2])
    mock_torch_hub.return_value = (mock_model, (MagicMock(), MagicMock(), MagicMock()))
    
    vad = VoiceActivityDetector(threshold=0.5)
    
    dummy_audio = np.zeros(512, dtype=np.float32)
    sample_rate = 16000
    
    result = vad.is_speech(dummy_audio, sample_rate)
    
    assert result is False
