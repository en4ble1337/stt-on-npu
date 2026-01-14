
import numpy as np
import torch

class VoiceActivityDetector:
    """
    Wrapper for Silero VAD.
    """
    def __init__(self, threshold: float = 0.5):
        """
        Initialize Silence VAD.
        
        Args:
            threshold (float): Speech probability threshold (0.0 to 1.0).
        """
        self.threshold = threshold
        
        # Load Silero VAD model from Torch Hub or local cache
        # Using trust_repo=True as Silero is a trusted source in this context
        # We load the onnx version if available, or the standard jit version
        # For simplicity in this "Basic Test" we use the torch hub load which is standard
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False  # Using JIT for simplicity, can switch to ONNX for NPU later if supported
        )
        self.get_speech_timestamps = utils[0]
        
    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Check if the given audio chunk contains speech.
        
        Args:
            audio_chunk (np.ndarray): Audio data (float32).
            sample_rate (int): Sample rate (must be 8000 or 16000).
            
        Returns:
            bool: True if speech detected, False otherwise.
        """
        # Ensure input is torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk)
        else:
            audio_tensor = audio_chunk
            
        # Add batch dimension if missing: (N,) -> (1, N)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Run inference
        # Silero expects (batch, time)
        # It returns a probability (0-1) for the chunk
        # Note: Silero VAD is typically stateful for streaming, but 'silero_vad' model call returns probability
        # for the whole chunk or streaming context.
        # For this basic implementation, we just check probability of the chunk.
        
        speech_prob = self.model(audio_tensor, sample_rate).item()
        
        return speech_prob > self.threshold
