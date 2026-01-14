
import numpy as np
import torch
from optimum.intel import OVModelForCTC
from transformers import AutoProcessor

class Transcriber:
    """
    A wrapper around Optimum Intel's OVModelForCTC for NPU-accelerated transcription.
    Uses Wav2Vec2 with CTC decoding - ideal for NPU due to encoder-only architecture.
    """
    
    # Wav2Vec2 expects 16kHz audio
    SAMPLE_RATE = 16000
    # Fixed input length for NPU static shapes (30 seconds of audio)
    STATIC_INPUT_LENGTH = 16000 * 30  # 480,000 samples
    
    def __init__(self, model_path: str, device: str = "NPU"):
        """
        Initialize the Transcriber.

        Args:
            model_path (str): Path to the OpenVINO IR model directory.
            device (str): target device (NPU, CPU, GPU). Defaults to NPU.
        """
        self.model_path = model_path
        self.device = device.upper()
        
        print(f"Loading Wav2Vec2 model from {model_path} to {self.device}...")
        
        if self.device == "NPU":
            # For NPU: load without compiling, reshape to static, then compile
            print("Loading model with compile=False for NPU reshaping...")
            self.model = OVModelForCTC.from_pretrained(
                model_path, 
                compile=False
            )
            
            # Reshape for static input length (NPU requires this)
            print(f"Reshaping model for static input length [{1}, {self.STATIC_INPUT_LENGTH}]...")
            self.model.model.reshape({
                "input_values": [1, self.STATIC_INPUT_LENGTH]
            })
            
            # Compile for NPU - set device then compile
            print("Compiling model for NPU...")
            self.model._device = "NPU"
            self.model.compile()
            print("Model compiled and loaded on NPU successfully.")
        else:
            # For CPU/GPU: direct loading works fine
            self.model = OVModelForCTC.from_pretrained(model_path, device=self.device)
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            print(f"Warning: AutoProcessor failed ({e}), falling back to Wav2Vec2Processor")
            from transformers import Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a chunk of audio using CTC decoding.

        Args:
            audio_chunk (np.ndarray): Raw audio data (float32), 16kHz.

        Returns:
            str: Transcribed text.
        """
        original_length = len(audio_chunk)
        
        # Pad or truncate to static length for NPU
        if self.device == "NPU":
            if len(audio_chunk) < self.STATIC_INPUT_LENGTH:
                # Pad with zeros
                padded = np.zeros(self.STATIC_INPUT_LENGTH, dtype=np.float32)
                padded[:len(audio_chunk)] = audio_chunk
                audio_chunk = padded
            elif len(audio_chunk) > self.STATIC_INPUT_LENGTH:
                # Truncate (should rarely happen with 30s limit)
                audio_chunk = audio_chunk[:self.STATIC_INPUT_LENGTH]
                original_length = self.STATIC_INPUT_LENGTH
        
        # Process audio through feature extractor
        inputs = self.processor(
            audio_chunk, 
            sampling_rate=self.SAMPLE_RATE, 
            return_tensors="pt",
            padding=False
        )
        
        # Run inference - CTC model outputs logits directly
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Calculate how many output frames correspond to actual audio
        # Wav2Vec2 has a stride of 320 samples per output frame
        output_frames_for_actual_audio = original_length // 320
        
        # Only take logits for actual audio portion
        if self.device == "NPU" and output_frames_for_actual_audio < logits.shape[1]:
            logits = logits[:, :output_frames_for_actual_audio, :]
        
        # CTC decode: take argmax and decode tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to text
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription
