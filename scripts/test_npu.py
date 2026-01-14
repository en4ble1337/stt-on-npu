
import os
import sys
import argparse
import time
import numpy as np
import sounddevice as sd
import queue

# Add src to path to allow imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stt_npu.core import Transcriber
from stt_npu.vad import VoiceActivityDetector

# Audio constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 512 # 32ms chunks
SILENCE_DURATION_MS = 500 # Trigger transcription after 500ms silence

def main():
    parser = argparse.ArgumentParser(description="NPU STT Test Module")
    parser.add_argument("--device", type=str, default="NPU", help="Device to use for transcription (NPU, CPU)")
    parser.add_argument("--model", type=str, default="models/whisper-tiny-fp16", help="Path to OpenVINO IR model")
    parser.add_argument("--benchmark", action="store_true", help="Run comparison with CPU")
    args = parser.parse_args()

    print(f"Initializing Transcriber on {args.device}...")
    try:
        transcriber = Transcriber(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"Failed to initialize Transcriber: {e}")
        print("Ensure you have the model converted and OpenVINO installed.")
        return

    print("Initializing VAD...")
    vad = VoiceActivityDetector(threshold=0.5)

    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # Flatten and copy
        audio_queue.put(indata.flatten().copy())

    print("\nListening... (Press Ctrl+C to stop)")
    
    # State
    speech_buffer = []
    silence_counter = 0
    is_speaking = False
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Check VAD
                # VAD expects float32
                if vad.is_speech(chunk, SAMPLE_RATE):
                    if not is_speaking:
                        print("\n[Speech Detected]", end="", flush=True)
                        is_speaking = True
                    
                    speech_buffer.append(chunk)
                    silence_counter = 0
                    print(".", end="", flush=True)
                else:
                    if is_speaking:
                        # We were speaking, now silence
                        speech_buffer.append(chunk) # Include trailing silence context
                        silence_counter += (BLOCK_SIZE / SAMPLE_RATE) * 1000
                        
                        if silence_counter >= SILENCE_DURATION_MS:
                            print(f"\n[Processing {len(speech_buffer) * BLOCK_SIZE / SAMPLE_RATE:.1f}s audio]...")
                            
                            # Concatenate buffer
                            full_audio = np.concatenate(speech_buffer)
                            
                            # Transcribe
                            start_time = time.time()
                            text = transcriber.transcribe(full_audio)
                            inference_time = time.time() - start_time
                            
                            print(f"> {text}")
                            print(f"[Stats] {inference_time:.2f}s | RTF: {inference_time / (len(full_audio)/SAMPLE_RATE):.2f}")
                            
                            # Reset
                            speech_buffer = []
                            is_speaking = False
                            silence_counter = 0
                            print("Listening...")
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
