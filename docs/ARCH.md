# Architecture: NPU Speech-to-Text Module

## 1. Overview
The system is a local, high-performance speech-to-text module leveraging Intel NPU hardware via OpenVINO. While initially implemented as a standalone test script (`test_npu.py`), the project is structured as a Python package (`src/stt_npu`) to verify scalability and facilitate retrofitting into other applications. It uses Silero VAD for efficient audio segmentation and OpenVINO GenAI for NPU-accelerated inference.

## 2. Dictionary

| Term | Definition | Example |
|------|------------|---------|
| **VAD** | Voice Activity Detection. Algorithms used to detect presence or absence of human speech. | Silero VAD detects speech vs silence. |
| **Inference** | The process of running data (audio) through the model to get a result (text). | "Running inference on the NPU took 0.5s." |
| **NPU** | Neural Processing Unit. Specialized hardware for AI math operations. | Intel Core Ultra NPU. |
| **Real-Time Factor (RTF)** | Ratio of processing time to audio duration. Lower is better. | RTF 0.5 means 10s audio takes 5s to process. |
| **Chunk** | A discrete segment of audio data processed at one time. | "Processing a 500ms audio chunk." |
| **OpenVINO IR** | Intermediate Representation. The proprietary model format (`.xml` + `.bin`) optimized for Intel hardware. | `whisper-tiny-fp16.xml` |

## 3. Tech Stack

| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| **Language** | Python | 3.10+ | Required for recent OpenVINO versions |
| **NPU Inference** | `openvino-genai` | 2024.5+ | Optimized high-level API for NPU |
| **Audio Capture** | `sounddevice` | Latest | Uses PortAudio backend |
| **Audio Processing** | `numpy`, `librosa` | Latest | For resampling/buffer management |
| **VAD Engine** | `silero-vad` | v5 (via ONNX/Torch) | High accuracy, low latency |
| **Dependency Mgmt** | `pip` | - | `requirements.txt` |
| **Build System** | `setuptools` | - | `pyproject.toml` layout |

## 4. Data Models (In-Memory)

#### AudioBuffer
Manages the raw audio stream before processing.
| Field | Type | Description |
|-------|------|-------------|
| raw_data | `np.ndarray` | Float32 buffer of captured audio |
| sample_rate | `int` | Always 16000 for Whisper/Silero |
| is_speech | `bool` | Current VAD state |
| timestamp | `float` | Time of capture |

#### BenchMarkResult
Stores comparison metrics.
| Field | Type | Description |
|-------|------|-------------|
| device | `str` | "CPU" or "NPU" |
| audio_duration | `float` | Length of audio in seconds |
| inference_time | `float` | Time taken to transcribe |
| rtf | `float` | `inference_time / audio_duration` |

## 5. CLI Interactions (`test_npu.py`)

The script acts as the primary interface for this phase.

**Usage:**
```bash
python scripts/test_npu.py [options]
```

**Options:**
- `--device`: `CPU` | `NPU` (default: `NPU`)
- `--benchmark`: Run comparison mode (transcribe same buffer on both)
- `--model`: `tiny` | `small` | `base` (default: `tiny`)

**Output:**
```text
[VAD] Speech Detected...
[NPU] Transcribing (3.5s audio)...
> "Hello world this is a test."
[Stats] Inference: 0.2s | RTF: 0.05
```

## 6. Directory Structure

| Path | Purpose |
|------|---------|
| `src/` | Source root |
| `src/stt_npu/` | Main package |
| `src/stt_npu/core.py` | Transcriber class wrapping OpenVINO |
| `src/stt_npu/vad.py` | VAD handling logic |
| `src/stt_npu/utils.py` | Audio resampling/buffering |
| `scripts/` | Executable scripts |
| `scripts/test_npu.py` | **Main entry point** for this task |
| `models/` | (Optional) Local storage if not using default cache |
| `docs/` | PRD and Architecture docs |
| `pyproject.toml` | Package definition |
| `requirements.txt` | Pinned dependencies |

## 7. Error Handling Strategy

**Initialization:**
- Check for NPU availability immediately on startup.
- If NPU missing/driver fail -> Fallback to CPU with warning OR hard exit (depending on flag).

**Runtime:**
- **VAD Glitches:** Short pulses (<200ms) of "speech" are ignored (noise suppression).
- **Buffer Overflow:** If audio > 30s, force transcription of current buffer to prevent Whisper hallucination/OOM.

## 8. Development & Security

- **Secrets:** None required for local inference.
- **Model Security:** usage of `trust_remote_code` (if needed for obscure models) should be disabled; standard Whisper is safe.
- **Privacy:** Minimal. Audio is processed locally in RAM and discarded. No cloud upload.

## 9. Integration Points

| System | Purpose | Method |
|--------|---------|--------|
| Hugging Face Hub | Downloading Models | HTTPS (via `optimum-cli` or `openvino-genai`) |
| Microphone | Audio Input | OS Sound Server (via `sounddevice`) |

## 10. Non-Functional Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Latency** | < 500ms | From end-of-speech to text-on-screen |
| **Accuracy** | Intelligible | WER < 15% for clear speech |
| **Stability** | No crash on long silence | Run for > 1 hour idle |
| **VAD** | No cut-off | ~300ms padding before/after speech |

## 11. Open Technical Questions
- **VAD/Whisper overlap:** How to handle speaking *while* the previous chunk is transcribing? (Current plan: Synchronous blocking for "Basic Test", simple queuing for "Future").
- **Model Compilation:** First run on NPU takes minutes. Need to ensure caching is enabled by default in `openvino-genai`.
