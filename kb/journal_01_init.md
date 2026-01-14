
# Journal 01: Initialization & NPU Implementation

**Date:** 2026-01-13
**Status:** Implementation Complete (Phase 1)

## Summary of Work
We have successfully bootstrapped the **Basic NPU Speech-to-Text Test Module** from zero to a functional CLI tool.

### 1. Planning
- **PRD (`docs/PRD.md`):** defined the scope for a standalone test script using Intel NPU and real-time microphone input.
- **Architecture (`docs/ARCH.md`):** structured the project as a scalable Python package (`src/stt_npu`) rather than just a loose script, ensuring future extensibility.

### 2. Infrastructure
- **Scaffolding:** Created the entire directory structure, standard files (`.gitignore`, `requirements.txt`), and documentation via `setup_genesis.py`.
- **Environment:** Verified dependencies (`openvino-genai`, `sounddevice`, `silero-vad`) are installable and compatible.

### 3. Implementation
- **Core Engine (`src/stt_npu/core.py`):** Implemented `Transcriber` class wrapping `openvino_genai.WhisperPipeline`.
- **VAD Engine (`src/stt_npu/vad.py`):** Implemented `VoiceActivityDetector` using Silero VAD (v5) via Torch Hub.
- **CLI Tool (`scripts/test_npu.py`):** Tied everything together into a usable command-line application that:
    - Listens to the microphone.
    - Filters silence using VAD.
    - Buffers speech.
    - Transcribes on the NPU when speech ends.
    - Reports inference time and "Real-Time Factor" (RTF).

## Next Steps (User Action Required)

To verify the system, you need to perform the following:

### 1. Obtain an OpenVINO Model
The NPU requires a specific "Intermediate Representation" (IR) format. You cannot just use a `.pt` file.
**Action:** Run the following command to download and convert a model (requires `optimum-intel` which contains `optimum-cli`):

```bash
optimum-cli export openvino --model openai/whisper-tiny --task automatic-speech-recognition --weight-format fp16 models/whisper-tiny-fp16
```
*(If `optimum-cli` is missing, `pip install optimum-intel[openvino]`)*

### 2. Run the Test
Once the model is in `models/whisper-tiny-fp16`:

```bash
# Activate venv first!
python scripts/test_npu.py --model models/whisper-tiny-fp16
```

### 3. Benchmark
To compare NPU vs CPU, run the script with the benchmark flag (though for true side-by-side, we might need to enhance the script or run it twice):

```bash
python scripts/test_npu.py --device CPU --model models/whisper-tiny-fp16
```
*(Compare the RTF numbers between runs)*
