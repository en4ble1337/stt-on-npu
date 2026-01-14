# NPU Speech-to-Text with Wav2Vec2

![NPU Testing](assets/NPU-testing.png)
![NPU HLD](assets/logo.png)

Real-time speech recognition accelerated by Intel NPU using Wav2Vec2 CTC models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.4-green.svg)](https://docs.openvino.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **NPU-Accelerated Inference** - Up to 7x faster than CPU
- **Real-time Transcription** - 24.8x faster than real-time on 30s audio
- **Voice Activity Detection** - Silero VAD for speech segmentation
- **Simple CLI** - Easy-to-use command line interface

## NPU Capability Map

We extensively tested various model sizes on the Intel Core Ultra NPU. Here are the limits:

| Model Size | Parameters | Status | Performance |
| :--- | :--- | :--- | :--- |
| **Small** | 95M | Verified | **44x Speedup** (0.68s / 30s audio) |
| **Medium/Large** | 315M | Verified | **23x Speedup** (1.27s / 30s audio) |
| **Extra Large** | 600M | Failed | Crashes NPU (Exceeds Memory/Timeout) |
| **Giant** | 1B | Failed | Crashes NPU (Exceeds Memory/Timeout) |

**Recommendation:** Use the **315M Large model**. It matches the NPU's capabilities perfectly.

## Benchmark Results

| Duration | NPU RTF | CPU RTF | NPU Speedup |
|----------|---------|---------|-------------|
| 5s | 0.206 | 0.363 | **1.77x** |
| 10s | 0.104 | 0.253 | **2.44x** |
| 20s | 0.051 | 0.288 | **5.61x** |
| 30s | 0.040 | 0.290 | **7.18x** |

*RTF = Real-Time Factor (lower is faster)*

See full [Benchmark Results](benchmarks/benchmark_results_2026-01-14.md).

## Why NPU?

Read our deep dive on why we chose this architecture:
[NPU vs Whisper Concept](kb/npu_vs_whisper_cpu_concept.md) - Understanding Encoder-Only vs Encoder-Decoder for NPU.

## Requirements

### Hardware
- **Intel Core Ultra** processor (Meteor Lake or newer) with integrated NPU
- Microphone for real-time transcription

### Software
- Windows 10/11
- Python 3.10 or higher
- **Intel NPU Driver** version **32.0.100.4404** or newer (critical!)

## Installation

### Step 1: Check/Update Intel NPU Driver

This is **critical** - older drivers will cause inference errors.

```powershell
# Check current driver version
Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like '*Intel*AI*' } | Select-Object DeviceName, DriverVersion
```

**Required:** Version `32.0.100.4404` or newer

If your driver is older, download the latest from:  
[Intel NPU Driver Downloads](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)

### Step 2: Clone the Repository

```bash
git clone https://github.com/en4ble1337/stt-on-npu.git
cd stt-npu
```

### Step 3: Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Export the Model (if not included)

The Wav2Vec2 model needs to be exported to OpenVINO format:

```powershell
# For best accuracy (315M params, ~630MB)
optimum-cli export openvino --model facebook/wav2vec2-large-960h --task automatic-speech-recognition --weight-format fp16 models/wav2vec2-large-960h

# For smaller/faster model (95M params, ~190MB)
optimum-cli export openvino --model facebook/wav2vec2-base-960h --task automatic-speech-recognition --weight-format fp16 models/wav2vec2-base-960h
```

## Usage

### Real-time Transcription

```powershell
# Run on NPU (recommended)
python scripts/test_npu.py --model models/wav2vec2-large-960h --device NPU

# Run on CPU
python scripts/test_npu.py --model models/wav2vec2-large-960h --device CPU
```

Speak into your microphone - transcriptions appear after speech pauses.

### Run Benchmarks

```powershell
python scripts/benchmark.py --model models/wav2vec2-large-960h --iterations 5 --durations "5,10,20,30"
```

## Project Structure

```
stt-npu/
├── models/                     # Exported OpenVINO models
│   ├── wav2vec2-large-960h/   # Large model (best accuracy)
│   └── wav2vec2-base-960h/    # Base model (smaller)
├── scripts/
│   ├── test_npu.py            # Real-time transcription CLI
│   └── benchmark.py           # Performance benchmarking
├── src/stt_npu/
│   ├── core.py                # Transcriber class
│   └── vad.py                 # Voice Activity Detection
├── benchmarks/                 # Benchmark results
├── kb/                         # Knowledge base / journals
└── docs/                       # Documentation
```

## Troubleshooting

### "LLVM ERROR: Failed to infer result type"

**Cause:** Outdated Intel NPU driver  
**Solution:** Update to version 32.0.100.4404 or newer

### Model fails to load on NPU

**Cause:** NPU requires static input shapes  
**Solution:** The code automatically pads audio to 30 seconds. Ensure you're using the provided `core.py`.

### No NPU activity in Task Manager

**Cause:** Model may be running on CPU fallback  
**Solution:** Check that `--device NPU` is specified and driver is updated

### Poor transcription quality

**Cause:** Using base model or short audio segments  
**Solution:** Use `wav2vec2-large-960h` for better accuracy. Longer utterances (5+ seconds) work better.

## Technical Details

### Why Wav2Vec2 instead of Whisper?

| Model | Architecture | NPU Compatible |
|-------|--------------|----------------|
| **Wav2Vec2** | Encoder-only + CTC | Yes |
| Whisper | Encoder-Decoder | No (decoder needs dynamic shapes) |

NPU requires **static input shapes**. Wav2Vec2's encoder-only architecture with CTC decoding works perfectly, while Whisper's autoregressive decoder fails.

### Static Shape Padding

For NPU inference, all audio is padded to 30 seconds (480,000 samples at 16kHz). This enables:
- Consistent ~1 second inference time
- Higher throughput for longer audio
- Trade-off: slightly slower for very short clips (<3s)

## Documentation

- [Architecture](docs/ARCH.md)
- [Benchmark Results](benchmarks/)
- [Development Journals](kb/)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's AI toolkit
- [Optimum Intel](https://github.com/huggingface/optimum-intel) - HuggingFace integration
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-960h) - Meta's speech model
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
