# PRD: Basic NPU Speech-to-Text Test Module

## Introduction
A lightweight, standalone Python script to validate and benchmark NPU-accelerated Whisper transcription on Intel Core Ultra hardware. The primary goal is to confirm functional NPU inference and observe performance gains compared to legacy CPU execution, using live microphone input and Voice Activity Detection (VAD).

## Goals
- **Enable NPU Acceleration:** successfully run OpenAI Whisper models on the Intel NPU using OpenVINO.
- **Live Input Processing:** Capture real-time audio from the microphone, segmented by Silero VAD.
- **Performance Verification:** Provide a mechanism to compare inference speed (NPU vs. CPU) to validate hardawre benefits.
- **Foundational Code:** Create a functional, isolated script (`test_npu.py`) that serves as a proof-of-concept for future modularization.

## User Stories

### US-001: Live Audio Capture with VAD
**Description:** As a developer, I want the system to listen to my microphone and only process audio when I am speaking.
**Acceptance Criteria:**
- [ ] Integrate Silero VAD to detect speech segments.
- [ ] Ignore silence/background noise to prevent unnecessary processing.
- [ ] Buffer valid speech audio for transcription.
- [ ] Console indicates "Listening..." vs "Processing...".

### US-002: NPU-Accelerated Transcription
**Description:** As a developer, I want the speech-to-text conversion to happen on the NPU so that I can utilize the dedicated hardware.
**Acceptance Criteria:**
- [ ] Load Whisper model (Tiny/Small) converted for OpenVINO (FP16/INT8).
- [ ] Execute inference specifically on the "NPU" device.
- [ ] Output transcribed text to the console.
- [ ] Fail gracefully (with error) if NPU is unavailable/driver missing.

### US-003: CPU vs NPU Benchmark Mode
**Description:** As a tester, I want to compare NPU speed against CPU speed to quantify the performance benefit.
**Acceptance Criteria:**
- [ ] Script accepts a flag (e.g., `--benchmark` or prompt selection) to run simultaneous or sequential inference.
- [ ] Report inference time (latency) for the same audio chunk on both CPU and NPU.
- [ ] Display a "Speedup Factor" (e.g., "NPU is 3.5x faster").

## Functional Requirements
1.  **FR-1:** System must use `openvino-genai` or `optimum-intel` to interface with the NPU.
2.  **FR-2:** System must implement Silero VAD for efficient audio segmentation.
3.  **FR-3:** The core logic must be contained in `test_npu.py`.
4.  **FR-4:** System must support switching between CPU and NPU devices for testing purposes.
5.  **FR-5:** Input audio must be resampled to 16kHz as required by Whisper/Silero.

## Non-Goals
- No Graphical User Interface (GUI). Console only.
- No complex streaming architecture (simple chunk-based processing is acceptable for this test).
- No saving of transcripts to files (console output is sufficient).
- No fine-tuning or training of models.

## Design Considerations
- **Hardware:** Intel Core Ultra 5 225u.
- **Drivers:** Requires Intel NPU Driver and OpenVINO toolkit installed.
- **Model:** Start with `whisper-tiny` or `whisper-small` for immediate feedback.

## Technical Considerations
- **Dependencies:** `openvino`, `torch`, `librosa` (audio processing), `sounddevice` (mic input), `silero-vad`.
- **Latency:** NPU compilation can take time on first run; implement model caching to mitigate this.
- **Fallback:** Verification that the script *actually* used the NPU (check OpenVINO device usage logs).

## Success Metrics
- Successful transcription of spoken sentences.
- Measurable inference time difference between NPU and CPU.
- VAD correctly identifying speech segments vs. silence.

## Open Questions
- Do we need to support specific model sizes (e.g., medium/large) for this initial test, or is `small` sufficient? (Assumption: `small` is sufficient for speed validation).
