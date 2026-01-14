# Journal 02: NPU Integration Debugging

**Date:** 2026-01-14  
**Status:** Blocked on NPU Driver Update

## Summary of Work

### Session Goal
Run Whisper transcription on the Intel NPU using the exported OpenVINO model.

### Issues Encountered & Resolved

#### 1. Virtual Environment Not Active
- **Problem:** User ran `pip install` globally instead of in the project venv.
- **Cause:** The venv folder was named `.venv` (with dot), but activation was attempted with `venv`.
- **Fix:** Activated using `.\.venv\Scripts\Activate.ps1`.

#### 2. `StatefulToStateless` Error (Initial)
```
Exception from src\core\src\pass\stateful_to_stateless.cpp:112:
Stateful models without `beam_idx` input are not supported
```
- **Cause:** The `openvino-genai` library (`WhisperPipeline`) expects models exported in a specific way that differs from standard `optimum-cli` exports.
- **Fix:** Replaced `openvino-genai` with `optimum-intel` in `src/stt_npu/core.py`. Now uses `OVModelForSpeechSeq2Seq` which is compatible with `optimum-cli` exports.

#### 3. VAD Initialization Error
```python
self.get_speech_timestamps, _, _ = utils  # ValueError: not enough values to unpack
```
- **Cause:** The `silero-vad` torch hub load returns a variable number of utilities depending on version.
- **Fix:** Changed to `self.get_speech_timestamps = utils[0]` for robust unpacking.

#### 4. CPU Test: SUCCESS ✅
```
> Mike test, Mike test, Mike test.
[Stats] 0.68s | RTF: 0.29
```
- Confirmed the code logic works correctly on CPU.

#### 5. NPU Test: FAILED (Driver Issue)
```
LLVM ERROR: Failed to infer result type(s).
filter shape must be the same: -9223372036854775808 != 80
```
- **Cause:** Known bug in Intel NPU driver version `32.0.100.4239`.
- **Fix:** Requires driver update to `32.0.100.4404` or newer (released November 2025).

---

## Code Changes Made

| File | Change |
|------|--------|
| `src/stt_npu/core.py` | Replaced `openvino_genai.WhisperPipeline` with `optimum.intel.OVModelForSpeechSeq2Seq` |
| `src/stt_npu/vad.py` | Fixed `silero-vad` utility unpacking (`utils[0]` instead of tuple unpack) |
| `requirements.txt` | Updated to use `optimum-intel[openvino]`, `transformers`, `accelerate` |
| `models/whisper-tiny-fp16-stateless/` | Re-exported model with `--disable-stateful` flag (precautionary) |

---

## Next Steps (After Reboot)

### 1. Update Intel NPU Driver
- **Current Version:** `32.0.100.4239`
- **Required Version:** `32.0.100.4404` or newer
- **Download:** https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html

### 2. Reboot the PC

### 3. Verify Driver Update
```powershell
Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like '*Intel*AI*' } | Select-Object DeviceName, DriverVersion
```
Should show version `32.0.100.4404` or higher.

### 4. Test NPU Again
```powershell
cd "c:\Users\Bart\SynologyDrive\SynologyDrive\Personal\AI\Projects\STT on NPU"
.\.venv\Scripts\Activate.ps1
python scripts/test_npu.py --model models/whisper-tiny-fp16-stateless --device NPU
```

### 5. If NPU Works
- Compare RTF between CPU and NPU
- Document results in Journal 03
- Proceed to benchmarking phase per ARCH.md

### 6. If NPU Still Fails
- Try the original model: `models/whisper-tiny-fp16`
- Check for newer `optimum-intel` / `openvino` versions
- Consider filing an issue on OpenVINO GitHub if problem persists
