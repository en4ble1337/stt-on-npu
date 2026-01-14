# Journal 03: Wav2Vec2 NPU Success

**Date:** 2026-01-14  
**Status:** ✅ Complete - Wav2Vec2-Large Working on NPU

## Summary

After encountering issues with Whisper's decoder dynamic shapes on NPU, we successfully pivoted to **Wav2Vec2** - an encoder-only model with CTC decoding that is NPU-compatible.

---

## Key Breakthrough: Whisper vs Wav2Vec2

| Aspect | Whisper | Wav2Vec2 |
|--------|---------|----------|
| Architecture | Encoder-Decoder (Seq2Seq) | Encoder-only + CTC |
| NPU Compatibility | ❌ Decoder fails on dynamic shapes | ✅ Works perfectly |
| Static Shape Support | ❌ Decoder needs dynamic seq len | ✅ Easy to pad to fixed length |

---

## Performance Results

### Wav2Vec2-Large-960h on NPU vs CPU

| Device | ~6s Audio | ~10s Audio | Avg RTF |
|--------|-----------|------------|---------|
| **NPU** | 1.08s (RTF 0.18) | 1.07s (RTF 0.10) | **~0.14** |
| **CPU** | 0.78s (RTF 0.13) | 0.97s (RTF 0.11) | **~0.12** |

**Observations:**
- CPU is slightly faster (12% vs 14% RTF) for the large model
- NPU has more consistent inference time (~1s regardless of audio length)
- Both devices provide real-time transcription (RTF < 0.2)

### Transcription Quality (Wav2Vec2-Large)
Excellent accuracy on test sentences:
- `"THE DOG WAS LYING ON THE GRASS IN THE MIDDLE OF THE LAWN..."`
- `"ITS EYES WERE CLOSED IT LOOKED LIKE IT WAS RUNNING ON THE SIDE THE WAY DOGS RUN WHEN THEY THINK THEY ARE CHASING A CAT IN A DREAM"`

---

## Code Changes Summary

| File | Change |
|------|--------|
| `src/stt_npu/core.py` | Rewrote for Wav2Vec2 using `OVModelForCTC`, static shapes [1, 480000], CTC decoding |
| `models/wav2vec2-large-960h/` | Exported from `facebook/wav2vec2-large-960h` |
| `models/wav2vec2-base-960h/` | Also available (smaller, faster, less accurate) |

---

## Driver & Library Versions

- **Intel NPU Driver:** 32.0.100.4514
- **OpenVINO:** 2025.4.1
- **optimum-intel:** 1.27.0
- **Model:** facebook/wav2vec2-large-960h (315M params)

---

## Next Steps

- [ ] Clean up Whisper-related code and models
- [ ] Update README with Wav2Vec2 instructions
- [ ] Consider testing `wav2vec2-conformer` for even better accuracy
