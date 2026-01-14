# NPU-Compatible ASR Models Research

**Date:** 2026-01-14  
**Goal:** Find speech-to-text models that deploy easily on Intel NPU

---

## Key Finding: NPU Requires Static Shapes

The Intel NPU compiler **requires all input dimensions to be fully static**. This is the fundamental constraint that determines which models will work.

| Architecture | NPU Compatibility | Why |
|--------------|-------------------|-----|
| **Encoder-only + CTC** | ✅ Excellent | Fixed input length, no autoregressive decoding |
| **Encoder-Decoder (Seq2Seq)** | ❌ Poor | Decoder has dynamic sequence lengths |

---

## Top Recommended Models for NPU

### 1. **Wav2Vec2** (META) - ⭐ RECOMMENDED
- **Model:** `facebook/wav2vec2-base-960h` or `facebook/wav2vec2-large-960h`
- **Architecture:** Encoder-only with CTC
- **WER:** <5% on LibriSpeech (English)
- **NPU Status:** Officially supported, OpenVINO has demos
- **Size:** 95M (base) / 315M (large)
- **Why it works:** Single forward pass, no autoregressive decoder
- **Languages:** English (fine-tuned versions for other languages available)

### 2. **Wav2Vec2-Conformer** (META)
- **Model:** `facebook/wav2vec2-conformer-rel-pos-large-960h`
- **Architecture:** Conformer encoder + CTC (hybrid attention + convolution)
- **WER:** ~2-3% on LibriSpeech
- **NPU Status:** Supported via Optimum-Intel
- **Size:** 600M+
- **Why it works:** Still encoder-only, just more powerful encoder

### 3. **HuBERT** (META)
- **Model:** `facebook/hubert-large-ls960-ft`
- **Architecture:** Similar to Wav2Vec2, encoder-only + CTC
- **WER:** Similar to Wav2Vec2
- **NPU Status:** Should work (same architecture family)
- **Size:** 315M

---

## Models to Avoid for NPU

### ❌ Whisper (OpenAI)
- **Problem:** Encoder-decoder architecture
- **Issue:** Decoder requires dynamic shapes for autoregressive generation
- **Status:** Encoder works on NPU, but full transcription fails

### ❌ Canary Qwen 2.5B (NVIDIA)
- **Problem:** Uses LLM decoder (Qwen)
- **Issue:** Far too large (2.5B params) and same decoder shape issues
- **Note:** Best accuracy (5.63% WER) but not NPU-friendly

### ❌ IBM Granite Speech
- **Problem:** LLM-based, 8B parameters
- **Issue:** Too large for NPU, dynamic shapes

---

## Recommendation

**Start with Wav2Vec2-base-960h** because:

1. ✅ Officially NPU-verified in OpenVINO
2. ✅ Encoder-only (no decoder complications)
3. ✅ CTC decoding is simple post-processing
4. ✅ Small model (95M params)
5. ✅ Good English accuracy (<5% WER)
6. ✅ OpenVINO has working demo code

### Export Command
```bash
optimum-cli export openvino \
  --model facebook/wav2vec2-base-960h \
  --task automatic-speech-recognition \
  --weight-format fp16 \
  models/wav2vec2-base-960h
```

---

## Alternative: If Multilingual Needed

Consider fine-tuned Wav2Vec2 models:
- `facebook/wav2vec2-large-xlsr-53` (53 languages)
- `jonatasgrosman/wav2vec2-large-xlsr-53-english`

Or accept Whisper on CPU (works well, 0.29 RTF on CPU from our tests).
