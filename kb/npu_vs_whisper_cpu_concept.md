# NPU Architecture Decisions: Wav2Vec2 vs Whisper

## The Problem: Dynamic Shapes vs NPU

Intel's NPU (Neural Processing Unit) architecture, specifically in Core Ultra processors, is designed for **maximum efficiency** on **static workloads**.

### NPU Requirements
- **Static Input Shapes:** The NPU compiler needs to know exact input dimensions (e.g., `[1, 480000]`) at compile time to optimize memory layout and data movement.
- **Fixed Compute Graph:** Operations that change dynamically (loops based on data content) interrupt the NPU's pipelining.

### The Whisper Problem
Whisper uses an **Encoder-Decoder** architecture (Seq2Seq).
1. **Encoder:** Processes audio. (This *can* be made static).
2. **Decoder:** Generates text token-by-token.
   - It is **Auto-regressive**: The output of step 1 feeds step 2.
   - It uses **Dynamic Loop**: It doesn't know how many tokens it will generate.
   - It uses **KV Cache**: The input shape to the decoder Grows with every new word generated.

This "growing shape" (Dynamic Shape) is extremely hostile to the current NPU compiler. It forces the NPU to "recompile" or fallback to CPU for every single word generated, destroying performance.

## The Solution: Wav2Vec2 (CTC)

We pivoted to **Wav2Vec2**, which uses **CTC (Connectionist Temporal Classification)**.

### Why it works on NPU
- **Encoder-Only:** The model is just one big Encoder.
- **One-Shot Inference:** You feed in audio, and it outputs the entire matrix of probabilities for all time steps at once.
- **No Feedback Loop:** There is no auto-regressive generation.
- **Static Friendly:** We can pad all audio to exactly 30 seconds (`480,000 samples`). The model always outputs exactly, say, 1500 time-step predictions. We simply ignore the padded ones.

### The Trade-off
- **Whisper:** Better likelihood of perfect grammar (due to decoder language model).
- **Wav2Vec2:** Insanely fast (Parallelizable), but output can sometimes be phonetically correct but grammatically loose (unless an external Language Model is added).

For **Real-Time NPU** applications, **Wav2Vec2 is the clear winner** because it fits the hardware's "Static Shape" requirement perfectly.
