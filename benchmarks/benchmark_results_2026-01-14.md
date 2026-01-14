# Wav2Vec2 NPU vs CPU Benchmark Results

**Date:** 2026-01-14  
**Hardware:** Intel Core Ultra (with NPU)  
**Model:** facebook/wav2vec2-large-960h (315M parameters)

## Summary

**NPU is 1.77x to 7.18x faster than CPU**, with the advantage increasing for longer audio.

## Results Table

| Duration | NPU Time | NPU RTF | CPU Time | CPU RTF | NPU Speedup |
|----------|----------|---------|----------|---------|-------------|
| 5s | 1.028s | 0.206 | 1.816s | 0.363 | **1.77x** |
| 10s | 1.039s | 0.104 | 2.529s | 0.253 | **2.44x** |
| 20s | 1.028s | 0.051 | 5.767s | 0.288 | **5.61x** |
| 30s | 1.211s | 0.040 | 8.693s | 0.290 | **7.18x** |

*RTF = Real-Time Factor (lower is better, <1 means faster than real-time)*

## Model Load Times

| Device | Load Time |
|--------|-----------|
| NPU | 1.49s |
| CPU | 0.98s |

NPU requires additional time for model compilation with static shapes.

## Throughput

- **NPU**: Up to **24.8x real-time** (30s audio in 1.21s)
- **CPU**: About **3.5x real-time** (30s audio in 8.69s)

## Key Observations

1. **NPU has near-constant inference time** (~1s) regardless of audio length
   - This is because NPU uses static 30s input padding
   - All audio is padded to 480,000 samples for NPU compilation

2. **CPU scales linearly** with audio length
   - CPU uses dynamic shapes (no padding)
   - Longer audio = proportionally longer inference

3. **NPU advantage increases with audio length**
   - 5s audio: 1.77x faster
   - 30s audio: 7.18x faster

4. **NPU is ideal for**:
   - Batch processing of audio files
   - Longer audio segments (10+ seconds)
   - Power-efficient inference

5. **CPU may be better for**:
   - Very short audio clips (<3s)
   - When load time is critical
   - Systems without NPU

## Test Configuration

- **Iterations**: 5 per duration (plus 1 warmup)
- **Audio**: Synthetic speech-like waveform
- **Metrics**: Mean inference time, standard deviation, RTF

## Reproduce

```bash
python scripts/benchmark.py --model models/wav2vec2-large-960h --iterations 5 --durations "5,10,20,30"
```
