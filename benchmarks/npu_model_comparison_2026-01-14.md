# NPU Model Comparison: Wav2Vec2 Base vs Large

**Date:** 2026-01-14  
**Hardware:** Intel Core Ultra (NPU)  
**Input:** 30s Static Shape (480,000 samples)

## Executive Summary

We compared the **Base (95M)** and **Large (315M)** models on the NPU to understand the performance scaling.

- **Base (95M)** is **~2x faster** than Large.
- **Large (315M)** provides significantly better accuracy (WER ~3%) while still running at **24x real-time speed**.
- **Conclusion:** The NPU handles the Large model easily. The accuracy gain of the Large model is worth the performance cost for most applications.

## Detailed Results

### 1. Model Specifications

| Feature | Wav2Vec2 Base | Wav2Vec2 Large (Recommended) | Scaling Factor |
| :--- | :--- | :--- | :--- |
| **Parameters** | 95 Million | 315 Million | **3.3x** Larger |
| **Model Size (FP16)** | ~190 MB | ~630 MB | **3.3x** Larger |
| **Load Time** | 0.89s | 2.28s | **2.6x** Slower |

### 2. Inference Performance (NPU)

| Audio Duration | Base Time | Base RTF | Large Time | Large RTF | Speed Difference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5s** | 0.57s | 0.114 | 1.15s | 0.230 | Large is ~2x slower |
| **10s** | 0.57s | 0.057 | 1.10s | 0.110 | Large is ~1.9x slower |
| **20s** | 0.57s | 0.028 | 1.32s | 0.066 | Large is ~2.3x slower |
| **30s** | 0.68s | **0.023** | 1.27s | **0.042** | Large is ~1.9x slower |

*> Note: Inference time is nearly constant due to static shape padding.*

### 3. Throughput (Higher is Better)

- **Base Model:** Process **44 minutes** of audio in 1 minute.
- **Large Model:** Process **24 minutes** of audio in 1 minute.

## Recommendation

Use **Wav2Vec2-Large**.
- The **24x real-time** throughput is more than sufficient for almost any use case (live or batch).
- The drop in accuracy from switching to Base is not worth the extra speed, unless running on extremely constrained battery power.
