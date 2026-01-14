#!/usr/bin/env python
"""
Comprehensive NPU vs CPU Benchmark for Wav2Vec2

Tests:
1. Model load time
2. Inference time across multiple audio durations
3. Multiple iterations for statistical reliability
4. Throughput (audio seconds / inference seconds)
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def generate_test_audio(duration_seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic speech-like audio for testing."""
    samples = int(duration_seconds * sample_rate)
    # Generate a complex waveform (mix of frequencies like speech)
    t = np.linspace(0, duration_seconds, samples, dtype=np.float32)
    audio = np.zeros(samples, dtype=np.float32)
    
    # Add multiple frequency components (simulate speech formants)
    for freq in [100, 200, 300, 500, 800, 1200]:
        audio += 0.1 * np.sin(2 * np.pi * freq * t + np.random.random() * np.pi)
    
    # Add some amplitude modulation (like syllables)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # ~3 Hz syllable rate
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def benchmark_model_load(model_path: str, device: str) -> float:
    """Benchmark model loading time."""
    from stt_npu.core import Transcriber
    
    start = time.perf_counter()
    transcriber = Transcriber(model_path=model_path, device=device)
    load_time = time.perf_counter() - start
    
    return load_time, transcriber


def benchmark_inference(
    transcriber, 
    audio: np.ndarray, 
    num_iterations: int = 5
) -> Dict:
    """Benchmark inference with multiple iterations."""
    inference_times = []
    
    # Warmup run (don't count)
    _ = transcriber.transcribe(audio)
    
    # Timed runs
    for i in range(num_iterations):
        start = time.perf_counter()
        result = transcriber.transcribe(audio)
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)
    
    audio_duration = len(audio) / 16000
    
    return {
        "audio_duration_s": audio_duration,
        "inference_times": inference_times,
        "mean_inference_s": np.mean(inference_times),
        "std_inference_s": np.std(inference_times),
        "min_inference_s": np.min(inference_times),
        "max_inference_s": np.max(inference_times),
        "mean_rtf": np.mean(inference_times) / audio_duration,
        "throughput_x": audio_duration / np.mean(inference_times),
    }


def run_benchmark(model_path: str, devices: List[str], durations: List[float], iterations: int):
    """Run complete benchmark suite."""
    results = {}
    
    for device in devices:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {device}")
        print(f"{'='*60}")
        
        # Model load benchmark
        print(f"\n[1] Loading model on {device}...")
        load_time, transcriber = benchmark_model_load(model_path, device)
        print(f"    Load time: {load_time:.2f}s")
        
        results[device] = {
            "load_time_s": load_time,
            "benchmarks": {}
        }
        
        # Inference benchmarks for each duration
        print(f"\n[2] Running inference benchmarks ({iterations} iterations each)...")
        for duration in durations:
            print(f"\n    Testing {duration}s audio...")
            audio = generate_test_audio(duration)
            
            bench = benchmark_inference(transcriber, audio, iterations)
            results[device]["benchmarks"][f"{duration}s"] = bench
            
            print(f"    Mean: {bench['mean_inference_s']:.3f}s ± {bench['std_inference_s']:.3f}s")
            print(f"    RTF:  {bench['mean_rtf']:.3f} | Throughput: {bench['throughput_x']:.1f}x real-time")
        
        # Cleanup
        del transcriber
    
    return results


def print_comparison_table(results: Dict):
    """Print a formatted comparison table."""
    devices = list(results.keys())
    durations = list(results[devices[0]]["benchmarks"].keys())
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Load time comparison
    print("\n## Model Load Time")
    print("-" * 40)
    for device in devices:
        print(f"  {device:6s}: {results[device]['load_time_s']:6.2f}s")
    
    # Inference comparison table
    print("\n## Inference Performance (Mean)")
    print("-" * 80)
    header = f"{'Duration':>10s}"
    for device in devices:
        header += f" | {device + ' (s)':>12s} | {device + ' RTF':>10s}"
    print(header)
    print("-" * 80)
    
    for dur in durations:
        row = f"{dur:>10s}"
        for device in devices:
            bench = results[device]["benchmarks"][dur]
            row += f" | {bench['mean_inference_s']:>12.3f} | {bench['mean_rtf']:>10.3f}"
        print(row)
    
    # Speedup comparison
    if len(devices) == 2:
        print("\n## Speedup Comparison (NPU vs CPU)")
        print("-" * 60)
        for dur in durations:
            npu_time = results["NPU"]["benchmarks"][dur]["mean_inference_s"]
            cpu_time = results["CPU"]["benchmarks"][dur]["mean_inference_s"]
            speedup = cpu_time / npu_time
            winner = "NPU" if speedup > 1 else "CPU"
            print(f"  {dur:>10s}: {speedup:.2f}x ({winner} is faster)")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive NPU vs CPU Benchmark")
    parser.add_argument("--model", type=str, default="models/wav2vec2-large-960h",
                        help="Path to model")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations per test")
    parser.add_argument("--durations", type=str, default="2,5,10,20,30",
                        help="Comma-separated audio durations to test (seconds)")
    parser.add_argument("--device", type=str, default="both",
                        help="Device to test: NPU, CPU, or both")
    args = parser.parse_args()
    
    durations = [float(d) for d in args.durations.split(",")]
    
    if args.device.lower() == "both":
        devices = ["NPU", "CPU"]
    else:
        devices = [args.device.upper()]
    
    print("="*60)
    print("NPU vs CPU COMPREHENSIVE BENCHMARK")
    print("="*60)
    print(f"Model:      {args.model}")
    print(f"Devices:    {', '.join(devices)}")
    print(f"Durations:  {durations}")
    print(f"Iterations: {args.iterations}")
    
    results = run_benchmark(args.model, devices, durations, args.iterations)
    print_comparison_table(results)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
