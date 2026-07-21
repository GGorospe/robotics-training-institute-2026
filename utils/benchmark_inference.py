#!/usr/bin/env python3
"""
benchmark_inference.py

Measures how fast the robot's hardware can run inference with a trained
classifier, to help decide whether optimization (e.g. TensorRT / FP16) or a
reduced camera resolution (640x400 -> 224x224) is needed for smooth live
classification in B3b.

The full live-inference pipeline has two stages per frame:
  1. Preprocessing -- convert the BGR camera frame to a PIL image, resize
     to 224x224, convert to a tensor, normalize. The cost of this stage
     depends on the *camera* resolution.
  2. Model forward pass -- run the 224x224 tensor through resnet18. The
     cost of this stage does NOT depend on camera resolution (the model
     always sees 224x224).

This script times both stages separately at both candidate camera
resolutions, so the results directly answer: "would shrinking the camera
feed help, or is the model itself the bottleneck (meaning optimization is
the better lever)?"

Usage:
    python3 benchmark_inference.py /home/explorer/Models/best_model_red_blue_classifier_v1.pth
    python3 benchmark_inference.py <model.pth> --iterations 200
"""

import argparse
import time

import numpy as np
import torch

from inference_utils import load_model_and_metadata, predict_image

# (width, height) camera resolutions to compare
RESOLUTIONS = [(640, 400), (224, 224)]


def make_fake_frame(width, height):
    """Creates a random BGR uint8 frame shaped like a real camera frame,
    so the benchmark exercises the exact same code path as live inference
    without needing the camera.
    """
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def sync_if_cuda(device):
    """GPU work is asynchronous -- timing without synchronizing measures
    only how fast work was *queued*, not how fast it ran. This forces the
    GPU to finish before the clock is read.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()


def time_stage(fn, iterations, device, warmup=10):
    """Times `fn` over `iterations` runs after `warmup` untimed runs,
    returning (mean_ms, min_ms, max_ms).

    Warmup matters: the first few inferences pay one-time costs (CUDA
    context setup, memory allocation, cuDNN autotuning) that would badly
    skew a cold measurement.
    """
    for _ in range(warmup):
        fn()
    sync_if_cuda(device)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        sync_if_cuda(device)
        samples_ms.append((time.perf_counter() - start) * 1000)

    return (
        sum(samples_ms) / len(samples_ms),
        min(samples_ms),
        max(samples_ms),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark classifier inference speed.")
    parser.add_argument("model_path", help="Path to a .pth checkpoint trained in B3a")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Timed iterations per measurement (default: 100)")
    args = parser.parse_args()

    model, device, class_names, _record = load_model_and_metadata(args.model_path)
    print(f"\nDevice: {device}")
    print(f"Iterations per measurement: {args.iterations} (plus warmup)\n")

    # ---- Stage 2 first: model-only forward pass (resolution-independent) ----
    fixed_input = torch.randn(1, 3, 224, 224).to(device)

    def forward_only():
        with torch.no_grad():
            model(fixed_input)

    model_mean, model_min, model_max = time_stage(forward_only, args.iterations, device)
    model_only_fps = 1000.0 / model_mean

    print("Model forward pass only (224x224 input -- camera resolution doesn't affect this):")
    print(f"  mean {model_mean:6.1f} ms   min {model_min:6.1f} ms   max {model_max:6.1f} ms"
          f"   -> {model_only_fps:5.1f} fps ceiling\n")

    # ---- Full pipeline at each candidate camera resolution ----
    print("Full pipeline (preprocess + forward), per camera resolution:")
    results = {}
    for width, height in RESOLUTIONS:
        frame = make_fake_frame(width, height)

        def full_pipeline():
            predict_image(model, frame, class_names, device)

        mean_ms, min_ms, max_ms = time_stage(full_pipeline, args.iterations, device)
        fps = 1000.0 / mean_ms
        preprocess_ms = mean_ms - model_mean
        results[(width, height)] = mean_ms

        print(f"  {width}x{height}:")
        print(f"    mean {mean_ms:6.1f} ms   min {min_ms:6.1f} ms   max {max_ms:6.1f} ms"
              f"   -> {fps:5.1f} fps")
        print(f"    (~{preprocess_ms:.1f} ms of that is preprocessing)\n")

    # ---- Interpretation ----
    big = results[RESOLUTIONS[0]]
    small = results[RESOLUTIONS[1]]
    saved_ms = big - small
    saved_pct = (saved_ms / big) * 100 if big > 0 else 0

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Reducing the camera feed from {RESOLUTIONS[0][0]}x{RESOLUTIONS[0][1]} to "
          f"{RESOLUTIONS[1][0]}x{RESOLUTIONS[1][1]} saves about "
          f"{saved_ms:.1f} ms/frame ({saved_pct:.0f}% of total time).")
    if saved_pct < 15:
        print("-> Preprocessing is NOT the bottleneck. Reducing camera resolution")
        print("   won't help much; model optimization (e.g. TensorRT/FP16) is the")
        print("   better lever if live inference feels sluggish.")
    else:
        print("-> Preprocessing is a meaningful share of frame time. Reducing the")
        print("   camera resolution is worth trying before reaching for model")
        print("   optimization.")
    print(f"\nAbsolute ceiling with this model on this hardware: ~{model_only_fps:.1f} fps.")
    print("If that ceiling itself is too low for smooth live classification,")
    print("only model optimization (not camera resolution) can raise it.")


if __name__ == "__main__":
    main()
