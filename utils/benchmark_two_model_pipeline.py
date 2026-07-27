#!/usr/bin/env python3
"""
benchmark_two_model_pipeline.py

Measures the latency of a two-model decision pipeline for the upcoming
D-series flowchart: classify with the Free/Blocked model first; only if
the path is blocked, classify a second time with the Red/Blue model to
decide which way to turn (e.g. red -> turn left, blue -> turn right).

This directly answers the timing question behind that flowchart: is the
"blocked" branch -- where both models run back-to-back -- still fast
enough to fit inside behavior_utils.py's poll_interval (default 0.5s),
and comfortably under the RVR's own ~2 second command-timeout window?

Two scenarios are measured:
  BEST CASE  -- path free: only the Free/Blocked model runs.
  WORST CASE -- path blocked: Free/Blocked model runs, THEN the Red/Blue
                model runs too, in sequence, before a decision is made.

Usage:
    python3 benchmark_two_model_pipeline.py <free_blocked_model.pth> <red_blue_model.pth>
    python3 benchmark_two_model_pipeline.py <free_blocked_model.pth> <red_blue_model.pth> --iterations 200
"""

import argparse
import time

import numpy as np
import torch

from inference_utils import load_model_and_metadata, predict_image

CAMERA_RESOLUTION = (640, 400)  # (width, height)


def make_fake_frame(width, height):
    """Creates a random BGR uint8 frame shaped like a real camera frame,
    so the benchmark exercises the exact same code path as the live
    behavior loop without needing the camera or an actual obstacle.
    """
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def sync_if_cuda(device):
    """GPU work is asynchronous -- timing without synchronizing measures
    only how fast work was *queued*, not how fast it ran.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()


def time_stage(fn, iterations, device, warmup=10):
    """Times `fn` over `iterations` runs after `warmup` untimed runs,
    returning (mean_ms, min_ms, max_ms). See benchmark_inference.py for
    why warmup and synchronization both matter here.
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
    parser = argparse.ArgumentParser(
        description="Benchmark a two-model (Free/Blocked -> Red/Blue) decision pipeline."
    )
    parser.add_argument("free_blocked_model_path", help="Path to a Free/Blocked .pth checkpoint")
    parser.add_argument("red_blue_model_path", help="Path to a Red/Blue .pth checkpoint")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Timed iterations per measurement (default: 100)")
    parser.add_argument("--poll-interval", type=float, default=0.5,
                        help="behavior_utils.py's poll_interval to check against, in seconds (default: 0.5)")
    args = parser.parse_args()

    print("Loading Free/Blocked model...")
    fb_model, fb_device, fb_classes, _fb_record = load_model_and_metadata(args.free_blocked_model_path)

    print("\nLoading Red/Blue model...")
    rb_model, rb_device, rb_classes, _rb_record = load_model_and_metadata(args.red_blue_model_path)

    if fb_device != rb_device:
        print(f"\nNote: models loaded onto different devices ({fb_device} vs {rb_device}). "
              f"This is unexpected -- both should land on the same GPU/CPU -- and the "
              f"timings below reflect whatever this run actually did.")

    print(f"\nIterations per measurement: {args.iterations} (plus warmup)\n")

    width, height = CAMERA_RESOLUTION
    frame = make_fake_frame(width, height)

    # ---- Best case: path is free, only the Free/Blocked model runs ----
    def free_case():
        predict_image(fb_model, frame, fb_classes, fb_device)

    free_mean, free_min, free_max = time_stage(free_case, args.iterations, fb_device)
    free_rate = 1000.0 / free_mean

    print("BEST CASE -- path free (only the Free/Blocked model runs):")
    print(f"  mean {free_mean:6.1f} ms   min {free_min:6.1f} ms   max {free_max:6.1f} ms"
          f"   -> {free_rate:5.1f} decisions/sec equivalent\n")

    # ---- Worst case: path is blocked, both models run in sequence ----
    def blocked_case():
        predict_image(fb_model, frame, fb_classes, fb_device)
        predict_image(rb_model, frame, rb_classes, rb_device)

    blocked_mean, blocked_min, blocked_max = time_stage(blocked_case, args.iterations, fb_device)
    blocked_rate = 1000.0 / blocked_mean

    print("WORST CASE -- path blocked (Free/Blocked, then Red/Blue, in sequence):")
    print(f"  mean {blocked_mean:6.1f} ms   min {blocked_min:6.1f} ms   max {blocked_max:6.1f} ms"
          f"   -> {blocked_rate:5.1f} decisions/sec equivalent\n")

    # ---- Interpretation ----
    poll_interval_ms = args.poll_interval * 1000
    rvr_timeout_ms = 2000.0
    added_ms = blocked_mean - free_mean

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"behavior_utils.py's poll_interval is currently {args.poll_interval}s ({poll_interval_ms:.0f} ms).")
    print(f"The RVR's own command timeout is ~{rvr_timeout_ms:.0f} ms.\n")

    print(f"Running the second (Red/Blue) model adds ~{added_ms:.1f} ms to a 'blocked' decision "
          f"({blocked_mean / free_mean:.1f}x the free-only latency).\n")

    if blocked_mean < poll_interval_ms:
        margin_pct = (1 - blocked_mean / poll_interval_ms) * 100
        print(f"-> Worst-case latency ({blocked_mean:.1f} ms) fits inside poll_interval "
              f"({poll_interval_ms:.0f} ms), with {margin_pct:.0f}% headroom.")
        print("   The two-model pipeline should be safe to drop straight into behavior_utils.py's")
        print("   existing loop, without changing poll_interval or any other timing.")
    else:
        print(f"-> Worst-case latency ({blocked_mean:.1f} ms) EXCEEDS poll_interval "
              f"({poll_interval_ms:.0f} ms).")
        print("   The decision function itself would take longer than one loop cycle is budgeted")
        print("   for. Consider raising poll_interval for this behavior, or only running the")
        print("   second model on alternating iterations.")

    if blocked_mean < rvr_timeout_ms:
        headroom_ms = rvr_timeout_ms - blocked_mean
        print(f"\nWorst case is {headroom_ms:.0f} ms under the RVR's ~{rvr_timeout_ms:.0f} ms command "
              f"timeout -- a single slow iteration would not, by itself, cause an unexpected stop.")
    else:
        print(f"\nWARNING: worst-case latency ({blocked_mean:.1f} ms) approaches or exceeds the RVR's")
        print(f"own ~{rvr_timeout_ms:.0f} ms command timeout -- a single slow iteration could cause")
        print("an unexpected stop if a new drive command doesn't land in time.")


if __name__ == "__main__":
    main()
