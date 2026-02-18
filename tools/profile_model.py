"""Utility script to inspect a configured model and report its parameter count
and approximate GFLOPs."""

import argparse
import os
import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

# Make sure project modules are importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig, yaml_utils  # noqa: E402


def _resolve_input_size(cfg: YAMLConfig, args: argparse.Namespace) -> Tuple[int, int]:
    """Pick the spatial size that should be profiled."""
    if args.height is not None and args.width is not None:
        return int(args.height), int(args.width)

    eval_size = cfg.yaml_cfg.get("eval_spatial_size")
    if eval_size is not None and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])

    default_size = cfg.yaml_cfg.get("input_size", [640, 640])
    return int(default_size[0]), int(default_size[1])


def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _measure_gflops(forward_fn: Callable[[torch.Tensor], torch.Any], dummy_input: torch.Tensor, device: torch.device):
    """Estimate FLOPs with torch.profiler."""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    try:
        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                forward_fn(dummy_input)
    except Exception as exc:  # pragma: no cover - only triggered when profiler fails
        print(f"[WARN] Unable to collect FLOPs with torch.profiler: {exc}")
        return None

    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    return total_flops / 1e9


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model parameters and GFLOPs.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML config.")
    parser.add_argument(
        "-u", "--update", nargs="+", help="Optional key=value overrides that follow YAMLConfig rules."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for the dummy input.")
    parser.add_argument("--height", type=int, help="Input height. Defaults to eval_spatial_size in config.")
    parser.add_argument("--width", type=int, help="Input width. Defaults to eval_spatial_size in config.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run profiling on. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--with-hsa",
        action="store_true",
        help="Include vis_encoder + DINOSAUR when measuring FLOPs (mirrors vis_encoder → DINOSAUR → RT-DETR).",
    )
    parser.add_argument(
        "--slot-resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(320, 320),
        help="Spatial size fed into vis_encoder/DINOSAUR when --with-hsa is enabled. Defaults to 320x320.",
    )
    return parser.parse_args()


def _forward_with_hsa(
    inputs: torch.Tensor,
    model: torch.nn.Module,
    vis_encoder: torch.nn.Module,
    dinosaur: torch.nn.Module,
    slot_size: Tuple[int, int],
):
    resized = F.interpolate(inputs, size=slot_size, mode="bilinear", align_corners=False)
    dino_outputs = dinosaur(vis_encoder(resized))

    if isinstance(dino_outputs, (tuple, list)):
        slots = dino_outputs[1]
    else:
        raise RuntimeError("Unexpected DINOSAUR output format when --with-hsa is enabled.")

    return model(inputs, slots=slots, slots1=None)


def main():
    args = parse_args()

    cfg_overrides = yaml_utils.parse_cli(args.update)
    cfg = YAMLConfig(args.config, **cfg_overrides)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.device = device  # downstream modules can read the explicit device if needed

    height, width = _resolve_input_size(cfg, args)

    model = cfg.model
    model.to(device)
    model.eval()

    vis_encoder = None
    dinosaur = None
    if args.with_hsa:
        print("[INFO] --with-hsa enabled: profiling vis_encoder → DINOSAUR → model pipeline.")
        vis_encoder = cfg.vis_encoder.to(device).eval()
        dinosaur = cfg.dinosaur.to(device).eval()

    dummy_input = torch.randn(args.batch_size, 3, height, width, device=device)

    total_params, trainable_params = _count_parameters(model)
    module_breakdown = [("model", total_params, trainable_params)]

    if vis_encoder is not None and dinosaur is not None:

        vis_total, vis_trainable = _count_parameters(vis_encoder)
        dino_total, dino_trainable = _count_parameters(dinosaur)
        total_params += vis_total + dino_total
        trainable_params += vis_trainable + dino_trainable
        module_breakdown.append(("vis_encoder", vis_total, vis_trainable))
        module_breakdown.append(("dinosaur", dino_total, dino_trainable))

        def pipeline(inputs: torch.Tensor):
            return _forward_with_hsa(inputs, model, vis_encoder, dinosaur, args.slot_resize)

        gflops = _measure_gflops(pipeline, dummy_input, device)
    else:
        gflops = _measure_gflops(model, dummy_input, device)

    model_name = cfg.yaml_cfg.get("model", model.__class__.__name__)

    print("=== Model Profiling ===")
    print(f"Config: {args.config}")
    print(f"Model: {model_name}")
    print(f"Device: {device.type}")
    print(f"Input: batch={args.batch_size}, channels=3, height={height}, width={width}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if len(module_breakdown) > 1:
        print("Breakdown:")
        for name, total, trainable in module_breakdown:
            print(f"  - {name}: total={total:,}, trainable={trainable:,}")
    if gflops is not None:
        print(f"Approximate GFLOPs: {gflops:.2f}")
    else:
        print("Approximate GFLOPs: unavailable (see warnings above).")


if __name__ == "__main__":
    main()
