"""CUDA forward/backward smoke test for every isolated component candidate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from .same_subject_components import (
    COMPONENT_SPECS,
    DESCRIPTOR_DIM,
    SUPPORT_COUNT,
    SameSubjectComponentRegressor,
    waveform_descriptor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--signal-length", type=int, default=1250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the allocated GPU job")
    if args.batch_size < 2:
        raise ValueError("batch size must be at least two")

    device = torch.device("cuda")
    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    device_name = torch.cuda.get_device_name(0)
    results: list[dict[str, object]] = []

    for spec in COMPONENT_SPECS:
        started = time.perf_counter()
        ppg = torch.randn(args.batch_size, 1, args.signal_length, device=device)
        anchor = torch.tensor([120.0, 75.0], device=device).repeat(args.batch_size, 1)
        targets = anchor + torch.randn_like(anchor) * torch.tensor([8.0, 5.0], device=device)
        query_descriptor = waveform_descriptor(ppg)
        support_descriptors = torch.randn(
            args.batch_size, SUPPORT_COUNT, DESCRIPTOR_DIM, device=device
        )
        support_bp = anchor[:, None, :] + torch.randn(
            args.batch_size, SUPPORT_COUNT, 2, device=device
        )
        subject_index = torch.arange(args.batch_size, device=device)
        demographics = torch.randn(args.batch_size, 5, device=device)

        model = SameSubjectComponentRegressor(
            spec,
            subject_count=args.batch_size,
        ).to(device)
        model.train()
        predictions = model(
            ppg,
            anchor,
            query_descriptor=query_descriptor,
            support_descriptors=support_descriptors if spec.uses_support else None,
            support_bp=support_bp if spec.uses_support else None,
            subject_index=subject_index if spec.uses_subject_index else None,
            demographics=demographics if spec.uses_demographics else None,
        )
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if predictions.shape != (args.batch_size, 2):
            raise RuntimeError(f"{spec.name}: invalid output shape {tuple(predictions.shape)}")
        if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name}: non-finite prediction or loss")
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"{spec.name}: missing or non-finite gradients")
        if not any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
            raise RuntimeError(f"{spec.name}: all gradients are zero")

        results.append(
            {
                "candidate": spec.name,
                "backbone": spec.backbone,
                "adapter": spec.adapter,
                "loss": float(loss.detach().cpu()),
                "seconds": time.perf_counter() - started,
                "status": "pass",
            }
        )
        del model, ppg, anchor, targets, predictions, loss
        del query_descriptor, support_descriptors, support_bp, subject_index, demographics
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "status": "pass",
        "device": device_name,
        "torch_version": torch.__version__,
        "candidate_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"CUDA_COMPONENT_SMOKE_STATUS={payload['status']}")
    print(f"CUDA_DEVICE={device_name}")
    print(f"CANDIDATE_COUNT={len(results)}")
    print(f"OUTPUT={args.output}")


if __name__ == "__main__":
    main()
