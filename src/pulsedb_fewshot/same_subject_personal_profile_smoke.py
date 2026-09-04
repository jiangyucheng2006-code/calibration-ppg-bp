"""CUDA forward/backward smoke test for every personal-profile candidate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .same_subject_components import DESCRIPTOR_DIM, SUPPORT_COUNT, waveform_descriptor
from .same_subject_personal_profiles import (
    PROFILE_SPECS,
    SameSubjectPersonalProfileRegressor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--signal-length", type=int, default=1250)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the allocated GPU job")
    device = torch.device("cuda")
    torch.manual_seed(20260904)
    results = []
    for spec in PROFILE_SPECS:
        ppg = torch.randn(args.batch_size, 1, args.signal_length, device=device)
        anchor = torch.randn(args.batch_size, 2, device=device)
        model = SameSubjectPersonalProfileRegressor(
            spec, subject_count=args.batch_size
        ).to(device)
        kwargs: dict[str, torch.Tensor] = {
            "query_descriptor": waveform_descriptor(ppg),
            "subject_index": torch.arange(args.batch_size, device=device),
        }
        if spec.uses_support:
            kwargs["support_descriptors"] = torch.randn(
                args.batch_size, SUPPORT_COUNT, DESCRIPTOR_DIM, device=device
            )
            kwargs["support_bp"] = torch.randn(
                args.batch_size, SUPPORT_COUNT, 2, device=device
            )
        prediction = model(ppg, anchor, **kwargs)
        loss = torch.nn.functional.smooth_l1_loss(
            prediction, torch.randn_like(prediction)
        )
        loss.backward()
        if prediction.shape != (args.batch_size, 2):
            raise RuntimeError(f"{spec.name}: invalid output shape")
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name}: non-finite output")
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not any(torch.count_nonzero(g).item() for g in gradients):
            raise RuntimeError(f"{spec.name}: no nonzero gradient")
        results.append({"candidate": spec.name, "status": "pass"})
        del model, ppg, anchor, prediction, loss
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "status": "pass",
        "device": torch.cuda.get_device_name(0),
        "candidate_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("CUDA_PERSONAL_PROFILE_SMOKE_STATUS=pass")
    print(f"CANDIDATE_COUNT={len(results)}")
    print(f"OUTPUT={args.output}")


if __name__ == "__main__":
    main()
