"""Train one prespecified module combination on the same-subject track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calbased_screen import EARLY_STOPPING_PATIENCE
from .same_subject_combinations import (
    COMBINATION_SPLIT_MODES,
    COMBINATIONS,
    SCREEN_ID,
    SameSubjectCombinationRegressor,
)
from .same_subject_component_train import train_component


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=list(COMBINATIONS), required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--split-mode", choices=COMBINATION_SPLIT_MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demographics-path", type=Path)
    parser.add_argument("--beat-similarity-path", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--examples-per-epoch", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = train_component(
        args,
        components=COMBINATIONS,
        model_factory=SameSubjectCombinationRegressor,
        screen_id=SCREEN_ID,
        runner="combination_residual",
        allowed_split_modes=COMBINATION_SPLIT_MODES,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
