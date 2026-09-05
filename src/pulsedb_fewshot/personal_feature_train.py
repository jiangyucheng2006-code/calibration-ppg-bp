"""Train a frozen mechanism-led feature candidate on disjoint seen-user windows."""
import argparse
import json
from pathlib import Path
from .same_subject_component_train import train_component
from .personal_feature_models import FEATURE_MODELS, SCREEN_ID, SPLIT_MODES, PersonalFeatureRegressor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=list(FEATURE_MODELS), required=True)
    p.add_argument("--store-root", type=Path, required=True)
    p.add_argument("--split-mode", choices=SPLIT_MODES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--demographics-path", type=Path)
    p.add_argument("--beat-similarity-path", type=Path)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--examples-per-epoch", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=0.5)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    print(json.dumps(train_component(args, components=FEATURE_MODELS,
        model_factory=PersonalFeatureRegressor, screen_id=SCREEN_ID,
        runner="personal_feature_residual", allowed_split_modes=SPLIT_MODES), indent=2))


if __name__ == "__main__":
    main()
