"""Aggregate internal-validation CalBased analogue runs and select one winner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .calbased_metrics import POOLED_COLUMNS, pooled_diagnostics
from .calbased_screen import PROTOCOL_ID, SCREENING_READ_ROLES
from .training import save_json


def _load_valid_run(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run.json: {path}")
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError(f"incomplete run: {run_dir}")
    if run.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"wrong protocol in {run_dir}")
    if run.get("source_parent_split") != "meta_train":
        raise ValueError(f"run did not derive only from meta_train: {run_dir}")
    if tuple(run.get("read_roles", [])) != SCREENING_READ_ROLES:
        raise ValueError(f"run read a forbidden role: {run_dir}")
    if run.get("selection_role") != "internal_validation":
        raise ValueError(f"run used a forbidden selection role: {run_dir}")
    if run.get("heldout_test_accessed") is not False:
        raise ValueError(f"run accessed heldout_test: {run_dir}")
    prediction_path = run_dir / "best_internal_validation_predictions.parquet"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"missing validation predictions: {prediction_path}")
    return run


def aggregate_screen_runs(run_dirs: list[Path], output: Path) -> dict[str, object]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    loaded = [(run_dir, _load_valid_run(run_dir)) for run_dir in run_dirs]
    candidates = [str(run["candidate"]) for _, run in loaded]
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate runs must be unique")
    split_modes = {str(run["split_mode"]) for _, run in loaded}
    seeds = {int(run["seed"]) for _, run in loaded}
    if len(split_modes) != 1:
        raise ValueError("aggregate one split_mode at a time")
    if len(seeds) != 1:
        raise ValueError("first-round report must contain one common seed")

    rows: list[dict[str, object]] = []
    run_lookup: dict[str, str] = {}
    diagnostics: list[pd.DataFrame] = []
    reference_targets: pd.DataFrame | None = None
    for run_dir, run in loaded:
        candidate = str(run["candidate"])
        run_lookup[candidate] = str(run_dir)
        predictions = pd.read_parquet(
            run_dir / "best_internal_validation_predictions.parquet"
        )
        targets = predictions[
            ["event_id", "subject_uid", "source", "target_sbp", "target_dbp"]
        ].sort_values("event_id", kind="mergesort").reset_index(drop=True)
        if reference_targets is None:
            reference_targets = targets
        elif not targets.equals(reference_targets):
            raise ValueError(
                "all candidates must use the identical internal-validation "
                "events and targets"
            )
        diagnostics.append(pooled_diagnostics(predictions, candidate))
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or "Overall" not in metrics:
            raise ValueError(f"run has no Overall metrics: {run_dir}")
        for view, values in metrics.items():
            if not isinstance(values, dict):
                raise ValueError(f"invalid metric view {view!r} in {run_dir}")
            rows.append(
                {
                    "candidate": candidate,
                    "runner": run["runner"],
                    "backbone": run.get("backbone") or "",
                    "split_mode": run["split_mode"],
                    "seed": run["seed"],
                    "view": view,
                    "n_participants": values["n_participants"],
                    "n_events": values["n_events"],
                    "sbp_mae": values["sbp_mae"],
                    "dbp_mae": values["dbp_mae"],
                    "mean_mae": values["mean_mae"],
                    "sbp_bias": values["sbp_bias"],
                    "dbp_bias": values["dbp_bias"],
                    "worst_30_mean_mae": values["worst_30_mean_mae"],
                    "retained_70_mean_mae": values["retained_70_mean_mae"],
                    "run_dir": str(run_dir),
                }
            )
    summary = pd.DataFrame(rows).sort_values(
        ["view", "mean_mae", "candidate"], kind="mergesort"
    )
    overall = summary.loc[summary["view"].eq("Overall")].sort_values(
        ["mean_mae", "candidate"], kind="mergesort"
    )
    if len(overall) != len(loaded):
        raise ValueError("each candidate must have exactly one Overall metric row")
    winner = overall.iloc[0]
    result = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_mode": next(iter(split_modes)),
        "seed": next(iter(seeds)),
        "selection_role": "internal_validation",
        "heldout_test_accessed": False,
        "selection_metric": "Overall participant_macro_mean_mae",
        "candidate_count": len(loaded),
        "winner": {
            "candidate": str(winner["candidate"]),
            "mean_mae": float(winner["mean_mae"]),
            "sbp_mae": float(winner["sbp_mae"]),
            "dbp_mae": float(winner["dbp_mae"]),
            "run_dir": run_lookup[str(winner["candidate"])],
        },
        "winner_test_policy": (
            "No held-out test was accessed. Refit this one selected candidate "
            "on train plus internal_validation before a separate one-time test."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "internal_validation_summary.csv", index=False)
    summary.to_csv(output / "participant_macro_summary.csv", index=False)
    pooled = pd.concat(diagnostics, ignore_index=True)
    pooled.to_csv(output / "event_pooled_diagnostics_all_scopes.csv", index=False)
    for scope, suffix in (
        ("Overall", "overall"),
        ("MIMIC", "mimic"),
        ("VitalDB", "vitaldb"),
    ):
        pooled.loc[pooled["Scope"].eq(scope), list(POOLED_COLUMNS)].to_csv(
            output / f"event_pooled_{suffix}.csv", index=False
        )
    save_json(output / "selection.json", result)
    save_json(
        output / "internal_validation_summary.json",
        summary.to_dict(orient="records"),
    )
    lines = [
        "# Development CalBased analogue: internal-validation screen",
        "",
        "This is a seen-subject development analogue, not the official PulseDB "
        "CalBased benchmark and not the participant-disjoint primary result.",
        "",
        f"- Split mode: `{result['split_mode']}`",
        f"- Seed: `{result['seed']}`",
        "- Selection role: `internal_validation`",
        "- Held-out test accessed: `false`",
        f"- Selected candidate: `{result['winner']['candidate']}`",
        f"- Selection mean MAE: {result['winner']['mean_mae']:.4f} mmHg",
        "",
        "The held-out test remains sealed until winner-only refitting.",
        "",
        "Participant-macro MAE is the selection metric. Event-pooled AAMI/BHS "
        "entries are retrospective numerical screens only and do not establish "
        "formal standards or device compliance.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_screen_runs(args.run, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
