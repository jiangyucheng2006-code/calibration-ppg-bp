"""Aggregate leakage-safe repeat-seed meta-validation results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .training import file_sha256, participant_macro_metrics, save_json


KS = (1, 2, 3, 5)
TRAINED_METHODS = ("population", "m0", "m1", "m2")


def _find_complete_run(
    run_root: Path, pattern: str, *, marker: str = "run.json"
) -> Path:
    matches = sorted(
        path.parent for path in run_root.glob(f"{pattern}/{marker}") if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one complete run for {pattern}, found {len(matches)}: {matches}"
        )
    return matches[0]


def _assert_development_only(result: dict[str, object], path: Path) -> None:
    if result.get("split") != "meta_validation":
        raise AssertionError(f"non-validation result at {path}")
    if result.get("locked_test_accessed") is not False:
        raise AssertionError(f"locked-test status is not explicitly false at {path}")


def _row(
    *,
    seed: int,
    method: str,
    k: int,
    metrics: dict[str, object],
    run: dict[str, object] | None,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "seed": seed,
        "method": method,
        "k": k,
        "sbp_mae": float(metrics["sbp_mae"]),
        "dbp_mae": float(metrics["dbp_mae"]),
        "mean_mae": float(metrics["mean_mae"]),
        "sbp_rmse": float(metrics["sbp_rmse"]),
        "dbp_rmse": float(metrics["dbp_rmse"]),
        "sbp_bias": float(metrics["sbp_bias"]),
        "dbp_bias": float(metrics["dbp_bias"]),
        "n_participants": int(metrics["n_participants"]),
        "n_events": int(metrics["n_events"]),
        "best_epoch": int(run["best_epoch"]) if run is not None else None,
        "epochs_completed": int(run["epochs_completed"]) if run is not None else None,
        "stop_reason": str(run["stop_reason"]) if run is not None else "fixed_adaptation",
        "gpu": run.get("gpu") if run is not None else None,
        "run_dir": str(run_dir),
    }


def aggregate(
    run_root: Path,
    output: Path,
    *,
    run_prefix: str,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique prespecified seeds are required")

    rows: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    for seed in seeds:
        population_metrics_by_k: dict[int, dict[str, object]] = {}
        for method in TRAINED_METHODS:
            run_dir = _find_complete_run(
                run_root, f"{run_prefix}_{method}_seed{seed}_job*"
            )
            run_path = run_dir / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            _assert_development_only(run, run_path)
            if int(run["seed"]) != seed or run["method"] != method:
                raise AssertionError(f"run identity mismatch at {run_path}")

            if method == "population":
                history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
                best_record = next(
                    item for item in history if int(item["epoch"]) == int(run["best_epoch"])
                )
                metrics = best_record["validation"]
                for k in KS:
                    population_metrics_by_k[k] = metrics
                    rows.append(
                        _row(
                            seed=seed,
                            method=method,
                            k=k,
                            metrics=metrics,
                            run=run,
                            run_dir=run_dir,
                        )
                    )
            else:
                predictions_path = run_dir / "best_validation_predictions.parquet"
                predictions = pd.read_parquet(predictions_path)
                if set(predictions["k"].unique()) != set(KS):
                    raise AssertionError(f"unexpected K coverage at {predictions_path}")
                for k, group in predictions.groupby("k", sort=True):
                    metrics = participant_macro_metrics(group)
                    rows.append(
                        _row(
                            seed=seed,
                            method=method,
                            k=int(k),
                            metrics=metrics,
                            run=run,
                            run_dir=run_dir,
                        )
                    )
                evidence.append(
                    {
                        "path": str(predictions_path),
                        "sha256": file_sha256(predictions_path),
                    }
                )
            evidence.append({"path": str(run_path), "sha256": file_sha256(run_path)})

        controls_dir = _find_complete_run(
            run_root,
            f"{run_prefix}_calibration_controls_seed{seed}_job*",
            marker="metrics.json",
        )
        controls_path = controls_dir / "metrics.json"
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
        _assert_development_only(controls, controls_path)
        if str(controls["population_checkpoint"]).find(f"seed{seed}_") < 0:
            raise AssertionError(f"control checkpoint seed mismatch at {controls_path}")
        for method, by_k in controls["metrics"].items():
            for k_text, metrics in by_k.items():
                k = int(k_text)
                if method == "population":
                    expected = population_metrics_by_k[k]
                    if not np.isclose(
                        float(metrics["mean_mae"]),
                        float(expected["mean_mae"]),
                        atol=1e-6,
                    ):
                        raise AssertionError(
                            f"population metric mismatch for seed={seed}, K={k}"
                        )
                    continue
                rows.append(
                    _row(
                        seed=seed,
                        method=method,
                        k=k,
                        metrics=metrics,
                        run=None,
                        run_dir=controls_dir,
                    )
                )
        evidence.append({"path": str(controls_path), "sha256": file_sha256(controls_path)})

    per_seed = pd.DataFrame(rows).sort_values(["method", "k", "seed"])
    expected_counts = per_seed.groupby(["method", "k"])["seed"].nunique()
    if not expected_counts.eq(len(seeds)).all():
        raise AssertionError("incomplete method/K seed coverage")

    metric_columns = [
        "sbp_mae",
        "dbp_mae",
        "mean_mae",
        "sbp_rmse",
        "dbp_rmse",
        "sbp_bias",
        "dbp_bias",
    ]
    summary = (
        per_seed.groupby(["method", "k"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    output.mkdir(parents=True, exist_ok=False)
    per_seed_path = output / "per_seed_metrics.csv"
    summary_path = output / "summary_mean_sd.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    result = {
        "status": "complete",
        "split": "meta_validation",
        "locked_test_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_prefix": run_prefix,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "per_seed_rows": int(len(per_seed)),
        "summary_rows": int(len(summary)),
        "per_seed_metrics": str(per_seed_path),
        "summary_mean_sd": str(summary_path),
        "input_evidence": evidence,
    }
    save_json(output / "aggregation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated integer seeds")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    print(
        json.dumps(
            aggregate(
                args.run_root,
                args.output,
                run_prefix=args.run_prefix,
                seeds=seeds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
