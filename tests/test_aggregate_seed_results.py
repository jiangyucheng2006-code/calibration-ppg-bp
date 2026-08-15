import json
from pathlib import Path

import pandas as pd

from pulsedb_fewshot.aggregate_seed_results import KS, aggregate


def _metrics(error: float) -> dict[str, float]:
    return {
        "n_participants": 1,
        "n_events": 1,
        "sbp_mae": error,
        "dbp_mae": error,
        "mean_mae": error,
        "sbp_rmse": error,
        "dbp_rmse": error,
        "sbp_bias": error,
        "dbp_bias": error,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_aggregate_requires_complete_development_only_seed_matrix(tmp_path: Path) -> None:
    run_prefix = "repeat-test"
    seeds = (11, 12, 13)
    for seed in seeds:
        for method in ("population", "m0", "m1", "m2"):
            run_dir = tmp_path / f"{run_prefix}_{method}_seed{seed}_job1"
            run_dir.mkdir()
            run = {
                "method": method,
                "seed": seed,
                "split": "meta_validation",
                "locked_test_accessed": False,
                "best_epoch": 1,
                "epochs_completed": 9,
                "stop_reason": "early_stopping",
                "gpu": "test-gpu",
            }
            _write_json(run_dir / "run.json", run)
            if method == "population":
                _write_json(
                    run_dir / "history.json",
                    [{"epoch": 1, "validation": _metrics(2.0)}],
                )
            else:
                rows = []
                for k in KS:
                    rows.append(
                        {
                            "k": k,
                            "subject_uid": "subject-1",
                            "event_id": f"event-{k}",
                            "target_sbp": 100.0,
                            "target_dbp": 60.0,
                            "pred_sbp": 101.0,
                            "pred_dbp": 61.0,
                        }
                    )
                pd.DataFrame(rows).to_parquet(
                    run_dir / "best_validation_predictions.parquet", index=False
                )

        controls_dir = (
            tmp_path / f"{run_prefix}_calibration_controls_seed{seed}_job2"
        )
        controls_dir.mkdir()
        _write_json(
            controls_dir / "metrics.json",
            {
                "split": "meta_validation",
                "locked_test_accessed": False,
                "population_checkpoint": f"population_seed{seed}_job1/best.pt",
                "metrics": {
                    "population": {str(k): _metrics(2.0) for k in KS},
                    "residual_offset": {str(k): _metrics(1.5) for k in KS},
                },
            },
        )

    output = tmp_path / "aggregate"
    result = aggregate(
        tmp_path,
        output,
        run_prefix=run_prefix,
        seeds=seeds,
    )

    assert result["locked_test_accessed"] is False
    assert result["n_seeds"] == 3
    assert result["per_seed_rows"] == 60
    assert result["summary_rows"] == 20
    per_seed = pd.read_csv(output / "per_seed_metrics.csv")
    assert set(per_seed["seed"]) == set(seeds)
    assert set(per_seed["k"]) == set(KS)
