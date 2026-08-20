import json

import pandas as pd

from pulsedb_fewshot.report_round8 import build_report


def _write_run(path, predictions: pd.DataFrame, *, seed: int, standard: bool) -> None:
    path.mkdir()
    (path / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "split": "meta_validation",
                "locked_test_accessed": False,
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )
    name = "best_validation_predictions.parquet" if standard else "predictions.parquet"
    predictions.to_parquet(path / name, index=False)


def test_round8_report_separates_full_coverage_from_similarity_filter(tmp_path) -> None:
    reference = pd.DataFrame(
        {
            "subject_uid": ["m1", "m2", "v1", "v2"],
            "event_id": ["e1", "e2", "e3", "e4"],
            "k": [5, 5, 5, 5],
            "target_sbp": [120.0, 130.0, 110.0, 140.0],
            "target_dbp": [70.0, 80.0, 65.0, 90.0],
            "pred_sbp": [121.0, 128.0, 112.0, 137.0],
            "pred_dbp": [71.0, 78.0, 66.0, 87.0],
        }
    )
    candidate = reference.drop(columns="k").copy()
    candidate["source"] = ["MIMIC", "MIMIC", "VitalDB", "VitalDB"]
    candidate["pred_sbp"] += 0.25
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_run(reference_dir, reference, seed=1, standard=True)
    _write_run(candidate_dir, candidate, seed=2, standard=False)
    similarity = reference[["subject_uid", "event_id"]].copy()
    similarity["pairwise_corr_median"] = [0.95, 0.89, 0.99, float("nan")]
    similarity_path = tmp_path / "similarity.parquet"
    similarity.to_parquet(similarity_path, index=False)

    result = build_report(
        reference_name="Reference",
        reference_dir=reference_dir,
        candidates={"Candidate": candidate_dir},
        similarity_path=similarity_path,
        output=tmp_path / "report",
        expected_seed=2,
    )

    assert result["locked_test_accessed"] is False
    assert result["full_coverage_queries_per_setting"] == 4
    coverage = pd.read_csv(tmp_path / "report" / "similarity_filter_coverage.csv")
    overall = coverage.loc[coverage["Scope"].eq("Overall")].iloc[0]
    assert overall["Queries retained"] == 2
    assert overall["Query coverage (%)"] == 50.0
