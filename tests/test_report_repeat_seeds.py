from pathlib import Path

import numpy as np
import pandas as pd

from pulsedb_fewshot.report_repeat_seeds import (
    _public_evidence,
    summarize_extended,
    summarize_participant_macro,
    write_markdown,
)


def _rows() -> pd.DataFrame:
    rows = []
    for seed, shift in [(1, 0.0), (2, 2.0), (3, 4.0)]:
        for k in (1, 2, 3, 5):
            for bp, macro in [("SBP", 8.0 + shift), ("DBP", 4.0 + shift)]:
                rows.append(
                    {
                        "Seed": seed,
                        "Setting": f"M0 Variable-K residual anchor (K={k})",
                        "BP": bp,
                        "MAE": macro + 1.0,
                        "R²": 0.5 + shift / 100.0,
                        "ME": 1.0,
                        "STD": 7.0,
                        "≤5 mmHg (%)": 55.0,
                        "≤10 mmHg (%)": 80.0,
                        "≤15 mmHg (%)": 92.0,
                        "AAMI": "PASS*",
                        "BHS": "PASS (Grade B)*",
                        "Participant-macro MAE": macro,
                        "N participants": 2,
                        "N query events": 4,
                        "Method": "m0",
                        "K": k,
                        "Aggregation": "event-pooled diagnostic",
                        "Standards scope": "numerical screen only",
                    }
                )
    return pd.DataFrame(rows)


def test_participant_macro_summary_uses_sample_seed_sd() -> None:
    summary = summarize_participant_macro(_rows())
    summary = summary[summary["K"] == 1].iloc[0]
    assert summary["SBP MAE"] == 10.0
    assert summary["DBP MAE"] == 6.0
    assert summary["Mean MAE"] == 8.0
    assert summary["Mean MAE seed SD"] == 2.0


def test_extended_summary_preserves_required_columns_and_pass_counts() -> None:
    summary = summarize_extended(_rows())
    assert len(summary) == 8
    assert list(summary.columns[:11]) == [
        "Setting",
        "BP",
        "MAE",
        "R²",
        "ME",
        "STD",
        "≤5 mmHg (%)",
        "≤10 mmHg (%)",
        "≤15 mmHg (%)",
        "AAMI",
        "BHS",
    ]
    sbp = summary[(summary["BP"] == "SBP") & (summary["K"] == 1)].iloc[0]
    assert np.isclose(sbp["MAE"], 11.0)
    assert np.isclose(sbp["MAE seed SD"], 2.0)
    assert sbp["AAMI"] == "PASS* (3/3 seeds pass)"
    assert sbp["BHS"] == "PASS* (3/3 seeds pass; A=0, B=3, C=0, D=0)"


def test_markdown_writer_handles_multiple_k_without_duplicate_pivot(tmp_path: Path) -> None:
    per_seed = _rows()
    participant_macro = summarize_participant_macro(per_seed)
    extended = summarize_extended(per_seed)
    output = tmp_path / "report.md"
    write_markdown(participant_macro, extended, per_seed, output)
    text = output.read_text(encoding="utf-8")
    assert "Phase-5 five-seed development results" in text
    assert "M0 Variable-K residual anchor" in text
    assert "https://www.iso.org/standard/73339.html" in text


def test_public_evidence_does_not_expose_cluster_paths() -> None:
    result = _public_evidence(
        [
            {
                "path": "/home/user/work/project/run-1/predictions.parquet",
                "sha256": "abc",
                "rows": 10,
            }
        ]
    )
    assert result == [
        {
            "run_id": "run-1",
            "file_name": "predictions.parquet",
            "sha256": "abc",
            "rows": 10,
        }
    ]
