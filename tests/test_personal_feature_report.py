import json
import pandas as pd
from pulsedb_fewshot.personal_feature_models import FEATURE_MODELS, PRIMARY, REFERENCE
from pulsedb_fewshot.personal_feature_report import build_report


def test_gate_requires_both_modes_and_all_sources(tmp_path):
    paths = []
    for mode in ["random_disjoint", "chronological_blocked"]:
        path = tmp_path / mode
        path.mkdir()
        (path / "selection.json").write_text(json.dumps({"split_mode": mode, "seed": 3, "heldout_test_accessed": False}))
        rows = []
        for candidate in FEATURE_MODELS:
            for view in ["Overall", "MIMIC", "VitalDB"]:
                mae = 4 if candidate == REFERENCE else 3.8
                if candidate == PRIMARY and mode == "chronological_blocked" and view == "MIMIC":
                    mae = 4.1
                rows.append({"candidate": candidate, "runner": "personal_feature_residual", "view": view, "mean_mae": mae})
        pd.DataFrame(rows).to_csv(path / "participant_macro_summary.csv", index=False)
        paths.append(path)
    result = build_report(*paths, tmp_path / "out")
    assert PRIMARY not in result["eligible_candidates"]
    assert REFERENCE not in result["eligible_candidates"]
    assert "shared_bilinear32" in result["eligible_candidates"]
