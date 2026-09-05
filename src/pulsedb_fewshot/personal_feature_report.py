"""Cross-split promotion gate for the frozen eight-candidate feature screen."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .personal_feature_models import FEATURE_MODELS, REFERENCE, PRIMARY, SCREEN_ID
from .training import save_json


def build_report(random_report, chronological_report, output):
    frames, seeds = [], []
    for mode, path in [("random_disjoint", random_report), ("chronological_blocked", chronological_report)]:
        selection = json.loads((path / "selection.json").read_text())
        if selection.get("heldout_test_accessed") is not False or selection["split_mode"] != mode:
            raise ValueError("invalid development report")
        seeds.append(selection["seed"])
        f = pd.read_csv(path / "participant_macro_summary.csv")
        assert set(f.candidate) == set(FEATURE_MODELS)
        assert f.runner.eq("personal_feature_residual").all()
        assert len(f) == 24 and not f.duplicated(["candidate", "view"]).any()
        assert set(f.view) == {"Overall", "MIMIC", "VitalDB"}
        f["split_mode"] = mode
        ref = f.loc[f.candidate.eq(REFERENCE), ["view", "mean_mae"]].rename(columns={"mean_mae": "reference_mean_mae"})
        f = f.merge(ref, on="view", validate="many_to_one")
        f["gain_mmhg"] = f.reference_mean_mae - f.mean_mae
        frames.append(f)
    assert seeds[0] == seeds[1]
    combined = pd.concat(frames, ignore_index=True)
    gates = []
    for candidate, f in combined.groupby("candidate", sort=False):
        overall = f.loc[f.view.eq("Overall")]
        sources = f.loc[~f.view.eq("Overall")]
        assert len(overall) == 2 and len(sources) == 4
        gates.append({"candidate": candidate,
                      "passes_accuracy_gate": bool(overall.gain_mmhg.ge(0.15).all() and sources.gain_mmhg.gt(0).all()),
                      "mean_across_modes": float(overall.mean_mae.mean())})
    gate = pd.DataFrame(gates)
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "cross_split_comparison.csv", index=False)
    gate.to_csv(output / "promotion_gate.csv", index=False)
    eligible = gate.loc[gate.passes_accuracy_gate].sort_values("mean_across_modes")
    result = {"status": "complete", "screen_id": SCREEN_ID, "heldout_test_accessed": False,
              "primary_candidate": PRIMARY, "reference": REFERENCE, "seed": seeds[0],
              "eligible_candidates": eligible.candidate.tolist(),
              "recommendation": "retain reference" if eligible.empty else "confirm eligible candidates with new seeds; not final selection",
              "official_pulsedb_calbased_reproduction": False}
    save_json(output / "selection.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--random-report", type=Path, required=True)
    p.add_argument("--chronological-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build_report(args.random_report, args.chronological_report, args.output), indent=2))
