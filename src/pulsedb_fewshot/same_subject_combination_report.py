"""Select a robust same-subject combination across both development splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .calbased_screen import PROTOCOL_ID
from .same_subject_combinations import (
    COMBINATIONS,
    PROMOTION_MARGIN_MMHG,
    REFERENCE_CANDIDATE,
    SCREEN_ID,
)
from .training import save_json


SCOPES = ("Overall", "MIMIC", "VitalDB")
MODES = ("random_disjoint", "chronological_blocked")


def _load_report(path: Path, expected_mode: str) -> tuple[dict[str, object], pd.DataFrame]:
    selection_path = path / "selection.json"
    summary_path = path / "participant_macro_summary.csv"
    if not selection_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"incomplete split report: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"wrong protocol in {path}")
    if selection.get("split_mode") != expected_mode:
        raise ValueError(f"wrong split mode in {path}")
    if selection.get("heldout_test_accessed") is not False:
        raise ValueError(f"split report accessed held-out data: {path}")
    frame = pd.read_csv(summary_path)
    required = {"candidate", "runner", "view", "mean_mae", "sbp_mae", "dbp_mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"split report missing columns: {sorted(missing)}")
    frame = frame.loc[frame["view"].isin(SCOPES)].copy()
    if set(frame["candidate"]) != set(COMBINATIONS):
        raise ValueError("split report does not contain the frozen combination matrix")
    if not frame["runner"].eq("combination_residual").all():
        raise ValueError("split report contains a non-combination runner")
    if frame.duplicated(["candidate", "view"]).any():
        raise ValueError("split report has duplicate candidate/view rows")
    return selection, frame


def build_final_report(
    random_report: Path,
    chronological_report: Path,
    output: Path,
) -> dict[str, object]:
    random_selection, random_frame = _load_report(random_report, MODES[0])
    chrono_selection, chrono_frame = _load_report(chronological_report, MODES[1])
    if int(random_selection["seed"]) != int(chrono_selection["seed"]):
        raise ValueError("split reports must use the same seed")

    merged = random_frame.merge(
        chrono_frame,
        on=["candidate", "view"],
        suffixes=("_random", "_chronological"),
        validate="one_to_one",
    )
    reference = merged.loc[merged["candidate"].eq(REFERENCE_CANDIDATE)].set_index(
        "view"
    )
    if set(reference.index) != set(SCOPES):
        raise ValueError("reference rows are incomplete")
    for mode in ("random", "chronological"):
        merged[f"mean_mae_gain_{mode}"] = merged.apply(
            lambda row: float(reference.loc[row["view"], f"mean_mae_{mode}"])
            - float(row[f"mean_mae_{mode}"]),
            axis=1,
        )
    merged["robust_mean_mae"] = (
        merged["mean_mae_random"] + merged["mean_mae_chronological"]
    ) / 2.0

    gates: dict[str, bool] = {}
    for candidate in COMBINATIONS:
        rows = merged.loc[merged["candidate"].eq(candidate)].set_index("view")
        if set(rows.index) != set(SCOPES):
            raise ValueError(f"incomplete candidate rows for {candidate}")
        if candidate == REFERENCE_CANDIDATE:
            gates[candidate] = True
            continue
        gates[candidate] = bool(
            rows.loc["Overall", "mean_mae_gain_random"] >= PROMOTION_MARGIN_MMHG
            and rows.loc["Overall", "mean_mae_gain_chronological"]
            >= PROMOTION_MARGIN_MMHG
            and all(
                rows.loc[scope, f"mean_mae_gain_{mode}"] > 0.0
                for scope in ("MIMIC", "VitalDB")
                for mode in ("random", "chronological")
            )
        )
    merged["passes_robust_gate"] = merged["candidate"].map(gates)

    overall = merged.loc[merged["view"].eq("Overall")].copy()
    promoted = overall.loc[
        overall["candidate"].ne(REFERENCE_CANDIDATE)
        & overall["passes_robust_gate"]
    ].sort_values(["robust_mean_mae", "candidate"], kind="mergesort")
    selected = (
        promoted.iloc[0]
        if not promoted.empty
        else overall.loc[overall["candidate"].eq(REFERENCE_CANDIDATE)].iloc[0]
    )

    result = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "screen_id": SCREEN_ID,
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(random_selection["seed"]),
        "selection_role": "internal_validation",
        "heldout_test_accessed": False,
        "candidate_count": len(COMBINATIONS),
        "promotion_margin_mmhg": PROMOTION_MARGIN_MMHG,
        "selection_rule": (
            "minimum mean Overall participant-macro mean MAE across random and "
            "chronological splits among candidates gaining at least 0.15 mmHg "
            "Overall in both modes and improving MIMIC and VitalDB in both modes; "
            "otherwise retain LoRA"
        ),
        "winner": {
            "candidate": str(selected["candidate"]),
            "robust_mean_mae": float(selected["robust_mean_mae"]),
            "random_mean_mae": float(selected["mean_mae_random"]),
            "chronological_mean_mae": float(selected["mean_mae_chronological"]),
            "passes_robust_gate": bool(selected["passes_robust_gate"]),
        },
        "random_report": str(random_report),
        "chronological_report": str(chronological_report),
        "winner_test_policy": (
            "No held-out target was accessed. Freeze the selected architecture and "
            "complete mechanism controls before any one-time held-out evaluation."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    merged.sort_values(
        ["view", "robust_mean_mae", "candidate"], kind="mergesort"
    ).to_csv(output / "combination_comparison.csv", index=False)
    save_json(output / "selection.json", result)
    lines = [
        "# Same-subject combination screen",
        "",
        "This is a development-only seen-participant analysis. The held-out role",
        "and the participant-disjoint primary test remain sealed.",
        "",
        f"- Seed: `{result['seed']}`",
        f"- Candidate count: `{result['candidate_count']}`",
        f"- Selected candidate: `{result['winner']['candidate']}`",
        f"- Random-disjoint mean MAE: {result['winner']['random_mean_mae']:.4f} mmHg",
        f"- Chronological mean MAE: {result['winner']['chronological_mean_mae']:.4f} mmHg",
        f"- Robust gate: `{str(result['winner']['passes_robust_gate']).lower()}`",
        "- Held-out test accessed: `false`",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--chronological-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_final_report(args.random_report, args.chronological_report, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
