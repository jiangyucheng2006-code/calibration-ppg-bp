"""Compare compact personal profiles with paired residual and LoRA controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .calbased_screen import PROTOCOL_ID
from .same_subject_personal_profiles import (
    PERSONAL_PROFILES,
    PRIMARY_CANDIDATE,
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
    if set(frame["candidate"]) != set(PERSONAL_PROFILES):
        raise ValueError("split report does not contain the frozen profile matrix")
    if not frame["runner"].eq("personal_profile_residual").all():
        raise ValueError("split report contains a foreign runner")
    if frame.duplicated(["candidate", "view"]).any():
        raise ValueError("split report has duplicate candidate/view rows")
    return selection, frame


def build_final_report(
    random_report: Path,
    chronological_report: Path,
    output: Path,
) -> dict[str, object]:
    random_selection, random_frame = _load_report(random_report, MODES[0])
    chronological_selection, chronological_frame = _load_report(
        chronological_report, MODES[1]
    )
    if int(random_selection["seed"]) != int(chronological_selection["seed"]):
        raise ValueError("split reports must use the same seed")
    merged = random_frame.merge(
        chronological_frame,
        on=["candidate", "view"],
        suffixes=("_random", "_chronological"),
        validate="one_to_one",
    )
    reference = merged.loc[
        merged["candidate"].eq(REFERENCE_CANDIDATE)
    ].set_index("view")
    if set(reference.index) != set(SCOPES):
        raise ValueError("LoRA reference rows are incomplete")
    for mode in ("random", "chronological"):
        merged[f"mean_mae_gain_vs_lora_{mode}"] = merged.apply(
            lambda row: float(reference.loc[row["view"], f"mean_mae_{mode}"])
            - float(row[f"mean_mae_{mode}"]),
            axis=1,
        )
    merged["robust_mean_mae"] = (
        merged["mean_mae_random"] + merged["mean_mae_chronological"]
    ) / 2.0

    def passes_gate(candidate: str) -> bool:
        rows = merged.loc[merged["candidate"].eq(candidate)].set_index("view")
        return bool(
            rows.loc["Overall", "mean_mae_gain_vs_lora_random"]
            >= PROMOTION_MARGIN_MMHG
            and rows.loc["Overall", "mean_mae_gain_vs_lora_chronological"]
            >= PROMOTION_MARGIN_MMHG
            and all(
                rows.loc[scope, f"mean_mae_gain_vs_lora_{mode}"] > 0.0
                for scope in ("MIMIC", "VitalDB")
                for mode in ("random", "chronological")
            )
        )

    merged["passes_robust_gate"] = merged["candidate"].map(
        lambda candidate: passes_gate(str(candidate))
        if str(candidate) not in {REFERENCE_CANDIDATE, "residual_reference"}
        else False
    )
    overall = merged.loc[merged["view"].eq("Overall")].copy()
    numerical_best = overall.sort_values(
        ["robust_mean_mae", "candidate"], kind="mergesort"
    ).iloc[0]
    primary = overall.loc[overall["candidate"].eq(PRIMARY_CANDIDATE)].iloc[0]
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
        "candidate_count": len(PERSONAL_PROFILES),
        "reference_candidate": REFERENCE_CANDIDATE,
        "promotion_margin_mmhg": PROMOTION_MARGIN_MMHG,
        "primary_candidate": {
            "candidate": PRIMARY_CANDIDATE,
            "random_mean_mae": float(primary["mean_mae_random"]),
            "chronological_mean_mae": float(primary["mean_mae_chronological"]),
            "robust_mean_mae": float(primary["robust_mean_mae"]),
            "passes_robust_gate": bool(primary["passes_robust_gate"]),
        },
        "numerical_best": {
            "candidate": str(numerical_best["candidate"]),
            "robust_mean_mae": float(numerical_best["robust_mean_mae"]),
            "passes_robust_gate": bool(numerical_best["passes_robust_gate"]),
        },
        "selection_rule": (
            "The prespecified primary model must improve participant-macro mean MAE "
            "over paired rank-4 LoRA by at least 0.15 mmHg Overall in both split "
            "modes and improve both MIMIC and VitalDB in both modes. Other variants "
            "are mechanism and capacity controls, not replacements chosen after seeing results."
        ),
        "test_policy": (
            "No held-out target was accessed. A promotion decision requires the "
            "prespecified gate before any winner-only refit or one-time test."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    merged.sort_values(
        ["view", "robust_mean_mae", "candidate"], kind="mergesort"
    ).to_csv(output / "personal_profile_comparison.csv", index=False)
    save_json(output / "selection.json", result)
    lines = [
        "# Compact personal-profile screen",
        "",
        "Development-only persistent seen-participant personalization. Train and",
        "internal-validation participants overlap, but their 10-second windows are",
        "disjoint. The held-out role was not accessed.",
        "",
        f"- Seed: `{result['seed']}`",
        f"- Prespecified primary: `{PRIMARY_CANDIDATE}`",
        f"- Paired reference: `{REFERENCE_CANDIDATE}`",
        f"- Primary robust gate: `{str(result['primary_candidate']['passes_robust_gate']).lower()}`",
        f"- Numerical best: `{result['numerical_best']['candidate']}`",
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
            build_final_report(
                args.random_report, args.chronological_report, args.output
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
