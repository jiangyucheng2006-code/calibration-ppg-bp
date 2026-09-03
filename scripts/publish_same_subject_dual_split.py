"""Build public, path-free tables for the same-subject dual-split screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_SCOPES = {"Overall", "MIMIC", "VitalDB"}
EXPECTED_SPLITS = {"random_disjoint", "chronological_blocked"}
WINNER = "subject_mean_residual_ppg"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_report(
    report_dir: Path,
    expected_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    participant_path = report_dir / "participant_macro_summary.csv"
    pooled_path = report_dir / "event_pooled_diagnostics_all_scopes.csv"
    selection_path = report_dir / "selection.json"
    participant = pd.read_csv(participant_path)
    pooled = pd.read_csv(pooled_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    if set(participant["split_mode"]) != {expected_split}:
        raise ValueError(f"unexpected split in {participant_path}")
    if set(participant["view"]) != EXPECTED_SCOPES:
        raise ValueError(f"missing participant-macro scope in {participant_path}")
    if set(pooled["Scope"]) != EXPECTED_SCOPES:
        raise ValueError(f"missing pooled scope in {pooled_path}")
    if selection.get("split_mode") != expected_split:
        raise ValueError(f"unexpected split in {selection_path}")
    if selection.get("heldout_test_accessed") is not False:
        raise ValueError(f"held-out access recorded in {selection_path}")
    if selection.get("winner", {}).get("candidate") != WINNER:
        raise ValueError(f"unexpected winner in {selection_path}")

    candidates = set(participant["candidate"])
    if len(candidates) != 9 or set(pooled["Setting"]) != candidates:
        raise ValueError(f"candidate coverage mismatch in {report_dir}")
    overall = participant.loc[participant["view"].eq("Overall")]
    sources = participant.loc[participant["view"].isin(["MIMIC", "VitalDB"])]
    for candidate, row in overall.set_index("candidate").iterrows():
        selected = sources.loc[sources["candidate"].eq(candidate)]
        if int(selected["n_participants"].sum()) != int(row["n_participants"]):
            raise ValueError(f"participant count mismatch for {candidate}")
        if int(selected["n_events"].sum()) != int(row["n_events"]):
            raise ValueError(f"event count mismatch for {candidate}")

    participant = participant.drop(columns=["run_dir"], errors="ignore").rename(
        columns={"view": "Scope"}
    )
    pooled.insert(0, "split_mode", expected_split)
    public_selection = copy.deepcopy(selection)
    public_selection.get("winner", {}).pop("run_dir", None)
    checksums = {
        participant_path.name: _sha256(participant_path),
        pooled_path.name: _sha256(pooled_path),
        selection_path.name: _sha256(selection_path),
    }
    return participant, pooled, {
        "selection": public_selection,
        "checksums": checksums,
    }


def build_publication(
    random_report: Path,
    chronological_report: Path,
    output: Path,
) -> None:
    loaded = {
        "random_disjoint": _load_report(random_report, "random_disjoint"),
        "chronological_blocked": _load_report(
            chronological_report, "chronological_blocked"
        ),
    }
    if set(loaded) != EXPECTED_SPLITS:
        raise AssertionError("both split modes are required")

    participant = pd.concat([value[0] for value in loaded.values()], ignore_index=True)
    participant = participant.sort_values(
        ["split_mode", "Scope", "mean_mae", "candidate"], kind="mergesort"
    )
    pooled = pd.concat([value[1] for value in loaded.values()], ignore_index=True)
    pooled = pooled.sort_values(
        ["split_mode", "Scope", "Setting", "BP"], kind="mergesort"
    )

    winner_participant = participant.loc[
        participant["candidate"].eq(WINNER)
        & participant["Scope"].isin(["MIMIC", "VitalDB"])
    ].copy()
    winner_pooled = pooled.loc[
        pooled["Setting"].eq(WINNER)
        & pooled["Scope"].isin(["MIMIC", "VitalDB"])
    ].copy()
    four_view = winner_pooled.merge(
        winner_participant,
        on=["split_mode", "Scope"],
        how="inner",
        validate="many_to_one",
    )
    if len(four_view) != 8:
        raise ValueError("expected two BP endpoints for each of four source/split views")
    four_view = four_view[
        [
            "split_mode",
            "Scope",
            "Setting",
            "BP",
            "n_participants",
            "n_events",
            "sbp_mae",
            "dbp_mae",
            "mean_mae",
            "MAE",
            "R²",
            "ME",
            "STD",
            "≤5 mmHg",
            "≤10 mmHg",
            "≤15 mmHg",
            "AAMI",
            "BHS",
        ]
    ].sort_values(["split_mode", "Scope", "BP"], kind="mergesort")

    output.mkdir(parents=True, exist_ok=False)
    participant.to_csv(output / "participant_macro.csv", index=False)
    pooled.to_csv(output / "event_pooled_diagnostics.csv", index=False)
    four_view.to_csv(output / "four_view_winner.csv", index=False)
    manifest = {
        "protocol_id": "development-calbased-analogue-v1",
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "seed": 20260828,
        "selection_role": "internal_validation",
        "heldout_test_accessed": False,
        "winner": WINNER,
        "split_reports": {mode: value[2] for mode, value in loaded.items()},
        "public_files": [
            "participant_macro.csv",
            "event_pooled_diagnostics.csv",
            "four_view_winner.csv",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--chronological-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_publication(args.random_report, args.chronological_report, args.output)


if __name__ == "__main__":
    main()
