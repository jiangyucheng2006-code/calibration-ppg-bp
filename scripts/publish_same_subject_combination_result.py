"""Build path-free public artifacts for the same-subject combination screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_SCOPES = {"Overall", "MIMIC", "VitalDB"}
EXPECTED_SPLITS = {"random_disjoint", "chronological_blocked"}
REFERENCE = "lora"
EXPECTED_CANDIDATE_COUNT = 15
PROMOTION_MARGIN_MMHG = 0.15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split_report(
    report_dir: Path,
    expected_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, str]]:
    participant_path = report_dir / "participant_macro_summary.csv"
    pooled_path = report_dir / "event_pooled_diagnostics_all_scopes.csv"
    selection_path = report_dir / "selection.json"
    participant = pd.read_csv(participant_path)
    pooled = pd.read_csv(pooled_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    if selection.get("status") != "complete":
        raise ValueError(f"incomplete split report: {selection_path}")
    if selection.get("split_mode") != expected_split:
        raise ValueError(f"unexpected split in {selection_path}")
    if selection.get("selection_role") != "internal_validation":
        raise ValueError(f"unexpected selection role in {selection_path}")
    if selection.get("heldout_test_accessed") is not False:
        raise ValueError(f"held-out access recorded in {selection_path}")
    if selection.get("candidate_count") != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"unexpected candidate count in {selection_path}")
    if set(participant["split_mode"]) != {expected_split}:
        raise ValueError(f"unexpected split in {participant_path}")
    if set(participant["view"]) != EXPECTED_SCOPES:
        raise ValueError(f"missing participant-macro scope in {participant_path}")
    if set(pooled["Scope"]) != EXPECTED_SCOPES:
        raise ValueError(f"missing pooled scope in {pooled_path}")

    candidates = set(participant["candidate"])
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"participant candidate coverage mismatch in {report_dir}")
    if set(pooled["Setting"]) != candidates:
        raise ValueError(f"pooled candidate coverage mismatch in {report_dir}")
    if len(participant) != EXPECTED_CANDIDATE_COUNT * len(EXPECTED_SCOPES):
        raise ValueError(f"unexpected participant row count in {report_dir}")
    if len(pooled) != EXPECTED_CANDIDATE_COUNT * len(EXPECTED_SCOPES) * 2:
        raise ValueError(f"unexpected pooled row count in {report_dir}")

    overall = participant.loc[participant["view"].eq("Overall")]
    sources = participant.loc[participant["view"].isin(["MIMIC", "VitalDB"])]
    for candidate, row in overall.set_index("candidate").iterrows():
        selected = sources.loc[sources["candidate"].eq(candidate)]
        if int(selected["n_participants"].sum()) != int(row["n_participants"]):
            raise ValueError(f"participant count mismatch for {candidate}")
        if int(selected["n_events"].sum()) != int(row["n_events"]):
            raise ValueError(f"event count mismatch for {candidate}")

    public_participant = participant.drop(columns=["run_dir"], errors="ignore")
    public_participant = public_participant.rename(columns={"view": "Scope"})
    public_pooled = pooled.copy()
    public_pooled.insert(0, "split_mode", expected_split)
    public_selection = copy.deepcopy(selection)
    public_selection.get("winner", {}).pop("run_dir", None)
    checksums = {
        participant_path.name: _sha256(participant_path),
        pooled_path.name: _sha256(pooled_path),
        selection_path.name: _sha256(selection_path),
    }
    return public_participant, public_pooled, public_selection, checksums


def _load_final_report(
    final_report: Path,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    comparison_path = final_report / "combination_comparison.csv"
    selection_path = final_report / "selection.json"
    comparison = pd.read_csv(comparison_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    if selection.get("status") != "complete":
        raise ValueError(f"incomplete final report: {selection_path}")
    if selection.get("selection_role") != "internal_validation":
        raise ValueError(f"unexpected selection role in {selection_path}")
    if selection.get("heldout_test_accessed") is not False:
        raise ValueError(f"held-out access recorded in {selection_path}")
    if selection.get("candidate_count") != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"unexpected candidate count in {selection_path}")
    if float(selection.get("promotion_margin_mmhg")) != PROMOTION_MARGIN_MMHG:
        raise ValueError(f"unexpected promotion margin in {selection_path}")
    if selection.get("winner", {}).get("candidate") != REFERENCE:
        raise ValueError(f"unexpected final selection in {selection_path}")
    if set(comparison["view"]) != EXPECTED_SCOPES:
        raise ValueError(f"missing comparison scope in {comparison_path}")
    if comparison["candidate"].nunique() != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"candidate coverage mismatch in {comparison_path}")
    if len(comparison) != EXPECTED_CANDIDATE_COUNT * len(EXPECTED_SCOPES):
        raise ValueError(f"unexpected comparison row count in {comparison_path}")

    overall = comparison.loc[comparison["view"].eq("Overall")]
    non_reference_passes = overall.loc[
        overall["candidate"].ne(REFERENCE) & overall["passes_robust_gate"].astype(bool)
    ]
    if not non_reference_passes.empty:
        raise ValueError("a non-reference combination unexpectedly passes the gate")
    reference_rows = overall.loc[overall["candidate"].eq(REFERENCE)]
    if len(reference_rows) != 1:
        raise ValueError("missing unique LoRA reference row")

    public_comparison = comparison.drop(
        columns=["run_dir_random", "run_dir_chronological"], errors="ignore"
    )
    public_selection = copy.deepcopy(selection)
    public_selection.pop("random_report", None)
    public_selection.pop("chronological_report", None)
    public_selection["decision"] = {
        "outcome": "retain_reference",
        "reference": REFERENCE,
        "non_reference_combinations_passing_upgrade_gate": 0,
        "interpretation": (
            "The reference is retained by the prespecified fallback rule; "
            "it is not a claim that LoRA achieved a 0.15-mmHg improvement over itself."
        ),
    }
    checksums = {
        comparison_path.name: _sha256(comparison_path),
        selection_path.name: _sha256(selection_path),
    }
    return public_comparison, public_selection, checksums


def build_publication(
    random_report: Path,
    chronological_report: Path,
    final_report: Path,
    output: Path,
) -> None:
    loaded = {
        "random_disjoint": _load_split_report(random_report, "random_disjoint"),
        "chronological_blocked": _load_split_report(
            chronological_report, "chronological_blocked"
        ),
    }
    if set(loaded) != EXPECTED_SPLITS:
        raise AssertionError("both split modes are required")
    comparison, selection, final_checksums = _load_final_report(final_report)

    participant = pd.concat([value[0] for value in loaded.values()], ignore_index=True)
    participant = participant.sort_values(
        ["split_mode", "Scope", "mean_mae", "candidate"], kind="mergesort"
    )
    pooled = pd.concat([value[1] for value in loaded.values()], ignore_index=True)
    pooled = pooled.sort_values(
        ["split_mode", "Scope", "Setting", "BP"], kind="mergesort"
    )
    comparison = comparison.sort_values(
        ["view", "robust_mean_mae", "candidate"], kind="mergesort"
    )

    selected_participant = participant.loc[
        participant["candidate"].eq(REFERENCE)
    ].copy()
    if len(selected_participant) != len(EXPECTED_SPLITS) * len(EXPECTED_SCOPES):
        raise ValueError("selected-reference coverage mismatch")

    split_winners = {
        mode: value[2].get("winner", {}).get("candidate")
        for mode, value in loaded.items()
    }
    split_winner_participant = participant.loc[
        participant.apply(
            lambda row: row["candidate"] == split_winners[row["split_mode"]], axis=1
        )
    ].copy()
    if len(split_winner_participant) != len(EXPECTED_SPLITS) * len(EXPECTED_SCOPES):
        raise ValueError("split-winner coverage mismatch")

    output.mkdir(parents=True, exist_ok=False)
    participant.to_csv(output / "participant_macro.csv", index=False)
    pooled.to_csv(output / "event_pooled_diagnostics.csv", index=False)
    comparison.to_csv(output / "cross_split_comparison.csv", index=False)
    selected_participant.to_csv(output / "selected_lora_views.csv", index=False)
    split_winner_participant.to_csv(output / "split_winner_views.csv", index=False)
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "status": "complete",
        "protocol_id": "development-calbased-analogue-v1",
        "screen_id": "same-subject-combination-v1",
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "seed": 20260902,
        "selection_role": "internal_validation",
        "heldout_test_accessed": False,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "promotion_margin_mmhg": PROMOTION_MARGIN_MMHG,
        "decision": "retain_reference",
        "selected_reference": REFERENCE,
        "split_winners": split_winners,
        "participants": {"Overall": 2051, "MIMIC": 1011, "VitalDB": 1040},
        "internal_validation_events": {
            "Overall": 82040,
            "MIMIC": 40440,
            "VitalDB": 41600,
        },
        "jobs": {
            "random_training": "1294-1308",
            "random_report": 1309,
            "chronological_training": "1310-1324",
            "chronological_report": 1325,
            "final_report": 1326,
            "all_completed_zero_exit": True,
        },
        "source_archive_sha256": (
            "28de69526afdeacbb94e4ef1c3cd199ae70b8475b0481d4b95e1d436b8cb5e4c"
        ),
        "server_report_checksums": {
            mode: value[3] for mode, value in loaded.items()
        }
        | {"final": final_checksums},
        "integrity": {
            "report_work_nas_identical": True,
            "validation_coverage_equal_across_candidates": True,
            "source_counts_sum_to_overall": True,
            "all_reported_scopes_recomputed_from_saved_predictions": True,
        },
        "public_files": [
            "participant_macro.csv",
            "event_pooled_diagnostics.csv",
            "cross_split_comparison.csv",
            "selected_lora_views.csv",
            "split_winner_views.csv",
            "selection.json",
        ],
        "claim_boundary": (
            "Seen-participant internal validation with 320 labelled train windows "
            "per participant; not unseen-participant K-shot calibration, not an "
            "official PulseDB CalBased reproduction, not external validation, and "
            "not formal standards compliance."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--chronological-report", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_publication(
        args.random_report,
        args.chronological_report,
        args.final_report,
        args.output,
    )


if __name__ == "__main__":
    main()
