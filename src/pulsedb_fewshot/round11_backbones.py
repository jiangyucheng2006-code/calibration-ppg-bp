"""Leakage-safe Round-11A PPG-backbone screening and reporting.

All candidates fit on meta-train folds 0--2, use fold 3 only for patience-8
early stopping, and are ranked on fold 4.  Meta-validation and the locked
meta-test are outside this module's data boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .round10_end_to_end import (
    EARLY_FOLD,
    FIT_FOLDS,
    KEYS,
    SELECTION_FOLD,
    _diagnostic_rows,
    _markdown_table,
    prepare_round10,
)
from .training import file_sha256, participant_macro_metrics, save_json


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_training_run(
    root: Path, *, method: str, backbone: str
) -> dict[str, object]:
    payload = _read_json(root / "run.json")
    if payload.get("status") != "complete" or payload.get("method") != method:
        raise AssertionError(f"{root} is not a complete {method} run")
    if payload.get("backbone") != backbone:
        raise AssertionError(f"{root} has the wrong backbone")
    if payload.get("crossfit_fit_folds") != list(FIT_FOLDS):
        raise AssertionError(f"{root} has the wrong fit folds")
    if payload.get("crossfit_validation_fold") != EARLY_FOLD:
        raise AssertionError(f"{root} has the wrong early-stopping fold")
    if payload.get("crossfit_excluded_folds") != [SELECTION_FOLD]:
        raise AssertionError(f"{root} did not quarantine selection fold")
    if payload.get("locked_test_accessed") is not False:
        raise AssertionError(f"{root} accessed locked test")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise AssertionError(f"{root} lacks arguments")
    if method == "m0":
        required = {
            "train_support_policy": "fixed_first",
            "loss": "huber",
            "use_quality_gate": True,
            "ks": [5],
        }
        for key, value in required.items():
            if arguments.get(key) != value:
                raise AssertionError(f"{root} has unsafe/mismatched {key}")
    return payload


def evaluate_backbone(
    *,
    backbone: str,
    store_root: Path,
    folds: Path,
    population_run: Path,
    qgh_run: Path,
    output: Path,
    batch_size: int = 512,
) -> dict[str, object]:
    population = _validate_training_run(
        population_run, method="population", backbone=backbone
    )
    qgh = _validate_training_run(qgh_run, method="m0", backbone=backbone)
    if population.get("seed") != qgh.get("seed"):
        raise AssertionError("population and QGH seeds differ")
    result = prepare_round10(
        store_root=store_root,
        folds_path=folds,
        population_checkpoint=population_run / "best.pt",
        qgh_checkpoint=qgh_run / "best.pt",
        output=output,
        batch_size=batch_size,
        round_number=11,
        stage="backbone_evaluation",
        backbone=backbone,
    )
    result.update(
        {
            "seed": int(population["seed"]),
            "population_run_sha256": file_sha256(population_run / "run.json"),
            "qgh_run_sha256": file_sha256(qgh_run / "run.json"),
            "population_parameter_counts": population.get("parameter_counts"),
            "qgh_parameter_counts": qgh.get("parameter_counts"),
            "meta_validation_used_for_training": False,
            "meta_validation_used_for_early_stopping": False,
            "meta_validation_used_for_candidate_ranking": False,
            "meta_validation_predictions_generated": False,
        }
    )
    save_json(output / "run.json", result)
    return result


def _prediction_view(
    frame: pd.DataFrame, *, population: bool
) -> pd.DataFrame:
    result = frame.copy()
    if population:
        result["pred_sbp"] = result["population_pred_sbp"]
        result["pred_dbp"] = result["population_pred_dbp"]
    return result


def build_report(
    *,
    runs: dict[str, Path],
    reference_backbone: str,
    output: Path,
    expected_seed: int,
) -> dict[str, object]:
    if reference_backbone not in runs:
        raise KeyError(reference_backbone)
    canonical: pd.DataFrame | None = None
    records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    complexity: list[dict[str, object]] = []
    for backbone, root in runs.items():
        payload = _read_json(root / "run.json")
        if payload.get("status") != "complete" or payload.get("round") != 11:
            raise AssertionError(f"{backbone} is not a complete Round-11 run")
        if payload.get("stage") != "backbone_evaluation":
            raise AssertionError(f"{backbone} has the wrong stage")
        if payload.get("backbone") != backbone or payload.get("seed") != expected_seed:
            raise AssertionError(f"{backbone} metadata mismatch")
        for key in (
            "meta_validation_accessed",
            "meta_validation_used_for_training",
            "meta_validation_used_for_early_stopping",
            "meta_validation_used_for_candidate_ranking",
            "meta_validation_predictions_generated",
            "locked_test_accessed",
            "query_bp_model_input",
            "future_query_model_input",
            "source_model_input",
        ):
            if payload.get(key) is not False:
                raise AssertionError(f"{backbone} has unsafe flag {key}")
        frame = pd.read_parquet(root / "queries.parquet")
        frame = frame.loc[frame["fold"].eq(SELECTION_FOLD)].copy()
        if frame.empty or frame["split"].ne("meta_train").any():
            raise AssertionError(f"{backbone} selection rows are invalid")
        checked = frame[KEYS + ["target_sbp", "target_dbp"]].sort_values(KEYS)
        checked.reset_index(drop=True, inplace=True)
        if canonical is None:
            canonical = checked
        elif not canonical.equals(checked):
            raise AssertionError(f"{backbone} has different query keys or targets")
        for model_kind, population in (("Population", True), ("QGH", False)):
            setting = f"{backbone} | {model_kind}"
            predictions = _prediction_view(frame, population=population)
            prediction_frames.append(predictions.assign(Setting=setting))
            for scope, group in [("Overall", predictions)] + [
                (source, predictions.loc[predictions["source"].eq(source)])
                for source in sorted(predictions["source"].unique())
            ]:
                metric = participant_macro_metrics(group)
                records.append(
                    {
                        "Backbone": backbone,
                        "Model": model_kind,
                        "Setting": setting,
                        "Scope": scope,
                        "N participants": int(group["subject_uid"].nunique()),
                        "N queries": int(len(group)),
                        "SBP participant-macro MAE": metric["sbp_mae"],
                        "DBP participant-macro MAE": metric["dbp_mae"],
                        "Mean participant-macro MAE": metric["mean_mae"],
                    }
                )
        pop_counts = payload.get("population_parameter_counts") or {}
        qgh_counts = payload.get("qgh_parameter_counts") or {}
        complexity.append(
            {
                "Backbone": backbone,
                "Population parameters": pop_counts.get("total"),
                "QGH total parameters": qgh_counts.get("total"),
                "QGH trainable parameters": qgh_counts.get("trainable"),
            }
        )

    table = pd.DataFrame(records)
    complexity_table = pd.DataFrame(complexity)
    settings = [str(predictions["Setting"].iloc[0]) for predictions in prediction_frames]
    diagnostics = _diagnostic_rows(
        pd.concat(prediction_frames, ignore_index=True, sort=False),
        list(dict.fromkeys(settings)),
    )
    qgh_overall = table.loc[
        table["Model"].eq("QGH") & table["Scope"].eq("Overall")
    ].sort_values(["Mean participant-macro MAE", "Backbone"], kind="mergesort")
    winner = str(qgh_overall.iloc[0]["Backbone"])
    reference = table.loc[
        table["Backbone"].eq(reference_backbone) & table["Model"].eq("QGH")
    ].set_index("Scope")
    winning = table.loc[
        table["Backbone"].eq(winner) & table["Model"].eq("QGH")
    ].set_index("Scope")
    scopes = ("Overall", "MIMIC", "VitalDB")
    gains = {
        scope: float(
            reference.loc[scope, "Mean participant-macro MAE"]
            - winning.loc[scope, "Mean participant-macro MAE"]
        )
        for scope in scopes
    }
    passes = gains["Overall"] >= 0.15 and all(gains[scope] > 0 for scope in scopes[1:])
    comparison_rows: list[dict[str, object]] = []
    for backbone in runs:
        candidate = table.loc[
            table["Backbone"].eq(backbone) & table["Model"].eq("QGH")
        ].set_index("Scope")
        for scope in scopes:
            comparison_rows.append(
                {
                    "Backbone": backbone,
                    "Scope": scope,
                    "Reference mean participant-macro MAE": float(
                        reference.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate mean participant-macro MAE": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate minus reference": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                        - reference.loc[scope, "Mean participant-macro MAE"]
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "participant_macro_internal.csv", index=False)
    diagnostics.to_csv(output / "pooled_diagnostics_internal.csv", index=False)
    comparison.to_csv(output / "comparison_vs_reference_internal.csv", index=False)
    complexity_table.to_csv(output / "model_complexity.csv", index=False)
    summary = {
        "status": "complete",
        "round": 11,
        "stage": "backbone_screen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": expected_seed,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "meta_validation_used_for_candidate_ranking": False,
        "locked_test_accessed": False,
        "reference_backbone": reference_backbone,
        "winner_backbone": winner,
        "gain_vs_reference": gains,
        "passes_internal_gate": passes,
        "candidate_count": len(runs),
    }
    save_json(output / "selection.json", summary)
    lines = [
        "# Round-11A PPG-backbone internal screen",
        "",
        "All models use folds 0--2 for fitting, fold 3 for patience-8 early stopping, and fold 4 for candidate ranking. Meta-validation and the locked meta-test are not accessed.",
        "",
        "## Participant-macro results",
        "",
        _markdown_table(table.sort_values(["Model", "Scope", "Mean participant-macro MAE"])),
        "",
        "## QGH change versus the current ResNet reference",
        "",
        "Negative candidate-minus-reference values are better.",
        "",
        _markdown_table(comparison),
        "",
        "## Model complexity",
        "",
        _markdown_table(complexity_table),
        "",
        "## Event-pooled diagnostics",
        "",
        "Participant-macro MAE is primary. AAMI/BHS entries are retrospective numerical screens only.",
        "",
        _markdown_table(diagnostics),
        "",
        f"Internal numerical winner: **{winner}**.",
        f"Internal promotion gate passed: **{passes}**.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _parse_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run must use BACKBONE=PATH")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate backbone {name}")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--backbone", required=True)
    evaluate.add_argument("--store-root", type=Path, required=True)
    evaluate.add_argument("--folds", type=Path, required=True)
    evaluate.add_argument("--population-run", type=Path, required=True)
    evaluate.add_argument("--qgh-run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=512)
    report = commands.add_parser("report")
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--reference-backbone", default="resnet_small")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--expected-seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_backbone(
            backbone=args.backbone,
            store_root=args.store_root,
            folds=args.folds,
            population_run=args.population_run,
            qgh_run=args.qgh_run,
            output=args.output,
            batch_size=args.batch_size,
        )
    else:
        result = build_report(
            runs=_parse_runs(args.run),
            reference_backbone=args.reference_backbone,
            output=args.output,
            expected_seed=args.expected_seed,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
