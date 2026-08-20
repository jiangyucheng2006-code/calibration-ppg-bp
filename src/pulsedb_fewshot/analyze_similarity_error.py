"""Relate within-window beat similarity to BP prediction error.

This is a development-only diagnostic.  It never selects a deployment
threshold and it never reads the locked meta-test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .training import save_json


KEYS = ["subject_uid", "event_id"]
ERRORS = ["sbp_abs_error", "dbp_abs_error", "mean_abs_error"]
SIMILARITY_COLUMN = "pairwise_corr_median"
BIN_ORDER = ["<0.50", "0.50-0.80", "0.80-0.90", "0.90-0.95", ">=0.95"]


def _similarity_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, 0.50, 0.80, 0.90, 0.95, np.inf],
        labels=BIN_ORDER,
        right=False,
        ordered=True,
    )


def _spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return float("nan")
    xr = pair.x.rank(method="average").to_numpy(float)
    yr = pair.y.rank(method="average").to_numpy(float)
    if xr.std() <= 0 or yr.std() <= 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _bootstrap_correlation(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = frame[[x, y]].dropna().to_numpy(float)
    if len(values) < 3:
        return float("nan"), float("nan")
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = values[rng.integers(0, len(values), len(values))]
        estimates[index] = _spearman(
            pd.Series(sampled[:, 0]), pd.Series(sampled[:, 1])
        )
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, repetitions: int
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        estimates[index] = clean[rng.integers(0, len(clean), len(clean))].mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def _load_similarity(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(KEYS + ["source", "valid", SIMILARITY_COLUMN])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"similarity file missing columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError("similarity keys are not unique")
    if not frame.valid.all():
        raise ValueError("this audit requires a defined similarity for every query")
    result = frame[KEYS + ["source", SIMILARITY_COLUMN]].copy()
    if not np.isfinite(result[SIMILARITY_COLUMN]).all():
        raise ValueError("similarity contains non-finite values")
    return result


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(
        KEYS + ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    if "k" in frame:
        frame = frame.loc[frame.k.eq(5)].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError("prediction keys are not unique after selecting K=5")
    columns = KEYS + ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    if "source" in frame:
        columns.append("source")
    return frame[columns].copy()


def _scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("Overall", frame)] + [
        (source, frame.loc[frame.source.eq(source)])
        for source in sorted(frame.source.unique())
    ]


def _bin_metrics(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, scoped in _scopes(frame):
        for bin_name in BIN_ORDER:
            group = scoped.loc[scoped.similarity_bin.eq(bin_name)]
            if group.empty:
                continue
            participant = group.groupby("subject_uid", sort=False)[ERRORS].mean()
            rows.append(
                {
                    "setting": setting,
                    "scope": scope,
                    "similarity_bin": bin_name,
                    "windows": len(group),
                    "participants": group.subject_uid.nunique(),
                    "similarity_median": group[SIMILARITY_COLUMN].median(),
                    "sbp_mae": participant.sbp_abs_error.mean(),
                    "dbp_mae": participant.dbp_abs_error.mean(),
                    "mean_mae": participant.mean_abs_error.mean(),
                }
            )
    return pd.DataFrame(rows)


def _correlations(
    frame: pd.DataFrame,
    setting: str,
    *,
    seed: int,
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope_index, (scope, scoped) in enumerate(_scopes(frame)):
        participant = (
            scoped.groupby("subject_uid", sort=False)
            .agg(
                similarity=(SIMILARITY_COLUMN, "median"),
                sbp_abs_error=("sbp_abs_error", "mean"),
                dbp_abs_error=("dbp_abs_error", "mean"),
                mean_abs_error=("mean_abs_error", "mean"),
            )
            .reset_index()
        )
        for error_index, error in enumerate(ERRORS):
            estimate = _spearman(participant.similarity, participant[error])
            low, high = _bootstrap_correlation(
                participant,
                "similarity",
                error,
                seed=seed + scope_index * 100 + error_index,
                repetitions=repetitions,
            )
            rows.append(
                {
                    "setting": setting,
                    "scope": scope,
                    "level": "participant",
                    "outcome": error,
                    "n": len(participant),
                    "spearman_rho": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
            rows.append(
                {
                    "setting": setting,
                    "scope": scope,
                    "level": "event_descriptive_only",
                    "outcome": error,
                    "n": len(scoped),
                    "spearman_rho": _spearman(scoped[SIMILARITY_COLUMN], scoped[error]),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _within_participant_contrast(
    frame: pd.DataFrame,
    setting: str,
    *,
    seed: int,
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope_index, (scope, scoped) in enumerate(_scopes(frame)):
        scoped = scoped.assign(
            similarity_group=np.where(
                scoped[SIMILARITY_COLUMN] < 0.90, "below_0.90", "at_least_0.90"
            )
        )
        grouped = (
            scoped.groupby(["subject_uid", "similarity_group"], sort=False)[ERRORS]
            .mean()
            .unstack("similarity_group")
        )
        for error_index, error in enumerate(ERRORS):
            if error not in grouped:
                continue
            pair = grouped[error].dropna()
            low_similarity = pair["below_0.90"]
            high_similarity = pair["at_least_0.90"]
            difference = low_similarity - high_similarity
            low, high = _bootstrap_mean(
                difference.to_numpy(),
                seed=seed + scope_index * 100 + error_index,
                repetitions=repetitions,
            )
            rows.append(
                {
                    "setting": setting,
                    "scope": scope,
                    "outcome": error,
                    "paired_participants": len(difference),
                    "low_similarity_mae": low_similarity.mean(),
                    "high_similarity_mae": high_similarity.mean(),
                    "low_minus_high_mmHg": difference.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "threshold_note": "0.90 is descriptive, not a deployment cutoff",
                }
            )
    return pd.DataFrame(rows)


def analyze(
    similarity_path: Path,
    runs: list[tuple[str, Path]],
    output: Path,
    *,
    seed: int = 20260820,
    repetitions: int = 5000,
) -> dict[str, object]:
    similarity = _load_similarity(similarity_path)
    reference_keys: pd.DataFrame | None = None
    frames: dict[str, pd.DataFrame] = {}
    for setting, path in runs:
        predictions = _load_predictions(path)
        keys = predictions[KEYS + ["target_sbp", "target_dbp"]].sort_values(
            KEYS, kind="mergesort"
        )
        if reference_keys is None:
            reference_keys = keys.reset_index(drop=True)
        elif not keys.reset_index(drop=True).equals(reference_keys):
            raise ValueError(f"{setting} does not use the identical query set and targets")
        merged = predictions.merge(
            similarity, on=KEYS, suffixes=("", "_similarity"), validate="one_to_one"
        )
        if len(merged) != len(similarity):
            raise ValueError(f"{setting} does not cover every similarity row")
        if "source_similarity" in merged:
            if not merged.source.eq(merged.source_similarity).all():
                raise ValueError(f"{setting} source mismatch")
            merged = merged.drop(columns="source_similarity")
        merged["sbp_abs_error"] = (merged.target_sbp - merged.pred_sbp).abs()
        merged["dbp_abs_error"] = (merged.target_dbp - merged.pred_dbp).abs()
        merged["mean_abs_error"] = (
            merged.sbp_abs_error + merged.dbp_abs_error
        ) / 2
        merged["similarity_bin"] = _similarity_bin(merged[SIMILARITY_COLUMN])
        frames[setting] = merged

    bin_metrics = pd.concat(
        [_bin_metrics(frame, setting) for setting, frame in frames.items()],
        ignore_index=True,
    )
    correlations = pd.concat(
        [
            _correlations(
                frame,
                setting,
                seed=seed + index * 1000,
                repetitions=repetitions,
            )
            for index, (setting, frame) in enumerate(frames.items())
        ],
        ignore_index=True,
    )
    contrasts = pd.concat(
        [
            _within_participant_contrast(
                frame,
                setting,
                seed=seed + index * 1000,
                repetitions=repetitions,
            )
            for index, (setting, frame) in enumerate(frames.items())
        ],
        ignore_index=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    bin_metrics.to_csv(output / "similarity_bins.csv", index=False)
    correlations.to_csv(output / "correlations.csv", index=False)
    contrasts.to_csv(output / "within_participant_contrast.csv", index=False)

    payload: dict[str, object] = {
        "status": "complete",
        "split": "meta_validation",
        "locked_test_accessed": False,
        "k": 5,
        "windows": int(len(similarity)),
        "participants": int(similarity.subject_uid.nunique()),
        "settings": list(frames),
        "similarity_metric": SIMILARITY_COLUMN,
        "primary_analysis": "participant-level Spearman correlation with participant bootstrap CI",
        "secondary_analysis": "within-participant mean MAE below versus at least 0.90 similarity",
        "claim_limit": "exploratory development analysis; 0.90 is descriptive and may not be selected as a deployment threshold from these results",
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }
    save_json(output / "run.json", payload)

    primary = correlations.loc[
        correlations.level.eq("participant")
        & correlations.outcome.eq("mean_abs_error")
    ]
    paired = contrasts.loc[contrasts.outcome.eq("mean_abs_error")]
    lines = [
        "# Beat similarity and BP prediction error",
        "",
        "Development-only exploratory analysis; locked meta-test was not accessed.",
        "Negative correlation means higher similarity tends to accompany lower error.",
        "",
        "## Participant-level association",
        "",
        "| Setting | Scope | Participants | Spearman rho | 95% bootstrap CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary.to_dict("records"):
        lines.append(
            f"| {row['setting']} | {row['scope']} | {row['n']} | "
            f"{row['spearman_rho']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Within-participant comparison",
            "",
            "| Setting | Scope | Paired participants | MAE <0.90 | MAE >=0.90 | Difference | 95% bootstrap CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired.to_dict("records"):
        lines.append(
            f"| {row['setting']} | {row['scope']} | {row['paired_participants']} | "
            f"{row['low_similarity_mae']:.4f} | {row['high_similarity_mae']:.4f} | "
            f"{row['low_minus_high_mmHg']:+.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be SETTING=PATH")
    setting, path = value.split("=", 1)
    if not setting or not path:
        raise argparse.ArgumentTypeError("--run must be SETTING=PATH")
    return setting, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze(
                arguments.similarity,
                arguments.run,
                arguments.output,
                seed=arguments.seed,
                repetitions=arguments.bootstrap_repetitions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
