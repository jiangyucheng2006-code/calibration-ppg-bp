"""Train-only personal-neighbour diagnostic infrastructure (NumPy only).

This is NOT a trained novel model or an end-to-end leakage certificate.
Encoder/checkpoint/split lineage, near-duplicate waveform detection, identity
aliases, and timestamp origins must be audited upstream. Hashes must identify
canonical waveform content, not merely differently encoded source files.

The query interface accepts features, metadata and optionally existing BP
predictions, NEVER query BP labels. Retrieval uses a fixed uniform top-k mean;
there is no learned gate or automatic blend/hyperparameter selection. Residual
correction based on in-sample training predictions is an exploratory diagnostic,
not out-of-fold evidence that a learned correction will generalize.
"""

from dataclasses import dataclass
from typing import Literal, Sequence
import re

import numpy as np


@dataclass(frozen=True)
class WindowMetadata:
    """Explicit lineage; start/end are half-open seconds on time_axis_uid.

    Recording IDs must be globally canonical within a source; subject IDs must
    be canonical across the complete development store. A time axis identifies
    an actual comparable clock, not simply the units 'seconds'.
    """

    subject_uid: str
    source: str
    window_uid: str
    waveform_sha256: str
    recording_uid: str
    time_axis_uid: str
    start_s: float
    end_s: float
    role: str


@dataclass(frozen=True)
class ProbePrediction:
    bp_interpolation: np.ndarray
    neighbour_indices: np.ndarray
    neighbour_similarity: np.ndarray
    residual_corrected_bp: np.ndarray | None
    residual_provenance: str | None


def _readonly(array: np.ndarray) -> np.ndarray:
    """Bytes-backed copy cannot be made writeable by setflags(write=True)."""
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _matrix(values, rows: int, columns: int | None, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != rows or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape ({rows}, D>0)")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have exactly {columns} columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _unit_features(values, rows: int, dimension: int | None) -> np.ndarray:
    features = _matrix(values, rows, dimension, "features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("cosine similarity requires finite, non-zero feature vectors")
    return features / norms


def _metadata(rows: Sequence[WindowMetadata], expected_role: str) -> tuple[WindowMetadata, ...]:
    result = tuple(rows)
    if not result:
        raise ValueError("metadata must contain at least one window")
    ids, hashes = set(), set()
    for row in result:
        if not isinstance(row, WindowMetadata):
            raise ValueError("explicit WindowMetadata is required for every window")
        for field in ("subject_uid", "source", "window_uid", "recording_uid", "time_axis_uid"):
            value = getattr(row, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"non-empty {field} is required")
        if row.role != expected_role:
            raise ValueError(f"role must be exactly {expected_role}; got {row.role!r}")
        if not isinstance(row.waveform_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", row.waveform_sha256):
            raise ValueError("waveform_sha256 must be a lowercase canonical SHA-256")
        if not np.isfinite([row.start_s, row.end_s]).all() or row.end_s <= row.start_s:
            raise ValueError("window interval must be finite and have positive duration")
        if row.window_uid in ids or row.waveform_sha256 in hashes:
            raise ValueError("duplicate window ID or waveform hash in metadata")
        ids.add(row.window_uid)
        hashes.add(row.waveform_sha256)
    return result


class FrozenPersonalMemory:
    """Read-only train memory for known-user, internal-validation probes.

    Input BP/predictions have shape (N, 2), ordered SBP/DBP in mmHg. Optional
    prediction_provenance must be 'in_sample' or 'crossfit'. This flag records
    the caller's declaration; it cannot verify the upstream training history.
    No fit/update method or validation-label adaptation is provided.
    """

    def __init__(self, features, bp, metadata: Sequence[WindowMetadata], *,
                 train_predictions=None,
                 prediction_provenance: Literal["in_sample", "crossfit"] | None = None):
        self._metadata = _metadata(metadata, "train")
        self._features = _readonly(_unit_features(features, len(self._metadata), None))
        self._bp = _readonly(_matrix(bp, len(self._metadata), 2, "BP"))
        self._residuals = None
        self._provenance = prediction_provenance
        if train_predictions is None:
            if prediction_provenance is not None:
                raise ValueError("prediction provenance requires train predictions")
        else:
            if prediction_provenance not in {"in_sample", "crossfit"}:
                raise ValueError("train predictions require explicit in_sample/crossfit provenance")
            predictions = _matrix(train_predictions, len(self._metadata), 2, "train predictions")
            self._residuals = _readonly(self._bp - predictions)
        self._ids = frozenset(row.window_uid for row in self._metadata)
        self._hashes = frozenset(row.waveform_sha256 for row in self._metadata)
        self._subjects: dict[str, tuple[str, np.ndarray]] = {}
        self._recordings: dict[tuple[str, str], tuple[str, np.ndarray, np.ndarray]] = {}
        subject_groups: dict[str, list[int]] = {}
        recording_groups: dict[tuple[str, str], list[WindowMetadata]] = {}
        for i, row in enumerate(self._metadata):
            subject_groups.setdefault(row.subject_uid, []).append(i)
            recording_groups.setdefault((row.source, row.recording_uid), []).append(row)
        for subject, grouped_indices in subject_groups.items():
            indices = np.array(grouped_indices)
            sources = {self._metadata[i].source for i in indices}
            if len(sources) != 1:
                raise ValueError("one canonical subject cannot have conflicting sources")
            self._subjects[subject] = (sources.pop(), _readonly(indices))
        for key, rows in recording_groups.items():
            axes = {row.time_axis_uid for row in rows}
            if len(axes) != 1:
                raise ValueError("one recording cannot have inconsistent timestamp axes")
            self._recordings[key] = (axes.pop(), _readonly(np.array([r.start_s for r in rows])),
                                    _readonly(np.array([r.end_s for r in rows])))

    @property
    def metadata(self) -> tuple[WindowMetadata, ...]:
        return self._metadata

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def bp(self) -> np.ndarray:
        return self._bp

    def predict(self, query_features, query_metadata: Sequence[WindowMetadata], *,
                k: int = 5, mode: Literal["random_disjoint", "chronological_blocked"] = "random_disjoint",
                base_predictions=None) -> ProbePrediction:
        """Return uniform top-k BP and optional fixed residual correction.

        Exact ID/hash and recording-interval conflicts are checked globally,
        before restricting neighbours to the correct person. Half-open adjacent
        windows are permitted. Chronological mode requires comparable clocks and
        excludes every memory window ending after the query starts. Insufficient
        history raises an error instead of silently changing k or query coverage.
        """
        queries = _metadata(query_metadata, "internal_validation")
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if mode not in {"random_disjoint", "chronological_blocked"}:
            raise ValueError("unsupported split mode")
        features = _unit_features(query_features, len(queries), self._features.shape[1])
        base = None
        if base_predictions is not None:
            if self._residuals is None:
                raise ValueError("residual correction needs provenance-qualified train predictions")
            base = _matrix(base_predictions, len(queries), 2, "base predictions")
        interpolation, neighbours, similarities, corrections = [], [], [], []
        for i, query in enumerate(queries):
            if query.window_uid in self._ids or query.waveform_sha256 in self._hashes:
                raise ValueError("query matches a train window ID or waveform hash")
            recording = self._recordings.get((query.source, query.recording_uid))
            if recording is not None:
                axis, starts, ends = recording
                if axis != query.time_axis_uid:
                    raise ValueError("recording timestamp axes are inconsistent")
                if np.any((starts < query.end_s) & (ends > query.start_s)):
                    raise ValueError("query physiological interval overlaps train memory")
            if query.subject_uid not in self._subjects:
                raise ValueError("unseen participant: no personal memory is available")
            source, indices = self._subjects[query.subject_uid]
            if source != query.source:
                raise ValueError("query source differs from its registered participant")
            if mode == "chronological_blocked":
                if any(self._metadata[j].time_axis_uid != query.time_axis_uid for j in indices):
                    raise ValueError("chronological retrieval requires comparable timestamp axes")
                indices = indices[np.array([self._metadata[j].end_s <= query.start_s for j in indices])]
            if len(indices) < k:
                raise ValueError("fewer than k eligible personal training windows")
            scores = np.clip(self._features[indices] @ features[i], -1.0, 1.0)
            tie_ids = np.array([self._metadata[j].window_uid for j in indices])
            order = np.lexsort((tie_ids, -scores))[:k]
            selected = indices[order]
            interpolation.append(self._bp[selected].mean(axis=0))
            neighbours.append(selected)
            similarities.append(scores[order])
            if base is not None:
                corrections.append(base[i] + self._residuals[selected].mean(axis=0))
        return ProbePrediction(_readonly(np.array(interpolation)), _readonly(np.array(neighbours)),
                               _readonly(np.array(similarities)),
                               _readonly(np.array(corrections)) if base is not None else None,
                               self._provenance if base is not None else None)
