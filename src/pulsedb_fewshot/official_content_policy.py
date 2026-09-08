"""Explicitly authorized retention of verified official-source duplicates.

Only the source-audited (waveform hash, event ID) pairs are accepted. This is
not a generic switch to disable lineage checks. Original/default protocols
remain strict, and duplicate IDs, new duplicates and target access still fail.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

POLICY = "official-source-duplicates-retained-20260908"
AUDIT_SHA256 = "ed3ce6fbc707ce4c4c635c9cf32a0dd25abb144ed4d54892947f6242376193ac"
POLICY_COLUMNS = ("official_content_policy", "official_duplicate_audit_path")
CLAIM_LIMIT = ("Exact official membership is retained, including verified source duplicates. "
               "Internal validation and OOF are index-disjoint, not strictly waveform-content-disjoint. "
               "No official test target row is accessed for fitting or adaptation.")


@lru_cache(maxsize=8)
def approved_groups(path_string, expected_hash=AUDIT_SHA256):
    path = Path(path_string)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("official duplicate audit checksum mismatch")
    report = json.loads(data)
    if report.get("test_bp_accessed") is not False or report.get("audit_status") != "verified_source_duplicates":
        raise ValueError("source duplicate audit is not label-free and verified")
    groups = {}
    for group in report["groups_private"]:
        if not all(group.get(key) is True for key in ("float64_equal", "raw_equal", "time_equal")):
            raise ValueError("unverified original-source duplicate")
        members = frozenset(str(r["segment_uid"]) for r in group["members"])
        if len(members) != group["rows"] or group["hash"] in groups:
            raise ValueError("ambiguous approved source duplicate group")
        groups[group["hash"]] = members
    return groups


def exceptions_for_rows(rows):
    """Require a uniform explicit policy plus the pinned private audit file."""
    marked = [r for r in rows if any(k in r for k in POLICY_COLUMNS)]
    if not marked:
        return None
    if len(marked) != len(rows) or any(r.get(POLICY_COLUMNS[0]) != POLICY for r in rows):
        raise ValueError("inconsistent official content policy")
    paths = {r.get(POLICY_COLUMNS[1]) for r in rows}
    if len(paths) != 1 or not next(iter(paths)):
        raise ValueError("missing or inconsistent official duplicate audit path")
    return approved_groups(str(next(iter(paths))))


def assert_allowed_duplicates(frame, *, hash_col="ppg_content_sha256", id_col="segment_uid", error="unapproved PPG content duplicate"):
    selected = frame.loc[frame[hash_col].duplicated(keep=False)]
    if selected.empty:
        return 0
    columns = [hash_col, id_col] + [k for k in POLICY_COLUMNS if k in frame]
    rows = selected[columns].to_dict("records")
    approved = exceptions_for_rows(rows)
    if approved is None:
        raise ValueError(error)
    seen = defaultdict(list)
    for row in rows:
        seen[str(row[hash_col])].append(str(row[id_col]))
    for content, ids in seen.items():
        if len(ids) != len(set(ids)) or not set(ids) <= approved.get(content, frozenset()):
            raise ValueError(error + "; not the verified official hash/identity pair")
    return len(seen)


def audit_official_metadata(bank_rows, query_rows):
    from .personal_memory_prepare import audit_metadata
    approved = exceptions_for_rows(bank_rows + query_rows)
    result = audit_metadata(bank_rows, query_rows, approved_content_groups=approved)
    if approved is not None:
        result.update(content_policy=POLICY, duplicate_audit_sha256=AUDIT_SHA256,
                      claim_limit=CLAIM_LIMIT)
    return result
