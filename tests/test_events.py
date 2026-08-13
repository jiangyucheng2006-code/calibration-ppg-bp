import pandas as pd

from pulsedb_fewshot.events import (
    build_episode_assignments,
    eventize_segments,
    summarize_eligibility,
)


def _segments(subject_id: str, times: list[float], *, source: str = "MIMIC") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "record_id": f"{subject_id}_record",
            "record_order": 0,
            "source": source,
            "segment_id": [f"{subject_id}_{i}" for i in range(len(times))],
            "start_time_s": times,
            "sbp": [110 + i for i in range(len(times))],
            "dbp": [65 + i / 2 for i in range(len(times))],
        }
    )


def test_adjacent_segments_do_not_inflate_event_count() -> None:
    segments = _segments("s1", [0, 10, 20, 60, 70, 120])
    events = eventize_segments(segments, bin_width_sec=60)

    assert len(events) == 3
    assert events["n_segments_in_bin"].tolist() == [3, 2, 1]
    assert events["event_index"].tolist() == [1, 2, 3]


def test_eligibility_requires_common_future_queries() -> None:
    eligible_segments = _segments("eligible", [60 * i for i in range(10)])
    ineligible_segments = _segments("short", [60 * i for i in range(9)], source="VitalDB")
    events = eventize_segments(pd.concat([eligible_segments, ineligible_segments]), bin_width_sec=60)
    summary = summarize_eligibility(events, max_k=5, min_query_events=5)

    flags = summary.set_index("subject_id")["eligible"].to_dict()
    assert bool(flags["eligible"])
    assert not bool(flags["short"])


def test_all_k_values_share_the_same_query_events() -> None:
    segments = _segments("s1", [60 * i for i in range(10)])
    events = eventize_segments(segments, bin_width_sec=60)
    assignments = build_episode_assignments(events, ks=(1, 2, 3, 5), min_query_events=5)

    query_sets = {
        k: tuple(group.loc[group["role"] == "query", "event_id"])
        for k, group in assignments.groupby("k")
    }
    assert len(set(query_sets.values())) == 1
    assert len(query_sets[1]) == 5

    k1 = assignments[assignments["k"] == 1]["role"].value_counts().to_dict()
    k5 = assignments[assignments["k"] == 5]["role"].value_counts().to_dict()
    assert k1 == {"query": 5, "unused_calibration_pool": 4, "support": 1}
    assert k5 == {"support": 5, "query": 5}
