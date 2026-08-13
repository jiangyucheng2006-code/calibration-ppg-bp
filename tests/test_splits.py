import pandas as pd

from pulsedb_fewshot.splits import assign_subject_splits, assert_disjoint_subject_splits


def test_subject_splits_are_deterministic_and_disjoint() -> None:
    subjects = pd.DataFrame(
        {
            "subject_id": [f"m{i:03d}" for i in range(20)] + [f"v{i:03d}" for i in range(20)],
            "source": ["MIMIC"] * 20 + ["VitalDB"] * 20,
        }
    )

    first = assign_subject_splits(subjects, seed=7)
    second = assign_subject_splits(subjects, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert_disjoint_subject_splits(first)
    assert set(first["split"]) == {"meta_train", "meta_validation", "meta_test"}
    assert first.groupby("subject_id")["split"].nunique().max() == 1
