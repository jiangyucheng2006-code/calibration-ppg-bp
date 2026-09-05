import numpy as np
import pytest

from pulsedb_fewshot.personal_mechanism_diagnostic import grouped_derangement


def test_derangement_preserves_subject_membership_and_reproducibility():
    groups = np.repeat(["a", "b", "c"], [40, 5, 2])
    indices = grouped_derangement(groups, 27)
    assert np.array_equal(indices, grouped_derangement(groups, 27))
    assert sorted(indices) == list(range(len(groups)))
    assert np.all(indices != np.arange(len(groups)))
    assert np.array_equal(groups[indices], groups)


def test_single_member_is_not_silently_left_unperturbed():
    with pytest.raises(ValueError, match="at least two"):
        grouped_derangement(["a", "a", "b"], 27)
