"""Analytic infrastructure tests; no patient data, model fitting or torch."""

from dataclasses import replace
import hashlib
import inspect
import unittest

import numpy as np

from pulsedb_fewshot.personal_memory_probe import FrozenPersonalMemory, WindowMetadata


def raises(exception, match=None):
    return unittest.TestCase().assertRaisesRegex(exception, match or ".*")


def cases(names, values):
    """Expand analytic cases with the standard-library unittest loader below."""
    def decorate(function):
        function.case_arguments = [(value,) if "," not in names else tuple(value) for value in values]
        return function
    return decorate


def stamp(uid, start, *, subject="a", source="MIMIC", role="train", record="r1", axis="clock1"):
    return WindowMetadata(subject, source, uid, hashlib.sha256(uid.encode()).hexdigest(),
                          record, axis, start, start + 10, role)


def fixture_memory(**kwargs):
    features = np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    bp = np.array([[100., 60.], [120., 80.], [140., 90.], [200., 100.]])
    metadata = [stamp("b", 0), stamp("a", 10), stamp("c", 20), stamp("d", 0, subject="b", record="r2")]
    return FrozenPersonalMemory(features, bp, metadata, **kwargs)


def query(**kwargs):
    return stamp("q", 40, role="internal_validation", **kwargs)


def test_known_person_only_and_deterministic_ties():
    memory = fixture_memory()
    result = memory.predict([[1., 0.]], [query()], k=2)
    np.testing.assert_allclose(result.bp_interpolation, [[110., 70.]])
    np.testing.assert_array_equal(result.neighbour_indices, [[1, 0]])
    assert result.residual_corrected_bp is None
    np.testing.assert_array_equal(memory.predict([[1., 0.]], [query()], k=1).bp_interpolation, [[120., 80.]])


def test_copies_and_results_are_read_only():
    x, y = np.array([[1., 0.]]), np.array([[100., 60.]])
    memory = FrozenPersonalMemory(x, y, [stamp("m", 0)])
    x[:] = 999
    y[:] = 999
    result = memory.predict([[1., 0.]], [query()], k=1)
    np.testing.assert_array_equal(result.bp_interpolation, [[100., 60.]])
    for array in (memory.features, memory.bp, result.bp_interpolation, result.neighbour_indices):
        with raises(ValueError):
            array.setflags(write=True)


@cases("role", ["internal_validation", "heldout", "test", "meta_test"])
def test_memory_requires_exact_train_role(role):
    with raises(ValueError, match="role"):
        FrozenPersonalMemory([[1., 0.]], [[100., 60.]], [stamp("m", 0, role=role)])


@cases("role", ["train", "heldout", "test", "meta_test"])
def test_development_query_role_is_closed(role):
    with raises(ValueError, match="role"):
        fixture_memory().predict([[1., 0.]], [replace(query(), role=role)], k=1)


@cases("change", [{"window_uid": "d"}, {"waveform_sha256": hashlib.sha256(b"d").hexdigest()},
                                    {"recording_uid": "r2", "start_s": 5., "end_s": 15.}])
def test_overlap_audit_is_global_before_person_retrieval(change):
    with raises(ValueError, match="matches|overlaps"):
        fixture_memory().predict([[1., 0.]], [replace(query(), **change)], k=1)


def test_adjacent_interval_is_permitted_but_future_memory_is_not():
    memory = fixture_memory()
    q = replace(query(), start_s=30., end_s=40.)
    result = memory.predict([[0., 1.]], [q], k=3, mode="chronological_blocked")
    assert result.neighbour_indices.shape == (1, 3)
    q_early = replace(query(record="separate"), start_s=15., end_s=25.)
    result = memory.predict([[1., 0.]], [q_early], k=1, mode="chronological_blocked")
    np.testing.assert_array_equal(result.neighbour_indices, [[0]])
    with raises(ValueError, match="fewer than k"):
        memory.predict([[1., 0.]], [q_early], k=2, mode="chronological_blocked")
    with raises(ValueError, match="timestamp axes"):
        memory.predict([[1., 0.]], [query(record="other", axis="unrelated")], k=1, mode="chronological_blocked")


@cases("change,pattern", [({"subject_uid": "new"}, "unseen"), ({"source": "VitalDB"}, "source"),
                                          ({"recording_uid": ""}, "recording_uid"), ({"time_axis_uid": ""}, "time_axis_uid"),
                                          ({"waveform_sha256": ""}, "SHA-256"), ({"end_s": 40}, "interval")])
def test_required_lineage_and_identity_errors(change, pattern):
    with raises(ValueError, match=pattern):
        fixture_memory().predict([[1., 0.]], [replace(query(), **change)], k=1)


@cases("bad", [[[0., 0.]], [[np.nan, 0.]], [[np.inf, 0.]], [[1., 2., 3.]], [1., 0.]])
def test_invalid_query_features(bad):
    with raises(ValueError):
        fixture_memory().predict(bad, [query()], k=1)


def test_invalid_train_features_bp_and_predictions():
    with raises(ValueError):
        FrozenPersonalMemory([[0., 0.]], [[100., 60.]], [stamp("m", 0)])
    for bp in ([[100.]], [[100., 60., 5.]], [[np.nan, 60.]]):
        with raises(ValueError):
            FrozenPersonalMemory([[1., 0.]], bp, [stamp("m", 0)])
    with raises(ValueError, match="provenance"):
        fixture_memory(train_predictions=np.zeros((4, 2)))
    with raises(ValueError):
        fixture_memory(train_predictions=np.zeros((4, 1)), prediction_provenance="crossfit")
    with raises(ValueError):
        fixture_memory(prediction_provenance="crossfit")


@cases("provenance", ["in_sample", "crossfit"])
def test_fixed_residual_correction_explicit_provenance(provenance):
    prediction = np.array([[90., 65.], [110., 85.], [130., 95.], [190., 105.]])
    memory = fixture_memory(train_predictions=prediction, prediction_provenance=provenance)
    result = memory.predict([[1., 0.]], [query()], k=2, base_predictions=[[115., 75.]])
    np.testing.assert_allclose(result.residual_corrected_bp, [[125., 70.]])
    assert result.residual_provenance == provenance
    assert "labels" not in inspect.signature(memory.predict).parameters
    with raises(TypeError):
        memory.predict([[1., 0.]], [query()], k=2, query_bp=[[200., 100.]])


def test_external_scoring_targets_cannot_change_prediction():
    memory = fixture_memory()
    scoring_bp = np.array([[200., 100.]])
    before = memory.predict([[1., 0.]], [query()], k=2).bp_interpolation.copy()
    scoring_bp[:] = -999
    after = memory.predict([[1., 0.]], [query()], k=2).bp_interpolation
    np.testing.assert_array_equal(before, after)


@cases("k", [0, -1, True, 1.5, 5])
def test_invalid_or_unavailable_k(k):
    with raises(ValueError):
        fixture_memory().predict([[1., 0.]], [query()], k=k)


def test_conflicting_source_and_recording_clock_fail_closed():
    for rows in ([stamp("a", 0), stamp("b", 10, source="VitalDB")],
                 [stamp("a", 0), stamp("b", 10, axis="another")]):
        with raises(ValueError):
            FrozenPersonalMemory([[1., 0.], [0., 1.]], [[100., 60.], [120., 80.]], rows)


def test_duplicate_train_content_rejected_even_across_people():
    first = stamp("a", 0)
    second = replace(stamp("b", 10, subject="b"), waveform_sha256=first.waveform_sha256)
    with raises(ValueError, match="duplicate"):
        FrozenPersonalMemory([[1., 0.], [0., 1.]], [[100., 60.], [120., 80.]], [first, second])


class TestPersonalMemoryProbe(unittest.TestCase):
    """Expanded analytic cases, discoverable by unittest and pytest alike."""


for _name, _function in list(globals().items()):
    if not _name.startswith("test_") or not callable(_function):
        continue
    _function.__test__ = False  # Pytest should collect the expanded class only.
    for _number, _arguments in enumerate(getattr(_function, "case_arguments", [()])):
        def _run(self, function=_function, arguments=_arguments):
            function(*arguments)
        setattr(TestPersonalMemoryProbe, f"{_name}_{_number}", _run)
del _name, _function, _number, _arguments, _run


if __name__ == "__main__":
    unittest.main()
