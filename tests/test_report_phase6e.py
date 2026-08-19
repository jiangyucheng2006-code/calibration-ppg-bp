import pytest

from pulsedb_fewshot.report_phase6e import _parse


def test_parse_phase6e_runs_rejects_bad_or_duplicate_specs() -> None:
    assert str(_parse(["Ridge=/tmp/run"])["Ridge"])=="/tmp/run"
    with pytest.raises(ValueError): _parse(["bad"])
    with pytest.raises(ValueError): _parse(["A=/one","A=/two"])
