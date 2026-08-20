import numpy as np

from pulsedb_fewshot.analyze_beat_similarity import beat_similarity


def test_repeated_pulses_have_high_similarity() -> None:
    t=np.linspace(0,1,125,endpoint=False); pulse=np.exp(-((t-.28)/.09)**2)+.25*np.exp(-((t-.55)/.08)**2); signal=np.tile(pulse,10)
    result=beat_similarity(signal,125)
    assert result["valid"] and result["n_beats"]>=7
    assert result["pairwise_corr_median"]>.98


def test_flat_signal_is_rejected() -> None:
    assert beat_similarity(np.ones(1250),125)["valid"] is False
