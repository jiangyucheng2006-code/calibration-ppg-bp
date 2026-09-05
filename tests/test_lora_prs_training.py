import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from pulsedb_fewshot.lora_prs_models import PRS_MODELS, LoraPRSRegressor
from pulsedb_fewshot.lora_prs_train import initialize, SOURCE_HASHES
from pulsedb_fewshot.training import file_sha256
import pulsedb_fewshot.same_subject_component_train as trainer


def test_initializer_checks_provenance_mapping_and_scaler(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    store = tmp_path / 'store'
    store.mkdir()
    (store / 'materialization.json').write_text('{}')
    m = LoraPRSRegressor(PRS_MODELS['lora_prs_dynamic'], subject_count=2)
    scaler = {'mean': [110., 70.], 'std': [10., 8.]}
    mapping = {'a': 0, 'b': 1}
    torch.save({'model_state': {'reference.' + k: v for k, v in m.base.state_dict().items()},
                'target_scaler': scaler, 'subject_to_index': mapping}, source / 'best.pt')
    digest = file_sha256(source / 'best.pt')
    monkeypatch.setitem(SOURCE_HASHES, 'random_disjoint', digest)
    run = dict(status='complete', candidate='subject_lora_rank4', screen_id='same-subject-personal-profile-v1',
               split_mode='random_disjoint', heldout_test_accessed=False, selection_role='internal_validation',
               source_parent_split='meta_train', read_roles=['train', 'internal_validation'],
               checkpoint_sha256=digest, store_manifest_sha256=file_sha256(store / 'materialization.json'),
               best_epoch=12, seed=7, gpu='fixture')
    (source / 'run.json').write_text(json.dumps(run))
    args = SimpleNamespace(source_run=source, store_root=store, split_mode='random_disjoint',
                           learning_rate=3e-4, base_learning_rate=1e-5)
    assert initialize(m, args, scaler, mapping)['source_checkpoint_sha256'] == digest
    with pytest.raises(ValueError, match='mapping'):
        initialize(m, args, scaler, {'a': 1, 'b': 0})
    with pytest.raises(ValueError, match='scaler'):
        initialize(m, args, {'mean': [111., 70.], 'std': [10., 8.]}, mapping)
    run['heldout_test_accessed'] = True
    (source / 'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='ineligible'):
        initialize(m, args, scaler, mapping)


def test_epoch_zero_is_kept_if_all_continuation_epochs_are_worse(tmp_path, monkeypatch):
    frame = pd.DataFrame({'subject_uid': ['a', 'a', 'b', 'b'], 'source': ['MIMIC']*2 + ['VitalDB']*2,
        'segment_uid': ['a1', 'a2', 'b1', 'b2'], 'sbp': [110., 115., 130., 135.],
        'dbp': [65., 70., 75., 80.], 'ppg_f_std': [1.]*4,
        'waveform_file': ['waves.npy']*4, 'waveform_row': list(range(4))})
    np.save(tmp_path / 'waves.npy', np.random.default_rng(9).normal(size=(4, 1250)).astype('float32'))
    (tmp_path / 'materialization.json').write_text('{}')
    monkeypatch.setattr(trainer, 'load_screen_metadata', lambda *_: (frame.copy(), frame.copy()))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    calls = []
    def predictions(*_):
        error = 1. if not calls else 2.
        calls.append(error)
        return pd.DataFrame({'subject_uid': frame.subject_uid, 'source': frame.source, 'event_id': frame.segment_uid,
            'target_sbp': frame.sbp, 'target_dbp': frame.dbp, 'pred_sbp': frame.sbp + error, 'pred_dbp': frame.dbp + error})
    monkeypatch.setattr(trainer, 'predict', predictions)
    class Tiny(torch.nn.Module):
        def __init__(self, *_, **__):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, ppg, anchor, **_):
            return anchor + self.bias
    args = SimpleNamespace(candidate='lora_continue', split_mode='random_disjoint', epochs=0, patience=8,
        seed=7, store_root=tmp_path, output=tmp_path/'out', beat_similarity_path=None, require_cuda=False,
        examples_per_epoch=4, workers=0, batch_size=2, learning_rate=.01, weight_decay=0., huber_delta=.5)
    result = trainer.train_component(args, components=PRS_MODELS, model_factory=Tiny,
        initializer=lambda *_: {'fixture': True})
    assert result['epochs_completed'] == 8 and result['best_epoch'] == 0
    checkpoint = torch.load(args.output/'best.pt', weights_only=False)
    assert checkpoint['epoch'] == 0 and checkpoint['model_state']['bias'].item() == 0.
    initial = pd.read_parquet(args.output/'initial_internal_validation_predictions.parquet')
    selected = pd.read_parquet(args.output/'best_internal_validation_predictions.parquet')
    pd.testing.assert_frame_equal(initial, selected)
