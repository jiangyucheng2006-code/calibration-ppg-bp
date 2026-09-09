"""Read-only verification of completed enrollment outputs; emit aggregates only.

Run with the project Python on the server. Does not fit models, re-open source
BP labels, alter artifacts, export subject IDs, or discard any test observations.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/jiangyu.cheng/work/ppg_bp/outputs/post-enrollment-30-v1_20260909-102000')
ARCHIVE = Path('/home/jiangyu.cheng/nas/ppg_bp/outputs') / ROOT.name
METHODS = ('personal_mean', 'shared_with_anchor', 'new_person_lora', 'new_person_lora_memory')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), minimum=float(values.min()), median=float(np.median(values)),
                maximum=float(values.max()), mean=float(values.mean()))


plan = read_json(ROOT / 'prepare/plan.json')
receipt = read_json(ROOT / 'evaluate/evaluation_receipt.json')
inner = read_json(ROOT / 'population_inner/run.json')
final = read_json(ROOT / 'population_final/run.json')
assert receipt['status'] == inner['status'] == final['status'] == 'complete'
assert receipt['plan_sha256'] == digest(ROOT / 'prepare/plan.json')
assert receipt['population_checkpoint_sha256'] == final['checkpoint_sha256']
assert inner['old_checkpoint_used'] is False and final['old_checkpoint_used'] is False
assert inner['enrollment_labels_accessed'] is False and final['enrollment_labels_accessed'] is False
assert receipt['all_four_methods_frozen_before_targets'] is True
assert receipt['test_based_selection'] is False

profiles = []
for i in range(2):
    path = ROOT / f'personal_{i}/run.json'
    run = read_json(path)
    assert digest(path) == receipt['personal_runs'][i]['run_sha256']
    assert run['status'] == 'complete' and run['test_targets_accessed'] is False
    assert run['shared_frozen'] is True and run['other_person_adapter_copied'] is False
    assert run['population_checkpoint_sha256'] == final['checkpoint_sha256']
    assert run['subjects'] == plan['selected_subjects'][i::2]
    profiles.extend(run['profiles'].values())
assert len(profiles) == 30
assert all(p['registration_rows'] == 360 and p['test_rows'] == 40 for p in profiles)
assert all(p['shared_frozen_verified'] and p['reload_equivalence'] for p in profiles)

frames, per_person = {}, {}
keys = ['subject_uid', 'event_id', 'source']
saved_macro = pd.read_csv(ROOT / 'evaluate/participant_macro.csv')
for method in METHODS:
    name = f'{method}_frozen_predictions.parquet'
    assert digest(ROOT / 'evaluate' / name) == receipt['prediction_files'][name]
    frame = pd.read_parquet(ROOT / 'evaluate' / f'{method}_scored_predictions.parquet')
    frame = frame.sort_values(keys).reset_index(drop=True)
    assert len(frame) == 1200 and frame.subject_uid.nunique() == 30
    assert frame.groupby('subject_uid').size().eq(40).all()
    assert set(frame.subject_uid) == set(plan['selected_subjects'])
    assert not frame[keys].duplicated().any()
    for bp in ('sbp', 'dbp'):
        assert np.isfinite(frame[[f'pred_{bp}', f'target_{bp}']].to_numpy()).all()
        frame[f'{bp}_mae'] = (frame[f'pred_{bp}'] - frame[f'target_{bp}']).abs()
    frame['mean_mae'] = (frame.sbp_mae + frame.dbp_mae) / 2
    person = frame.groupby(['subject_uid', 'source'])[['sbp_mae', 'dbp_mae', 'mean_mae']].mean()
    for scope in ('Overall', 'MIMIC', 'VitalDB'):
        subset = person if scope == 'Overall' else person.xs(scope, level='source')
        stored = saved_macro.loc[saved_macro.Setting.eq(method) & saved_macro.Scope.eq(scope)].iloc[0]
        assert len(subset) == int(stored.n_participants)
        for col in ('sbp_mae', 'dbp_mae', 'mean_mae'):
            assert np.isclose(subset[col].mean(), stored[col], atol=1e-12, rtol=0)
    frames[method], per_person[method] = frame, person

base = frames[METHODS[0]]
for frame in frames.values():
    assert frame[keys + ['target_sbp', 'target_dbp']].equals(base[keys + ['target_sbp', 'target_dbp']])

comparisons = []
for old, new in (('shared_with_anchor', 'new_person_lora'), ('new_person_lora', 'new_person_lora_memory')):
    delta = per_person[old] - per_person[new]
    for scope in ('Overall', 'MIMIC', 'VitalDB'):
        d = delta if scope == 'Overall' else delta.xs(scope, level='source')
        for metric in ('sbp_mae', 'dbp_mae', 'mean_mae'):
            values = d[metric].to_numpy()
            comparisons.append(dict(reference=old, candidate=new, scope=scope, metric=metric,
                improvement_mmHg=describe(values), improved_people=int((values > 1e-10).sum()),
                worsened_people=int((values < -1e-10).sum()), tied_people=int((np.abs(values) <= 1e-10).sum())))

archive_files = {}
for stage in ('evaluate', 'population_benchmark'):
    paths = list((ROOT / stage).glob('*.csv'))
    paths += [ROOT / stage / name for name in ('evaluation_receipt.json', 'frozen_predictions.json', 'RESULT_TABLES.md')]
    for path in paths:
        relative = path.relative_to(ROOT)
        assert digest(path) == digest(ARCHIVE / relative), f'archive mismatch: {relative}'
        archive_files[str(relative)] = digest(path)

report = dict(status='pass', protocol_id=receipt['protocol_id'], analysis='descriptive paired comparison; no hypothesis tests or confidence intervals',
    source_audit=plan['audit'], population_inner_epochs=inner['epochs_completed'], population_selected_epoch=inner['best_epoch'],
    population_final_epochs=final['epochs_completed'], population_fit_subjects=final['fit_subjects'],
    subjects=30, windows=1200, source_subjects=base.groupby('source').subject_uid.nunique().to_dict(),
    selected_personal_epochs=describe([p['selected_epoch'] for p in profiles]),
    zero_epoch_profiles=sum(p['selected_epoch'] == 0 for p in profiles),
    refit_optimizer_steps=describe([p['refit_optimizer_steps'] for p in profiles]),
    adapter_parameter_counts=sorted(set(p['adapter_parameters'] for p in profiles)),
    shared_state_unchanged_profiles=sum(p['shared_frozen_verified'] for p in profiles),
    reload_equivalent_profiles=sum(p['reload_equivalence'] for p in profiles),
    registration_rows_per_person=360, test_rows_per_person=40,
    all_query_keys_and_targets_matched=True, all_primary_metrics_recomputed=True,
    paired_comparisons=comparisons, archived_aggregate_sha256=archive_files)
print(json.dumps(report, ensure_ascii=False, indent=2))
