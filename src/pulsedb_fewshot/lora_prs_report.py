"""Paired continuation report, with both continued and immutable-source comparators."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .lora_prs_models import PRS_MODELS, REFERENCE, PRIMARY, SCREEN_ID, SPLIT_MODES
from .lora_prs_train import SOURCE_HASHES
from .training import save_json


def build_report(random_report, chronological_report, output):
    frames, seeds = [], []
    for mode, path in zip(SPLIT_MODES, [random_report, chronological_report]):
        selection = json.loads((path / 'selection.json').read_text())
        if selection.get('heldout_test_accessed') is not False or selection['split_mode'] != mode:
            raise ValueError('invalid development report')
        seeds.append(selection['seed'])
        f = pd.read_csv(path / 'participant_macro_summary.csv')
        if (set(f.candidate) != set(PRS_MODELS) or len(f) != 15
                or f.duplicated(['candidate', 'view']).any()
                or not f.runner.eq('lora_prs_continuation').all()):
            raise ValueError('incomplete or mixed continuation screen')
        initial_by_candidate = {}
        for candidate in PRS_MODELS:
            run_dir = Path(f.loc[f.candidate.eq(candidate), 'run_dir'].iloc[0])
            init = json.loads((run_dir / 'initialization.json').read_text())
            if init['source_checkpoint_sha256'] != SOURCE_HASHES[mode]:
                raise ValueError('wrong pretrained source')
            initial_by_candidate[candidate] = init['metrics']
        baseline = initial_by_candidate[REFERENCE]
        for values in initial_by_candidate.values():
            for scope in ['Overall', 'MIMIC', 'VitalDB']:
                if not np.isclose(values[scope]['mean_mae'], baseline[scope]['mean_mae'], atol=1e-6, rtol=0):
                    raise ValueError('candidates did not start from identical initial predictions')
        ref = f.loc[f.candidate.eq(REFERENCE), ['view', 'mean_mae']].rename(columns={'mean_mae': 'continued_reference_mean_mae'})
        f = f.merge(ref, on='view', validate='many_to_one')
        f['initial_reference_mean_mae'] = f.view.map(lambda s: baseline[s]['mean_mae'])
        f['gain_vs_continued'] = f.continued_reference_mean_mae - f.mean_mae
        f['gain_vs_initial'] = f.initial_reference_mean_mae - f.mean_mae
        frames.append(f)
    if seeds[0] != seeds[1]:
        raise ValueError('different screening seeds')
    combined = pd.concat(frames, ignore_index=True)
    gates = []
    for candidate, f in combined.groupby('candidate', sort=False):
        overall = f.loc[f.view.eq('Overall')]
        sources = f.loc[~f.view.eq('Overall')]
        passed = len(overall) == 2 and len(sources) == 4
        for gain in ['gain_vs_continued', 'gain_vs_initial']:
            passed = passed and overall[gain].ge(.15).all() and sources[gain].gt(0).all()
        gates.append({'candidate': candidate, 'passes_accuracy_gate': bool(passed),
                      'mean_across_modes': float(overall.mean_mae.mean())})
    gate = pd.DataFrame(gates)
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / 'cross_split_comparison.csv', index=False)
    gate.to_csv(output / 'promotion_gate.csv', index=False)
    eligible = gate.loc[gate.passes_accuracy_gate].sort_values('mean_across_modes')
    result = {'status': 'complete', 'screen_id': SCREEN_ID, 'heldout_test_accessed': False,
              'primary_candidate': PRIMARY, 'reference': REFERENCE, 'seed': seeds[0],
              'eligible_candidates': eligible.candidate.tolist(),
              'recommendation': 'retain LoRA' if eligible.empty else 'confirm eligible candidates; not final promotion',
              'official_pulsedb_calbased_reproduction': False}
    save_json(output / 'selection.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--random-report', type=Path, required=True)
    p.add_argument('--chronological-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build_report(args.random_report, args.chronological_report, args.output), indent=2))
