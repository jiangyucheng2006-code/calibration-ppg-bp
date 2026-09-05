import json
import pandas as pd
import pytest
from pulsedb_fewshot.lora_prs_models import PRS_MODELS, PRIMARY, REFERENCE, SPLIT_MODES
from pulsedb_fewshot.lora_prs_train import SOURCE_HASHES
from pulsedb_fewshot.lora_prs_report import build_report


def test_paired_gates_and_initialization(tmp_path):
    paths = []
    for mode in SPLIT_MODES:
        path = tmp_path / mode
        path.mkdir()
        (path/'selection.json').write_text(json.dumps({'split_mode': mode, 'seed': 7, 'heldout_test_accessed': False}))
        rows = []
        for candidate in PRS_MODELS:
            run = path / candidate
            run.mkdir()
            (run/'initialization.json').write_text(json.dumps({'source_checkpoint_sha256': SOURCE_HASHES[mode],
                'metrics': {s: {'mean_mae': 4.} for s in ['Overall', 'MIMIC', 'VitalDB']}}))
            for view in ['Overall', 'MIMIC', 'VitalDB']:
                mae = 3.95 if candidate == REFERENCE else 3.7
                if candidate == PRIMARY and mode == 'chronological_blocked' and view == 'MIMIC':
                    mae = 4.1
                rows.append(dict(candidate=candidate, runner='lora_prs_continuation', view=view,
                    mean_mae=mae, run_dir=str(run), split_mode=mode))
        pd.DataFrame(rows).to_csv(path/'participant_macro_summary.csv', index=False)
        paths.append(path)
    result = build_report(*paths, tmp_path/'out')
    assert PRIMARY not in result['eligible_candidates'] and REFERENCE not in result['eligible_candidates']
    assert 'lora_prs_bias' in result['eligible_candidates']
    broken = paths[0]/PRIMARY/'initialization.json'
    init = json.loads(broken.read_text())
    init['metrics']['Overall']['mean_mae'] = 3.
    broken.write_text(json.dumps(init))
    with pytest.raises(ValueError, match='identical initial'):
        build_report(*paths, tmp_path/'out2')
