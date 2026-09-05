"""Small real train-role GPU check before continuation jobs can start."""
import argparse
import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .lora_prs_models import PRS_MODELS, LoraPRSRegressor
from .lora_prs_train import initialize
from .calbased_train import load_screen_metadata, _fit_scaler, _autocast
from .calbased_screen import fit_subject_train_means
from .same_subject_component_train import SameSubjectComponentDataset, fit_quality_proxy, _model_call
from .training import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store-root', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--split-mode', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.learning_rate, args.base_learning_rate, args.weight_decay = 3e-4, 1e-5, 1e-4
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    train, _ = load_screen_metadata(args.store_root, args.split_mode)
    scaler = _fit_scaler(train)
    means = fit_subject_train_means(train)
    indices = {s: i for i, s in enumerate(sorted(train.subject_uid.astype(str).unique()))}
    train, _ = fit_quality_proxy(train)
    # A few rows from different registered people; no held-out role read.
    subset = train.groupby('subject_uid', sort=True).head(1).iloc[:8]
    ds = SameSubjectComponentDataset(subset, args.store_root, scaler, means, indices)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    records = []
    reference = None
    for name, spec in PRS_MODELS.items():
        torch.manual_seed(20260907)
        m = LoraPRSRegressor(spec, subject_count=len(indices)).cuda()
        info = initialize(m, args, scaler, indices)
        m.eval()
        with torch.no_grad(), _autocast(torch.device('cuda')):
            before = _model_call(m, batch, torch.device('cuda'))
            expected = _model_call(m.base, batch, torch.device('cuda'))
        if not torch.equal(before, expected):
            raise ValueError('nonzero initial correction')
        if reference is None:
            reference = before.clone()
        elif not torch.equal(before, reference):
            raise ValueError('unequal initial candidate predictions')
        frozen = copy.deepcopy(m.base.state_dict()) if spec.freeze_base else None
        optimizer = m.optimizer(args)
        for _ in range(2):
            m.train()
            optimizer.zero_grad()
            with _autocast(torch.device('cuda')):
                output = _model_call(m, batch, torch.device('cuda'))
                loss = torch.nn.functional.huber_loss(output.float(), batch['target'].cuda().float(), delta=.5) + m.penalty()
            loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None):
                raise ValueError('nonfinite train step')
            optimizer.step()
        if frozen is not None and not all(torch.equal(v, m.base.state_dict()[k]) for k, v in frozen.items()):
            raise ValueError('frozen model changed')
        records.append({'candidate': name, 'status': 'pass', 'source_sha256': info['source_checkpoint_sha256'],
                        'initial_max_difference': 0, 'finite_steps': 2})
        del m, optimizer, frozen
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'smoke.json', {'status': 'pass', 'gpu': torch.cuda.get_device_name(0),
        'split_mode': args.split_mode, 'heldout_test_accessed': False, 'records': records})


if __name__ == '__main__':
    main()
