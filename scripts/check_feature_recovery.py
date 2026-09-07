"""Real train-role GPU smoke for the serial-loader retry of failed job1486."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pulsedb_fewshot.calbased_train import load_screen_metadata, _fit_scaler, _autocast
from pulsedb_fewshot.calbased_screen import fit_subject_train_means
from pulsedb_fewshot.same_subject_component_train import SameSubjectComponentDataset, fit_quality_proxy, _model_call
from pulsedb_fewshot.personal_feature_models import FEATURE_MODELS, PersonalFeatureRegressor
from pulsedb_fewshot.training import save_json, source_tree_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires allocated CUDA')
    train, _ = load_screen_metadata(args.store_root, 'random_disjoint')
    scaler = _fit_scaler(train)
    means = fit_subject_train_means(train)
    ids = {s: i for i, s in enumerate(sorted(train.subject_uid.astype(str).unique()))}
    train, _ = fit_quality_proxy(train)
    subset = train.groupby('subject_uid', sort=True).head(1).iloc[:64]
    dataset = SameSubjectComponentDataset(subset, args.store_root, scaler, means, ids)
    loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)
    first = next(iter(loader))
    # Serial collation must retain the original deterministic sample tensors.
    for i in range(16):
        item = dataset[i]
        for key, value in item.items():
            actual = first[key][i]
            if isinstance(value, torch.Tensor):
                assert torch.equal(actual, value), key
            else:
                assert actual == value, key
    device = torch.device('cuda')
    torch.manual_seed(20260906)
    model = PersonalFeatureRegressor(FEATURE_MODELS['shared_bilinear64'], subject_count=len(ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = 0
    for batch in loader:
        model.train()
        optimizer.zero_grad()
        with _autocast(device):
            output = _model_call(model, batch, device)
            loss = torch.nn.functional.huber_loss(output.float(), batch['target'].to(device).float(), delta=.5)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
        optimizer.step()
        steps += 1
    args.output.mkdir(parents=True, exist_ok=False)
    model.eval()
    with torch.no_grad(), _autocast(device):
        pred = _model_call(model, first, device).float()
    torch.save(model.state_dict(), args.output/'smoke_state.pt')
    restored = PersonalFeatureRegressor(FEATURE_MODELS['shared_bilinear64'], subject_count=len(ids)).to(device)
    restored.load_state_dict(torch.load(args.output/'smoke_state.pt', map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad(), _autocast(device):
        assert torch.equal(pred, _model_call(restored, first, device).float())
    # Private diagnostic artifact, never uploaded to the public repository.
    save_json(args.output/'prediction_example.json', {'standardized_prediction': pred[0].tolist()})
    import pulsedb_fewshot.personal_feature_train as original
    save_json(args.output/'smoke.json', {'status': 'pass', 'candidate': 'shared_bilinear64',
        'workers': 0, 'gpu': torch.cuda.get_device_name(0), 'finite_steps': steps,
        'original_samples_equal_serial_batch': True, 'checkpoint_roundtrip_exact': True,
        'source_tree_sha256': source_tree_sha256(Path(original.__file__).resolve().parents[2]),
        'heldout_test_accessed': False, 'smoke_weights_used_for_training': False})
    print('FEATURE_RECOVERY_SMOKE=pass', flush=True)


if __name__ == '__main__':
    main()
