"""Recompute reported results from saved predictions and cached spike counts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits
from .model import Config, make_weights, activity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/main')
    args = parser.parse_args()
    root = Path(args.output)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest() == manifest['model_code_sha256']
    assert json.loads((root / 'execution.json').read_text())['exit_code'] == 0
    assert json.loads((root / 'status.json').read_text())['completed_conditions'] == 15
    splits = np.load(root / 'split_indices.npz')
    assert len(splits['train']) == 2000 and len(splits['val']) == 500 and len(splits['test']) == 2000
    assert not np.intersect1d(splits['train'], splits['val']).size
    for split in splits.files:
        assert len(np.unique(splits[split])) == len(splits[split])
    names = ['baseline', 'homeostasis_60Hz', 'uniform_control', 'tau_5ms', 'tau_50ms']
    rows = []
    label_reference = None
    for seed in [42, 43, 44]:
        weights = np.load(root / f'weights_seed_{seed}.npy')
        np.testing.assert_array_equal(weights, make_weights(Config(**manifest['model']), seed))
        input_reference = None
        for name in names:
            folder = root / f'seed_{seed}' / name
            metrics = json.loads((folder / 'metrics.json').read_text())
            pred = np.load(folder / 'predictions.npz')
            y, yhat = pred['labels'], pred['predictions']
            if label_reference is None:
                label_reference = y.copy()
                assert np.all(np.bincount(y, minlength=10) == 200)
            np.testing.assert_array_equal(y, label_reference)
            np.testing.assert_allclose(accuracy_score(y, yhat), metrics['test_accuracy'])
            np.testing.assert_allclose(f1_score(y, yhat, average='macro'), metrics['test_macro_f1'])
            np.testing.assert_array_equal(confusion_matrix(y, yhat), metrics['confusion_matrix'])
            feature_files = sorted((folder / 'features').glob('test_*.npz'))
            assert len(feature_files) == 8
            counts, input_counts = [], []
            total_events = 0
            for file in feature_files:
                with np.load(file) as d:
                    counts.append(d['counts'])
                    input_counts.append(d['input_counts'])
                    total_events += int(d['synaptic_events'])
            counts = np.concatenate(counts)
            assert counts.shape == (2000, 256)
            input_counts = np.stack(input_counts)
            if input_reference is None:
                input_reference = input_counts.copy()
            np.testing.assert_array_equal(input_counts, input_reference)
            np.testing.assert_allclose(total_events / 2000, metrics['feature_stats']['test']['synaptic_events_per_image'])
            cfg = Config(**metrics['config'])
            measured = activity(counts, cfg)
            for key, value in measured.items():
                np.testing.assert_allclose(value, metrics['test_activity'][key])
            np.testing.assert_allclose(counts.mean(axis=0) / .1, pred['neuron_rates_hz'])
            model = joblib.load(folder / 'readout.joblib')
            with threadpool_limits(limits=1):
                np.testing.assert_array_equal(model.predict(counts), yhat)
            train = np.concatenate([np.load(p)['counts'] for p in sorted((folder / 'features').glob('train_*.npz'))])
            assert train.shape == (2000, 256)
            np.testing.assert_allclose(model[0].mean_, train.mean(axis=0))
            assert model[-1].n_iter_.max() < model[-1].max_iter
            row = {'seed': seed, 'condition': name, 'accuracy': metrics['test_accuracy'],
                   'macro_f1': metrics['test_macro_f1'], 'C': metrics['selected_C'],
                   **measured, 'zero_spike_image_fraction': float(np.mean(counts.sum(axis=1) == 0)),
                   'input_spikes_per_image': float(input_counts.sum() / 2000),
                   'synaptic_events_per_image': total_events / 2000,
                   'connection_density': float(np.count_nonzero(weights) / weights.size)}
            rows.append(row)
    aggregate = {}
    keys = ['accuracy', 'macro_f1', 'mean_rate_hz', 'spikes_per_image', 'silent_fraction',
            'between_neuron_rate_sd_hz', 'zero_spike_image_fraction',
            'input_spikes_per_image', 'synaptic_events_per_image', 'connection_density']
    for name in names:
        group = [r for r in rows if r['condition'] == name]
        aggregate[name] = {k: {'mean': float(np.mean([r[k] for r in group])),
                               'sd': float(np.std([r[k] for r in group], ddof=1))} for k in keys}
    reported = json.loads((root / 'aggregate.json').read_text())
    for name in names:
        for key, reported_key in [('accuracy', 'test_accuracy'), ('macro_f1', 'test_macro_f1')]:
            for measure in ['mean', 'sd']:
                np.testing.assert_allclose(aggregate[name][key][measure], reported[name][reported_key + '_' + measure])
        np.testing.assert_allclose(aggregate[name]['spikes_per_image']['mean'], reported[name]['spikes_per_image_mean'])
    differences = {}
    for name in names[1:]:
        differences[name] = [100 * (next(r['accuracy'] for r in rows if r['condition'] == name and r['seed'] == seed)
                                    - next(r['accuracy'] for r in rows if r['condition'] == 'baseline' and r['seed'] == seed))
                             for seed in [42, 43, 44]]
    audit = {'passed': True, 'conditions_verified': 15, 'test_predictions_verified': 30000,
             'checks': ['source hash', 'disjoint unique train/validation indices', 'balanced fixed test labels',
                        'reproduced initial weights', 'recomputed accuracy/F1/confusions',
                        'identical per-pixel input counts across conditions in every chunk',
                        'recomputed spike activity and synaptic-event counts', 'classifier predictions',
                        'scaler fitted on training counts only', 'classifier converged',
                        'published aggregate means and standard deviations'],
             'rows': rows, 'aggregate': aggregate, 'paired_accuracy_change_pp': differences}
    (root / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'passed': True, 'conditions_verified': 15,
                      'aggregate': aggregate, 'paired_accuracy_change_pp': differences}, indent=2))


if __name__ == '__main__':
    main()
