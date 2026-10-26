"""Resumable validation selection followed by locked, paired test comparisons."""
import argparse
from dataclasses import replace
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits
from .data import load_splits, download
from .model import Config, make_weights, simulate, activity, adapt_thresholds


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def features(images, config, weights, thresholds, seed, directory, split, chunk_size=250):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(config.dictionary(), sort_keys=True).encode())
    for obj in [images, weights, thresholds]:
        digest.update(np.asarray(obj).tobytes())
    digest.update(str(seed).encode())
    digest.update(Path(__file__).with_name('model.py').read_bytes())
    key = digest.hexdigest()[:16]
    all_counts, total_input, total_events, seconds = [], 0, 0, 0.
    for start in range(0, len(images), chunk_size):
        file = directory / f'{split}_{key}_{start:05d}.npz'
        if file.exists():
            with np.load(file) as stored:
                result = {k: stored[k] for k in stored.files}
        else:
            result = simulate(images[start:start + chunk_size], config, weights,
                              thresholds, seed=seed + start)
            np.savez_compressed(file, **result)
        all_counts.append(result['counts'])
        total_input += int(result['input_counts'].sum())
        total_events += int(result['synaptic_events'])
        seconds += float(result['wall_seconds'])
        print(f'  {split}: {min(start + chunk_size, len(images))}/{len(images)}', flush=True)
    return np.concatenate(all_counts), {'input_spikes_per_image': total_input / len(images),
                                       'synaptic_events_per_image': total_events / len(images),
                                       'simulation_seconds': seconds}


def fit_readout(train_x, train_y, val_x, val_y):
    scores, best, best_accuracy = [], None, -1.
    with threadpool_limits(limits=1):
        for c in [0.01, 0.1, 1., 10.]:
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=c, max_iter=2000, solver='lbfgs',
                                                     random_state=0))
            model.fit(train_x, train_y)
            prediction = model.predict(val_x)
            accuracy = float(accuracy_score(val_y, prediction))
            scores.append({'C': c, 'validation_accuracy': accuracy})
            if accuracy > best_accuracy:
                best, best_accuracy = model, accuracy
    return best, scores


def condition(directory, data, config, weights, thresholds, seed, name, test=False):
    directory = Path(directory) / f'seed_{seed}' / name
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'configuration.json',
               {'config': config.dictionary(), 'seed': seed, 'condition': name,
                'thresholds_mv': np.asarray(thresholds).tolist()})
    counts, feature_stats = {}, {}
    for split, offset in [('train', 100000), ('val', 200000)]:
        counts[split], feature_stats[split] = features(
            data[split][0], config, weights, thresholds, seed + offset,
            directory / 'features', split)
    classifier, grid = fit_readout(counts['train'], data['train'][1],
                                   counts['val'], data['val'][1])
    metrics = {'seed': seed, 'condition': name, 'config': config.dictionary(),
               'regularization_search': grid,
               'selected_C': classifier[-1].C,
               'validation_accuracy': float(accuracy_score(data['val'][1],
                                                            classifier.predict(counts['val']))),
               'training_activity': activity(counts['train'], config)}
    if test:
        counts['test'], feature_stats['test'] = features(
            data['test'][0], config, weights, thresholds, seed + 300000,
            directory / 'features', 'test')
        prediction = classifier.predict(counts['test'])
        metrics.update(test_accuracy=float(accuracy_score(data['test'][1], prediction)),
                       test_macro_f1=float(f1_score(data['test'][1], prediction, average='macro')),
                       test_activity=activity(counts['test'], config),
                       confusion_matrix=confusion_matrix(data['test'][1], prediction).tolist(),
                       n_train=len(data['train'][1]), n_validation=len(data['val'][1]),
                       n_test=len(data['test'][1]))
        np.savez_compressed(directory / 'predictions.npz', labels=data['test'][1],
                            predictions=prediction,
                            neuron_rates_hz=counts['test'].mean(axis=0) / (config.encoding_ms / 1000))
    metrics['feature_stats'] = feature_stats
    joblib.dump(classifier, directory / 'readout.joblib')
    write_json(directory / 'metrics.json', metrics)
    print(f'{name}, seed {seed}: val={metrics["validation_accuracy"]:.3f}' +
          (f', test={metrics["test_accuracy"]:.3f}' if test else ''), flush=True)
    return metrics


def adaptation(directory, data, config, weights, seed, target, adapt_size, epochs):
    directory = Path(directory) / f'seed_{seed}' / f'homeostasis_{target:g}Hz'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'adaptation.json'
    signature = {'config': config.dictionary(), 'seed': seed, 'target': target,
                 'adapt_size': min(adapt_size, len(data['train'][0])), 'epochs': epochs,
                 'data_hash': hashlib.sha256(data['train'][0].tobytes()).hexdigest(),
                 'code_hash': hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()}
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('signature') == signature:
            return np.array(cached['thresholds_mv'])
    print(f'Adapting thresholds: seed={seed}, target={target:g} Hz', flush=True)
    thresholds, history = adapt_thresholds(data['train'][0][:adapt_size], config,
                                          weights, seed, target_hz=target,
                                          epochs=epochs, batch_size=100)
    write_json(path, {'signature': signature, 'target_hz': target,
                      'thresholds_mv': thresholds.tolist(), 'history': history})
    return thresholds


def matched_uniform_threshold(data, config, weights, adapted, seed, output, sample_size=200):
    # Match overall rate on the same training images and identical Poisson draws.
    if Path(output).exists():
        saved = json.loads(Path(output).read_text())
        return np.full(config.n_hidden, saved['selected']['threshold_mv'])
    images = data['train'][0][:sample_size]
    desired = simulate(images, config, weights, adapted, seed=seed + 90000)['counts'].mean()
    low, high = config.reset_mv + 2., -30.
    observations = []
    for _ in range(7):
        threshold = (low + high) / 2
        counts = simulate(images, config, weights,
                           np.full(config.n_hidden, threshold), seed=seed + 90000)['counts']
        observed = float(counts.mean())
        observations.append({'threshold_mv': threshold, 'mean_count': observed})
        if observed > desired:
            low = threshold
        else:
            high = threshold
    best = min(observations, key=lambda x: abs(x['mean_count'] - desired))
    write_json(output, {'desired_mean_count': float(desired), 'selected': best,
                        'observations': observations, 'n_training_images': len(images)})
    return np.full(config.n_hidden, best['threshold_mv'])


def summarize(output, rows):
    fields = ['seed', 'condition', 'validation_accuracy', 'test_accuracy', 'test_macro_f1',
              'mean_rate_hz', 'spikes_per_image', 'silent_fraction', 'synaptic_events_per_image']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row[key] for key in fields if key in row}
            values.update({key: row['test_activity'][key] for key in fields if key in row['test_activity']})
            values['synaptic_events_per_image'] = row['feature_stats']['test']['synaptic_events_per_image']
            writer.writerow(values)
    aggregate = {}
    for name in sorted({row['condition'] for row in rows}):
        selected = [row for row in rows if row['condition'] == name]
        aggregate[name] = {'seeds': [r['seed'] for r in selected]}
        for key in ['test_accuracy', 'test_macro_f1']:
            values = [r[key] for r in selected]
            aggregate[name][key + '_mean'] = float(np.mean(values))
            aggregate[name][key + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate[name]['spikes_per_image_mean'] = float(np.mean([r['test_activity']['spikes_per_image'] for r in selected]))
    write_json(output / 'aggregate.json', aggregate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/main')
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--validation', type=int, default=500)
    parser.add_argument('--test', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--targets', type=float, nargs='+', default=[20., 40., 60.])
    parser.add_argument('--adapt-size', type=int, default=1000)
    parser.add_argument('--adapt-epochs', type=int, default=2)
    parser.add_argument('--match-images', type=int, default=200)
    parser.add_argument('--tau-sweep', action='store_true')
    parser.add_argument('--weight', type=float, default=0.12)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--backend', choices=['numpy', 'cython'], default='numpy')
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = replace(Config(), weight_mean_mv=args.weight, weight_sd_mv=args.weight,
                     dt_ms=args.dt, backend=args.backend)
    download()
    manifest = {'arguments': vars(args), 'model': config.dictionary(),
                'split_seed': 2026, 'selection_seed': args.seeds[0],
                'data_source': json.loads(Path('data/source.json').read_text()),
                'model_code_sha256': hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing output uses a different experiment configuration. Choose a new output directory.')
    write_json(manifest_path, manifest)
    data = load_splits(args.train, args.validation, args.test, output)
    write_json(output / 'status.json', {'stage': 'validation_selection', 'started_unix': started})
    selection_seed = args.seeds[0]
    weights = make_weights(config, selection_seed)
    candidates = []
    for target in args.targets:
        thresholds = adaptation(output, data, config, weights, selection_seed, target,
                                 args.adapt_size, args.adapt_epochs)
        metrics = condition(output, data, config, weights, thresholds, selection_seed,
                            f'homeostasis_{target:g}Hz', test=False)
        candidates.append({'target_hz': target, 'validation_accuracy': metrics['validation_accuracy']})
    selected_target = max(candidates, key=lambda x: x['validation_accuracy'])['target_hz']
    write_json(output / 'selection.json', {'selected_target_hz': selected_target,
                                          'candidates': candidates,
                                          'rule': 'Maximum validation accuracy on selection seed; ties use first target.'})
    rows = []
    for seed in args.seeds:
        weights = make_weights(config, seed)
        np.save(output / f'weights_seed_{seed}.npy', weights)
        baseline = np.full(config.n_hidden, config.threshold_mv)
        thresholds = adaptation(output, data, config, weights, seed, selected_target,
                                 args.adapt_size, args.adapt_epochs)
        control = matched_uniform_threshold(data, config, weights, thresholds, seed,
                                             output / f'seed_{seed}' / 'uniform_matching.json',
                                             sample_size=args.match_images)
        conditions = [('baseline', config, baseline),
                      (f'homeostasis_{selected_target:g}Hz', config, thresholds),
                      ('uniform_control', config, control)]
        if args.tau_sweep:
            conditions += [(f'tau_{tau:g}ms', replace(config, tau_ms=tau), baseline)
                           for tau in [5., 50.]]
        for name, current_config, current_thresholds in conditions:
            write_json(output / 'status.json', {'stage': 'final_comparisons', 'seed': seed,
                                               'condition': name, 'completed_conditions': len(rows)})
            rows.append(condition(output, data, current_config, weights,
                                  current_thresholds, seed, name, test=True))
            summarize(output, rows)
            if seed == args.seeds[0]:
                diagnostic = simulate(data['train'][0][:3], current_config, weights,
                                       current_thresholds, seed=seed + 400000, diagnostics=True)
                np.savez_compressed(output / f'seed_{seed}' / name / 'diagnostics.npz',
                                     **diagnostic, images=data['train'][0][:3],
                                     labels=data['train'][1][:3])
    write_json(output / 'status.json', {'stage': 'complete', 'wall_seconds': time.time() - started,
                                       'completed_conditions': len(rows)})
    print('Experiment complete.', flush=True)


if __name__ == '__main__':
    main()
