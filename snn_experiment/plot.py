"""Generate publication-style static figures from measured experiment outputs."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#2563a6', '#e28329', '#3e876c', '#8764a5', '#bb5353']


def label(name):
    if name == 'baseline':
        return 'Baseline (20 ms)'
    if name == 'uniform_control':
        return 'Uniform threshold'
    if name.startswith('homeostasis'):
        return 'Homeostasis (' + name.split('_')[1] + ')'
    return name.replace('tau_', 'Tau = ').replace('ms', ' ms')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/main')
    args = parser.parse_args()
    root = Path(args.output)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    aggregate = json.loads((root / 'aggregate.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    names = [n for n in ['baseline', f'homeostasis_{selection["selected_target_hz"]:g}Hz',
                         'uniform_control', 'tau_5ms', 'tau_50ms'] if n in aggregate]
    seed = manifest['selection_seed']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 120,
                         'savefig.dpi': 180, 'axes.titleweight': 'bold'})

    def save(fig, name):
        fig.savefig(out / (name + '.png'), bbox_inches='tight')
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    for index, name in enumerate(names):
        row = aggregate[name]
        for ax, key, title in zip(axes, ['test_accuracy', 'test_macro_f1'], ['Test accuracy (%)', 'Test macro-F1 (%)']):
            mean, sd = row[key + '_mean'] * 100, (row[key + '_sd'] or 0) * 100
            ax.bar(index, mean, yerr=sd if row[key + '_sd'] is not None else None,
                   color=COLORS[index], capsize=4, width=.7)
            ax.text(index, mean + sd + 1.5, f'{mean:.1f}', ha='center', fontsize=9)
            ax.set_ylabel(title)
            ax.set_ylim(0, 100)
            ax.set_xticks(range(len(names)), [label(n) for n in names], rotation=25, ha='right')
    n_seeds = min(len(aggregate[n]['seeds']) for n in names)
    fig.suptitle('Measured performance; error bars show standard deviation across seeds'
                 if n_seeds > 1 else 'Pipeline check: one seed; no across-seed uncertainty estimate')
    save(fig, 'performance')

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    for index, name in enumerate(names):
        row = aggregate[name]
        ax.errorbar(row['spikes_per_image_mean'], row['test_accuracy_mean'] * 100,
                     yerr=(row['test_accuracy_sd'] or 0) * 100,
                     fmt='o', markersize=8, capsize=3, color=COLORS[index], label=label(name))
    ax.set(xlabel='Hidden spikes per image (activity proxy)', ylabel='Test accuracy (%)',
            title='Accuracy and hidden-layer activity')
    ax.legend(fontsize=9, loc='best')
    save(fig, 'accuracy_spike_cost')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    for index, name in enumerate(names[:3]):
        path = root / f'seed_{seed}' / name / 'predictions.npz'
        if not path.exists():
            continue
        with np.load(path) as d:
            rates = d['neuron_rates_hz']
        axes[0].hist(rates, bins=15, alpha=.55, label=label(name), color=COLORS[index])
        axes[1].plot(np.sort(rates), label=label(name), color=COLORS[index])
    axes[0].set(xlabel='Mean firing rate per neuron (Hz)', ylabel='Number of neurons', title=f'Rate distribution, seed {seed}')
    axes[1].set(xlabel='Neuron rank (sorted separately)', ylabel='Mean firing rate (Hz)', title='Spread of neuron activity')
    axes[0].legend(fontsize=8)
    save(fig, 'firing_distributions')

    home = f'homeostasis_{selection["selected_target_hz"]:g}Hz'
    adaptation = json.loads((root / f'seed_{seed}' / home / 'adaptation.json').read_text())
    history = adaptation['history']
    thresholds = np.array([h['threshold_mv'] for h in history])
    rates = np.array([h['rate_hz'] for h in history])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    steps = np.arange(len(history)) + 1
    initial = np.full((1, thresholds.shape[1]), manifest['model']['threshold_mv'])
    threshold_plot = np.concatenate([initial, thresholds])
    axes[0].plot(np.arange(len(history) + 1), threshold_plot[:, :12], alpha=.65, lw=1)
    axes[0].plot(np.arange(len(history) + 1), threshold_plot.mean(axis=1), color='black',
                 label='All-neuron mean', lw=2, marker='o', markersize=3)
    axes[0].set(xlabel='Adaptation batch', ylabel='Threshold (mV)', title='Threshold adaptation (12 example neurons)')
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, rates.mean(axis=1), color=COLORS[1], label='Batch mean', marker='o', markersize=3)
    axes[1].fill_between(steps, rates.mean(axis=1) - rates.std(axis=1), rates.mean(axis=1) + rates.std(axis=1),
                         color=COLORS[1], alpha=.2, label='Across-neuron SD')
    axes[1].axhline(selection['selected_target_hz'], ls='--', color='black', label='Target')
    axes[1].set(xlabel='Adaptation batch', ylabel='Firing rate (Hz)', title='Training-only activity trajectory')
    axes[1].legend(fontsize=8)
    save(fig, 'homeostasis_adaptation')

    diagnostics = [n for n in names if (root / f'seed_{seed}' / n / 'diagnostics.npz').exists()]
    fig, axes = plt.subplots(len(diagnostics), 2, figsize=(10.5, 2.2 * len(diagnostics)),
                              layout='constrained', squeeze=False)
    for row, name in enumerate(diagnostics):
        with np.load(root / f'seed_{seed}' / name / 'diagnostics.npz') as d:
            t, neurons = d['spike_times_ms'], d['spike_neurons']
            mask = (t < 100) & (neurons < 40)
            axes[row, 0].scatter(t[mask], neurons[mask], s=2, color=COLORS[row % 5])
            mask_v = d['voltage_times_ms'] < 100
            for neuron in range(3):
                axes[row, 1].plot(d['voltage_times_ms'][mask_v], d['voltage_mv'][neuron, mask_v], lw=.8, alpha=.8)
            axes[row, 0].set(title=label(name) + ' | first 40 neurons', ylabel='Neuron index',
                              xlim=(0, 100), ylim=(-1, 40))
            if not np.any(mask):
                axes[row, 0].text(50, 20, 'No spikes in these neurons', ha='center', color='#666666')
            axes[row, 1].set(title='Voltage, first 3 neurons', ylabel='mV', xlim=(0, 100))
            for ax in axes[row]:
                ax.set_xlabel('Time within image (ms)')
    save(fig, 'rasters_and_voltage')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, name in zip(axes, ['baseline', home]):
        metrics = json.loads((root / f'seed_{seed}' / name / 'metrics.json').read_text())
        matrix = np.asarray(metrics['confusion_matrix'])
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit', ylabel='True digit', title=label(name))
    fig.colorbar(im, ax=axes, label='Fraction of true class', shrink=.85)
    save(fig, 'confusion_matrices')

    if all(n in aggregate for n in ['tau_5ms', 'baseline', 'tau_50ms']):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout='constrained')
        tau_names = ['tau_5ms', 'baseline', 'tau_50ms']
        axes[0].errorbar([5, 20, 50], [aggregate[n]['test_accuracy_mean'] * 100 for n in tau_names],
                          yerr=[(aggregate[n]['test_accuracy_sd'] or 0) * 100 for n in tau_names],
                          marker='o', capsize=4)
        axes[1].plot([5, 20, 50], [aggregate[n]['spikes_per_image_mean'] for n in tau_names], 'o-', color=COLORS[1])
        axes[0].set(ylabel='Test accuracy (%)', title='Accuracy versus membrane time constant')
        axes[1].set(ylabel='Hidden spikes per image', title='Activity versus membrane time constant')
        for ax in axes:
            ax.set_xlabel('Membrane time constant (ms)')
            ax.set_xticks([5, 20, 50])
        save(fig, 'tau_sweep')
    print(f'Figures written to {out}', flush=True)


if __name__ == '__main__':
    main()
