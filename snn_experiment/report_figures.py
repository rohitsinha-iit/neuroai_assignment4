"""Compact report figures derived exclusively from saved main-study data."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('results/main')
    out = root / 'figures'
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 200})
    names = ['baseline', 'homeostasis_60Hz', 'uniform_control']
    labels = ['Baseline', 'Homeostasis (60 Hz)', 'Uniform threshold']
    colors = ['#2563a6', '#e28329', '#3e876c']
    rates = [np.load(root / 'seed_42' / n / 'predictions.npz')['neuron_rates_hz'] for n in names]
    bins = np.linspace(np.min(rates) - 1, np.max(rates) + 1, 20)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), layout='constrained')
    for r, label, color in zip(rates, labels, colors):
        axes[0].hist(r, bins=bins, histtype='step', linewidth=1.8, color=color, label=label)
        axes[1].plot(np.sort(r), color=color)
    axes[0].set(xlabel='Mean neuron rate (Hz)', ylabel='Neurons per common bin')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Neuron rank (sorted separately)', ylabel='Mean neuron rate (Hz)')
    fig.savefig(out / 'report_rates.png', bbox_inches='tight')
    plt.close(fig)

    names = ['tau_5ms', 'baseline', 'tau_50ms']
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 5.6), layout='constrained')
    for row, (name, tau) in enumerate(zip(names, [5, 20, 50])):
        with np.load(root / 'seed_42' / name / 'diagnostics.npz') as d:
            spike_t, neurons = d['spike_times_ms'], d['spike_neurons']
            mask = (spike_t < 100) & (neurons < 40)
            axes[row, 0].scatter(spike_t[mask], neurons[mask], s=3, color=colors[row])
            axes[row, 0].set(xlim=(0, 100), ylim=(-1, 40), ylabel=f'{tau} ms\nNeuron index')
            if not mask.any():
                axes[row, 0].text(50, 20, 'No spikes in these neurons', ha='center', fontsize=9)
            mask_v = d['voltage_times_ms'] < 100
            for neuron in range(3):
                axes[row, 1].plot(d['voltage_times_ms'][mask_v], d['voltage_mv'][neuron, mask_v], lw=.8)
            axes[row, 1].axhline(-50, ls='--', color='#777777', lw=.8)
            axes[row, 1].set(xlim=(0, 100), ylim=(-66, -47), ylabel='Voltage (mV)')
            digit = int(d['labels'][0])
    axes[0, 0].set_title(f'Same training digit {digit}, seed 42: raster')
    axes[0, 1].set_title('First 3 neurons; dashed line = threshold')
    for ax in axes[-1]:
        ax.set_xlabel('Time within image (ms)')
    fig.savefig(out / 'report_tau_traces.png', bbox_inches='tight')
    plt.close(fig)
    print('Report figures generated from saved diagnostics.')


if __name__ == '__main__':
    main()
