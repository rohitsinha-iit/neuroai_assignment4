"""Actual Brian2 Poisson -> excitatory synapses -> refractory LIF simulation."""
from dataclasses import dataclass, asdict
import time
import numpy as np
import brian2 as b2


@dataclass(frozen=True)
class Config:
    n_hidden: int = 256
    encoding_ms: float = 100.0
    dt_ms: float = 0.5
    max_rate_hz: float = 100.0
    tau_ms: float = 20.0
    rest_mv: float = -65.0
    reset_mv: float = -65.0
    threshold_mv: float = -50.0
    refractory_ms: float = 2.0
    connection_probability: float = 1.0
    weight_mean_mv: float = 0.12
    weight_sd_mv: float = 0.12
    backend: str = 'numpy'

    def dictionary(self):
        return asdict(self)


def make_weights(config, seed):
    rng = np.random.default_rng(seed)
    weights = np.maximum(rng.normal(config.weight_mean_mv, config.weight_sd_mv,
                                    (784, config.n_hidden)), 0.0)
    if config.connection_probability < 1:
        weights *= rng.random(weights.shape) < config.connection_probability
    return weights


def simulate(images, config, weights, thresholds=None, seed=0, diagnostics=False):
    """Simulate a contiguous image block, explicitly resetting state each image.

    A single PoissonGroup and TimedArray avoid rebuilding a network per image.
    Identical seeds and images pair input spike trains across model conditions.
    There are no synaptic delays or synaptic state variables to carry across images.
    """
    started = time.perf_counter()
    images = np.asarray(images)
    if images.ndim != 2 or images.shape[1] != 784 or not len(images):
        raise ValueError('Expected a nonempty (images, 784) array.')
    if np.any(images < 0) or np.any(images > 1):
        raise ValueError('Pixel values must be in [0, 1].')
    steps = round(config.encoding_ms / config.dt_ms)
    if not np.isclose(steps * config.dt_ms, config.encoding_ms):
        raise ValueError('Encoding interval must be an integer number of timesteps.')
    if config.max_rate_hz * config.dt_ms / 1000 > 0.1:
        raise ValueError('Use a finer timestep for this Poisson rate.')
    b2.start_scope()
    b2.prefs.codegen.target = config.backend
    b2.defaultclock.dt = config.dt_ms * b2.ms
    b2.seed(seed)
    stimulus = b2.TimedArray(images * config.max_rate_hz * b2.Hz,
                            dt=config.encoding_ms * b2.ms)
    inputs = b2.PoissonGroup(784, rates='stimulus(t, i)',
                             namespace={'stimulus': stimulus}, name='input_pixels')
    hidden = b2.NeuronGroup(
        config.n_hidden,
        'dv/dt = (v_rest - v)/tau_m : volt (unless refractory)\n'
        'theta : volt (constant)',
        threshold='v >= theta', reset='v = v_reset',
        refractory=config.refractory_ms * b2.ms, method='exact',
        namespace={'v_rest': config.rest_mv * b2.mV,
                   'v_reset': config.reset_mv * b2.mV,
                   'tau_m': config.tau_ms * b2.ms}, name='hidden_lif')
    hidden.v = config.rest_mv * b2.mV
    hidden.theta = ((np.full(config.n_hidden, config.threshold_mv)
                     if thresholds is None else thresholds) * b2.mV)
    hidden.run_regularly(
        'not_refractory = True\nlastspike = t - 1*second\nv = v_rest',
        dt=config.encoding_ms * b2.ms, when='start', order=-10,
        name='reset_between_images')
    synapses = b2.Synapses(inputs, hidden, model='w : volt',
                           on_pre='v_post += w', name='input_synapses')
    ii, jj = np.nonzero(weights)
    synapses.connect(i=ii, j=jj)
    synapses.w = weights[ii, jj] * b2.mV
    spikes = b2.SpikeMonitor(hidden, name='hidden_spikes')
    input_spikes = b2.SpikeMonitor(inputs, record=False, name='input_spikes')
    objects = [inputs, hidden, synapses, spikes, input_spikes]
    if diagnostics:
        voltage = b2.StateMonitor(hidden, 'v', record=np.arange(min(6, config.n_hidden)),
                                   when='end', name='voltage_traces')
        objects.append(voltage)
    net = b2.Network(objects)
    net.run(len(images) * config.encoding_ms * b2.ms, namespace={})
    times_ms = np.asarray(spikes.t / b2.ms)
    neurons = np.asarray(spikes.i, dtype=np.int64)
    bins = np.rint(times_ms / config.dt_ms).astype(np.int64) // steps
    if len(bins) and (bins.min() < 0 or bins.max() >= len(images)):
        raise AssertionError('Spike assigned outside its image interval.')
    counts = np.bincount(bins * config.n_hidden + neurons,
                         minlength=len(images) * config.n_hidden)
    counts = counts.reshape(len(images), config.n_hidden).astype(np.uint16)
    input_counts = np.asarray(input_spikes.count, dtype=np.int64)
    result = {'counts': counts, 'input_counts': input_counts,
              'synaptic_events': int(input_counts @ np.count_nonzero(weights, axis=1)),
              'wall_seconds': time.perf_counter() - started}
    if diagnostics:
        result.update(spike_times_ms=times_ms, spike_neurons=neurons,
                      voltage_mv=np.asarray(voltage.v / b2.mV),
                      voltage_times_ms=np.asarray(voltage.t / b2.ms))
    return result


def activity(counts, config):
    rates = counts.mean(axis=0) / (config.encoding_ms / 1000)
    return {'mean_rate_hz': float(rates.mean()),
            'spikes_per_image': float(counts.sum(axis=1).mean()),
            'silent_fraction': float(np.mean(rates == 0)),
            'high_rate_fraction': float(np.mean(rates > 0.8 * 1000 / config.refractory_ms)),
            'between_neuron_rate_sd_hz': float(rates.std()),
            'variable_feature_fraction': float(np.mean(counts.std(axis=0) > 0))}


def adapt_thresholds(images, config, weights, seed, target_hz=30., epochs=2,
                     batch_size=100, learning_rate_mv=1.0):
    thresholds = np.full(config.n_hidden, config.threshold_mv)
    smooth_rate = np.full(config.n_hidden, target_hz)
    history = []
    rng = np.random.default_rng(seed + 70000)
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(images))
        for begin in range(0, len(images), batch_size):
            result = simulate(images[order[begin:begin + batch_size]], config, weights,
                              thresholds, seed=seed + 80000 + step)
            rate = result['counts'].mean(axis=0) / (config.encoding_ms / 1000)
            smooth_rate = 0.5 * smooth_rate + 0.5 * rate
            thresholds = np.clip(thresholds + learning_rate_mv *
                                 (smooth_rate - target_hz) / target_hz,
                                 config.reset_mv + 2, -30.)
            history.append({'step': step, 'epoch': epoch, 'rate_hz': rate.tolist(),
                            'threshold_mv': thresholds.tolist()})
            print(f'  adaptation epoch {epoch + 1}/{epochs}, '
                  f'images {min(begin + batch_size, len(images))}/{len(images)}, '
                  f'mean rate {rate.mean():.2f} Hz', flush=True)
            step += 1
    return thresholds, history
