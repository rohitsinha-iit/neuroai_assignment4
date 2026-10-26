"""Scientific correctness checks for state, randomness, units and refractoriness."""
import unittest
from dataclasses import replace
import numpy as np
from snn_experiment.model import Config, make_weights, simulate, adapt_thresholds


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(Config(), n_hidden=16, encoding_ms=40.)
        self.weights = make_weights(self.config, 7)

    def test_reset_dark_image_and_refractory(self):
        images = np.stack([np.ones(784), np.zeros(784), np.ones(784)])
        result = simulate(images, self.config, self.weights, seed=123, diagnostics=True)
        self.assertGreater(result['counts'][0].sum(), 0)
        self.assertEqual(result['counts'][1].sum(), 0)
        t = result['voltage_times_ms']
        self.assertTrue(np.all(result['voltage_mv'][:, (t >= 40) & (t < 80)] == -65.))
        spike_t = result['spike_times_ms']
        for neuron in range(self.config.n_hidden):
            times = spike_t[result['spike_neurons'] == neuron]
            for image in [0, 2]:
                times_in_image = times[(times >= image * 40) & (times < (image + 1) * 40)]
                self.assertTrue(np.all(np.diff(times_in_image) >= 2.0 - 1e-8))

    def test_reproducible_and_paired_inputs(self):
        images = np.full((3, 784), 0.5)
        a = simulate(images, self.config, self.weights, seed=123)
        b = simulate(images, self.config, self.weights, seed=123)
        c = simulate(images, replace(self.config, tau_ms=5), self.weights, seed=123)
        np.testing.assert_array_equal(a['counts'], b['counts'])
        np.testing.assert_array_equal(a['input_counts'], c['input_counts'])
        self.assertFalse(np.array_equal(a['counts'], c['counts']))
        expected = 3 * 784 * 0.04 * 50
        self.assertLess(abs(a['input_counts'].sum() - expected), 6 * np.sqrt(expected))

    def test_homeostasis_reduces_overactivity(self):
        images = np.full((20, 784), 0.8)
        thresholds, _ = adapt_thresholds(images, self.config, self.weights,
                                         seed=9, target_hz=30, epochs=2,
                                         batch_size=5, learning_rate_mv=2)
        a = simulate(images[:3], self.config, self.weights, seed=44)
        b = simulate(images[:3], self.config, self.weights, thresholds, seed=44)
        self.assertTrue(np.all(thresholds >= self.config.threshold_mv))
        self.assertLess(b['counts'].sum(), a['counts'].sum())
        self.assertTrue(np.all(thresholds <= -30.))


if __name__ == '__main__':
    unittest.main()
