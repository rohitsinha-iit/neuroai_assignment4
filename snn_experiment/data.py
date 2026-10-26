"""MNIST download and fixed stratified splits, without a deep-learning dependency."""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'


def download(root='data'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'mnist.npz'
    if not path.exists():
        temporary = path.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(path)
    with np.load(path, allow_pickle=False) as d:
        assert d['x_train'].shape == (60000, 28, 28)
        assert d['x_test'].shape == (10000, 28, 28)
        assert d['x_train'].dtype == np.uint8
        assert set(np.unique(d['y_train'])) == set(range(10))
    metadata = {'url': URL, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / 'source.json').write_text(json.dumps(metadata, indent=2))
    return path


def balanced_indices(labels, count, rng):
    if count < 10 or count > len(labels):
        raise ValueError('Requested split must have between 10 and the available samples.')
    parts = []
    for c in range(10):
        candidates = np.flatnonzero(labels == c)
        n = count // 10 + int(c < count % 10)
        if n > len(candidates):
            raise ValueError('Not enough examples in a class for a balanced split.')
        parts.append(rng.permutation(candidates)[:n])
    return rng.permutation(np.concatenate(parts))


def load_splits(train_size, val_size, test_size, output, seed=2026):
    path = download()
    rng = np.random.default_rng(seed)
    with np.load(path, allow_pickle=False) as d:
        selected = balanced_indices(d['y_train'], train_size + val_size, rng)
        relative_val = balanced_indices(d['y_train'][selected], val_size, rng)
        mask = np.ones(len(selected), dtype=bool)
        mask[relative_val] = False
        indices = {'train': selected[mask], 'val': selected[relative_val]}
        # A full test set preserves its original distribution and order.
        indices['test'] = (np.arange(10000) if test_size == 10000 else
                           balanced_indices(d['y_test'], test_size, rng))
        assert not np.intersect1d(indices['train'], indices['val']).size
        result = {}
        for name, idx in indices.items():
            source = 'test' if name == 'test' else 'train'
            result[name] = (d['x_' + source][idx].reshape(-1, 784).astype(np.float32) / 255,
                            d['y_' + source][idx].copy())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'split_indices.npz', **indices)
    return result


if __name__ == '__main__':
    print(download())
