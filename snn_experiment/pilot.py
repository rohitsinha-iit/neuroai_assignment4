"""Measure runtime and activity before committing to the full experiment."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from .data import load_splits
from .model import Config, make_weights, simulate, activity


def main():
    output = Path('results/pilot')
    data = load_splits(200, 100, 100, output)
    images = data['train'][0][:30]
    results = []
    for weight in [0.06, 0.12, 0.24]:
        cfg = replace(Config(), weight_mean_mv=weight, weight_sd_mv=weight)
        result = simulate(images, cfg, make_weights(cfg, 42), seed=10042)
        row = {'weight_mean_mv': weight, **activity(result['counts'], cfg),
               'wall_seconds': result['wall_seconds'], 'n_images': len(images)}
        print(json.dumps(row), flush=True)
        results.append(row)
    (output / 'calibration.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
