"""Run the experiment and generate figures, retaining a machine-readable exit state."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    args = sys.argv[1:]
    output = Path(args[args.index('--output') + 1] if '--output' in args else 'results/main')
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stages = [('experiment', 'snn_experiment.run', args),
              ('figures', 'snn_experiment.plot', ['--output', str(output)])]
    for stage, module, arguments in stages:
        print(f'START {stage}', flush=True)
        result = subprocess.run([sys.executable, '-u', '-X', 'utf8', '-m', module, *arguments])
        status = {'stage': stage, 'exit_code': result.returncode, 'wall_seconds': time.time() - started}
        (output / 'execution.json').write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise SystemExit(result.returncode)
    (output / 'execution.json').write_text(json.dumps(
        {'stage': 'complete', 'exit_code': 0, 'wall_seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
