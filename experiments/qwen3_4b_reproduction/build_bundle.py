#!/usr/bin/env python3
"""Build a source-only archive from the attested runtime and reproduction files."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run.verify_source()
    run.verify_recipe()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = run.source_root()
    with tempfile.TemporaryDirectory(prefix='q4-bundle-') as temporary:
        bundle = Path(temporary) / 'qwen3_4b_reproduction'
        bundle.mkdir()
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if any(p in ('.venv', '__pycache__', '.pytest_cache', 'verl', 'offline_eval') for p in relative.parts):
                continue
            if not path.is_file() or path.suffix in ('.gz', '.zip', '.pyc'):
                continue
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        manifest = json.loads((ROOT / 'provenance/source.json').read_text())
        for name in manifest['files']:
            target = bundle / 'verl' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        core = ROOT / 'offline_eval' if (ROOT / 'offline_eval').is_dir() else ROOT.parents[1] / 'analysis/offline_eval'
        shutil.copytree(core, bundle / 'offline_eval', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        launcher = ROOT / 'offline_evaluate.sh' if (ROOT / 'offline_evaluate.sh').exists() else ROOT.parents[1] / 'evaluate.sh'
        shutil.copy2(launcher, bundle / 'offline_evaluate.sh')
        checksums = []
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                checksums.append(f'{run.sha256(path)}  {path.relative_to(bundle).as_posix()}')
        (bundle / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
        with tarfile.open(output, 'w:gz') as archive:
            archive.add(bundle, arcname=bundle.name)
    output.with_name(output.name + '.sha256').write_text(f'{run.sha256(output)}  {output.name}\n')
    print(output)
    print(f'{output.stat().st_size / (1 << 20):.1f} MiB; {len(manifest["files"])} attested source files')


if __name__ == '__main__':
    main()
