"""Reproduce the frozen comparisons and CPU integration checks without installing verl."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[3]
output = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="ncbr-portable-verify-"))
output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", XDG_CACHE_HOME=str(output/"cache"))
summary = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
           "comparisons": {}, "commands": []}

def run(name, command):
    summary["commands"].append(command)
    with (output/f"{name}.log").open("x") as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

for script, golden in (("characterize.py", "golden.json"), ("characterize_boundaries.py", "boundary_golden.json")):
    artifact = output/golden
    run(script, [sys.executable, str(here/script), "--root", str(root), "--output", str(artifact)])
    assert artifact.read_bytes() == (here/golden).read_bytes(), f"Frozen characterization differs: {golden}"
    summary["comparisons"][golden] = {"exact": True, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}

paths = ["verl/tests/experimental/natural_continuation_boundary_return",
         "verl/tests/experimental/ncbr_portable/test_standard_on_cpu.py",
         "verl/tests/experimental/ncbr_portable/test_hook_on_cpu.py",
         "verl/tests/experimental/probe_credit/test_dapo_trainer_on_cpu.py",
         "verl/tests/experimental/probe_credit/test_dynamic_sampling_on_cpu.py"]
bootstrap = (
    f"import sys; sys.path.insert(0, {str(root/'verl')!r}); "
    "import ray; "
    "ray.init=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Ray initialization forbidden')); "
    "import pytest; "
    f"raise SystemExit(pytest.main({['-q', '-p', 'no:cacheprovider', *paths]!r}))"
)
run("pytest", [sys.executable, "-c", bootstrap])
run("diff-check", ["git", "diff", "--check"])
summary["status"] = "PASS"
(output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(f"PASS: {output}")
