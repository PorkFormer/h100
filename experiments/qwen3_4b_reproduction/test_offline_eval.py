"""The legacy CLI delegates protocol execution to the shared implementation."""
import importlib.util
from pathlib import Path


def test_shared_core():
    spec = importlib.util.spec_from_file_location('legacy_evaluation', Path(__file__).with_name('evaluate.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cli.parser().parse_args(['status', '--output', '/tmp/unused']).command == 'status'
    from protocol import seed
    assert seed(512, 0) == 239391688
