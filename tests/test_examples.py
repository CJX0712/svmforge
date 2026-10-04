"""Lock the demonstration scripts into CI (failure cases + ablation)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_failure_cases_is_safe():
    mod = _load("failure_cases", ROOT / "examples" / "failure_cases.py")
    assert mod.main() == 0


def test_ablation_non_inferior():
    mod = _load("ablation", ROOT / "examples" / "ablation.py")
    assert mod.main() == 0
