import json
import sys

import pytest


@pytest.fixture
def _argv(monkeypatch):
    saved = sys.argv
    yield
    sys.argv = saved


def test_cli_check(_argv, capsys):
    sys.argv = ["cli", "check"]
    import cli

    rc = cli.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "[check] RESULT: PASS" in out


def test_cli_bench_writes_json(_argv, tmp_path, capsys):
    out = str(tmp_path / "bench.json")
    sys.argv = ["cli", "bench", "--datasets", "moons", "--seeds", "2", "--out", out]
    import cli

    rc = cli.main()
    assert rc in (0, 1)  # gate may pass or fail; file must exist
    with open(out, encoding="utf-8") as fh:
        data = json.loads(fh.read())
    assert "summary" in data
    assert "reports" in data
