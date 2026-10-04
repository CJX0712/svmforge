# SVMForge — developer Makefile (author 晨星)
# Usage: make <target>

PY := python

.PHONY: help setup lint format test demo bench check ci ablate failures clean

help:
	@echo "Targets: setup lint format test demo bench check ci ablate failures"

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install ruff pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest -q

demo:
	$(PY) examples/run_demo.py

bench:
	$(PY) cli.py bench --out benchmark.json

check:
	$(PY) cli.py check

ablate:
	$(PY) examples/ablation.py

failures:
	$(PY) examples/failure_cases.py

ci: lint test demo

clean:
	$(PY) -c "import pathlib,os; [p.unlink() for p in pathlib.Path('.').rglob('__pycache__/**/*.pyc')]" || true
