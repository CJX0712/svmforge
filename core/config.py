"""Configuration with ENV_XXX_* overrides and schema validation.

The single source of truth is ``Config.from_env()``. All thresholds and
dataset sizing flow from one config object so the benchmark is reproducible
and its gates are auditable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError

DEFAULT_SEED = 20261004
DEFAULT_N_TRAIN = 600
DEFAULT_N_TEST = 400
DEFAULT_N_VAL = 150
DEFAULT_N_SEEDS = 3
DEFAULT_DEMO_TIMEOUT = 60.0
DEFAULT_C = 1.0
DEFAULT_TOL = 1e-3


@dataclass(frozen=True)
class Config:
    seed: int = DEFAULT_SEED
    n_train: int = DEFAULT_N_TRAIN
    n_test: int = DEFAULT_N_TEST
    n_val: int = DEFAULT_N_VAL
    n_seeds: int = DEFAULT_N_SEEDS
    demo_timeout: float = DEFAULT_DEMO_TIMEOUT
    C: float = DEFAULT_C
    tol: float = DEFAULT_TOL
    # Performance gate (DoD): KernelFuse must beat naive baseline by this margin.
    gate_accuracy_margin: float = 0.01
    # Non-inferiority tolerance vs best single backend.
    gate_noninferior_tol: float = 0.005

    def validate(self) -> Config:
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        if self.n_train <= 0 or self.n_test <= 0 or self.n_val < 0:
            raise ConfigError("dataset sizes must be positive (val may be 0)")
        if self.n_seeds < 1:
            raise ConfigError("n_seeds must be >= 1")
        if self.C <= 0 or self.tol <= 0:
            raise ConfigError("C and tol must be positive")
        if not (0.0 < self.gate_accuracy_margin < 0.5):
            raise ConfigError("gate_accuracy_margin out of sane range")
        return self

    @classmethod
    def from_env(cls, prefix: str = "SVMFORGE_") -> Config:
        def _int(name: str, default: int) -> int:
            v = os.environ.get(prefix + name)
            return int(v) if v is not None else default

        def _float(name: str, default: float) -> float:
            v = os.environ.get(prefix + name)
            return float(v) if v is not None else default

        cfg = cls(
            seed=_int("SEED", DEFAULT_SEED),
            n_train=_int("N_TRAIN", DEFAULT_N_TRAIN),
            n_test=_int("N_TEST", DEFAULT_N_TEST),
            n_val=_int("N_VAL", DEFAULT_N_VAL),
            n_seeds=_int("N_SEEDS", DEFAULT_N_SEEDS),
            demo_timeout=_float("DEMO_TIMEOUT", DEFAULT_DEMO_TIMEOUT),
            C=_float("C", DEFAULT_C),
            tol=_float("TOL", DEFAULT_TOL),
            gate_accuracy_margin=_float("GATE_MARGIN", 0.01),
            gate_noninferior_tol=_float("GATE_NONINF", 0.005),
        )
        return cfg.validate()


@dataclass(frozen=True)
class GateSpec:
    """Hard-coded performance gates (DoD). Never silently relaxed."""

    accuracy_margin_vs_naive: float = 0.01
    noninferior_vs_best_single: float = 0.005
    min_datasets: int = 5
    min_seeds: int = 3

    def as_dict(self) -> dict:
        return {
            "accuracy_margin_vs_naive": self.accuracy_margin_vs_naive,
            "noninferior_vs_best_single": self.noninferior_vs_best_single,
            "min_datasets": self.min_datasets,
            "min_seeds": self.min_seeds,
        }


# Populated at runtime; kept importable for docs generation.
ACTIVE_GATE: GateSpec = GateSpec()
