"""Error taxonomy with stable codes E100..E500.

Every raised error in svmforge derives from SVMForgeError so callers can
catch the whole family. Codes are stable across versions (DoD traceability).
"""

from __future__ import annotations


class SVMForgeError(Exception):
    """Base class for all svmforge errors."""

    code = "E000"
    doc = "unspecified error"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(
            f"[{self.code}] {self.doc}: {message}"
            if message
            else f"[{self.code}] {self.doc}"
        )


class DataLeakError(SVMForgeError):
    """E100 — detected or suspected train/test leakage."""

    code = "E100"
    doc = "data leakage (preprocessor fit on holdout / shuffle on time series)"


class StatisticalError(SVMForgeError):
    """E200 — statistical contract violated (e.g. undefined metric)."""

    code = "E200"
    doc = "statistical contract violation"


class NumericsError(SVMForgeError):
    """E300 — numerical invariant failed (KKT / dual objective / symmetry)."""

    code = "E300"
    doc = "numerical invariant failed"


class WindowsError_(SVMForgeError):
    """E400 — environment / IO failure."""

    code = "E400"
    doc = "environment or IO failure"


class ConfigError(SVMForgeError):
    """E500 — configuration schema validation failed."""

    code = "E500"
    doc = "configuration schema violation"


__all__ = [
    "ConfigError",
    "DataLeakError",
    "NumericsError",
    "SVMForgeError",
    "StatisticalError",
    "WindowsError_",
]
