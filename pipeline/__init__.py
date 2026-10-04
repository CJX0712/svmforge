"""Pipeline: train -> evaluate -> benchmark -> gate verdict."""

from .train_eval import (
    FLAGSHIP,
    METHODS,
    NAIVE_METHODS,
    SINGLE_SOTA,
    evaluate,
    run_benchmark,
    summarize,
)

__all__ = [
    "FLAGSHIP",
    "METHODS",
    "NAIVE_METHODS",
    "SINGLE_SOTA",
    "evaluate",
    "run_benchmark",
    "summarize",
]
