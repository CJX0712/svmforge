"""Evaluation metrics — pure numpy, no sklearn required (offline Tier-1)."""

from .metrics import accuracy, auc, ece, macro_f1

__all__ = ["accuracy", "auc", "ece", "macro_f1"]
