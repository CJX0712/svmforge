"""Fusion flagship — KernelFuse adaptive multi-kernel ensemble."""

from .kernel_fuse import KernelFuse, _SklearnSVCWrapper

__all__ = ["KernelFuse", "_SklearnSVCWrapper"]
