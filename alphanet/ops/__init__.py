"""Fused Triton operators for AlphaNet inference (plan phase 3).

Import is guarded: on machines without a CUDA device or without Triton the
package degrades to ``is_available() == False`` and the model falls back to
the reference (pure PyTorch) implementation, which remains the source of
truth for training and for fp64.
"""
import torch

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    _HAS_TRITON = False


def is_available() -> bool:
    """Fused ops run on CUDA + fp32 only (fp64 falls back to reference)."""
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:
    from alphanet.ops.scalarization import fused_scalarization_mlp  # noqa: F401
    from alphanet.ops.message import fused_equi_message  # noqa: F401
