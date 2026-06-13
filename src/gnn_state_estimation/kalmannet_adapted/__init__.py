"""Adapted KalmanNet baseline for the SPOT-OD measurement setting (loop163).

This package holds the differentiable (torch-autograd) state-transition and
line-of-sight measurement functions used by the *adapted* KalmanNet
transposition. The central technical correction over the faithful Loop-42/57
transposition is that ``f`` and ``h`` are implemented natively in torch so the
computational graph is preserved across the recurrent rollout and
backpropagation-through-time reaches the learned gain network. The faithful
transposition bridged ``f``/``h`` through numpy with ``.detach()``, severing
BPTT; that is documented diagnosis root cause R3 (see the loop53 design-gap
quarantine note).

Nothing in this package is paper-facing. It owns only the ``kalmannet_adapted``
namespace.
"""

from .torch_dynamics import (
    SPOTODTorchDynamics,
    rk4_step_torch,
    state_derivative_torch,
    los_measurement_torch,
)

__all__ = [
    "SPOTODTorchDynamics",
    "rk4_step_torch",
    "state_derivative_torch",
    "los_measurement_torch",
]
