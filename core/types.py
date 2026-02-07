from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Mapping, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

# ---- Array aliases ----
ArrF = NDArray[np.floating]
VecF = NDArray[np.floating]
MatF = NDArray[np.floating]

# ---- Parameters are read-only name->float maps ----
Params = Mapping[str, float]

# ---- Dynamics interfaces ----
class StepMap(Protocol):
    """Discrete step: x_{k+1} = f_dt(x_k, u_k, p)."""
    def __call__(self, x: VecF, u: VecF, p: Params) -> VecF: ...

class ContDynamics(Protocol):
    """Continuous-time dynamics: xdot = f_ct(x, u, p)."""
    def __call__(self, x: VecF, u: VecF, p: Params) -> VecF: ...

class Linearizer(Protocol):
    """Return (A,B) at (x,u) for a given discrete map f_dt."""
    def __call__(self, f_dt: StepMap, x: VecF, u: VecF, p: Params) -> Tuple[MatF, MatF]: ...

# ---- Learned residual model ----
class ResidualModel(Protocol):
    """Delta(x,u) predicting the next-step residual in state space."""
    def predict_one(self, x: VecF, u: VecF) -> VecF: ...
    def predict(self, X: ArrF, U: ArrF) -> ArrF: ...  # optional fast batch

# ---- Controller interface ----
class Controller(Protocol):
    """Return control u_k at step k for state x."""
    def __call__(self, k: int, x: VecF) -> VecF: ...
    def reset(self, x0: Optional[VecF] = None) -> None: ...  # optional

