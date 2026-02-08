import numpy as np
from numpy.typing import NDArray
from core.structs import Trajectory
from typing import Callable

Vec = np.ndarray

def clamp(u: NDArray, lo: NDArray, hi: NDArray) -> NDArray:
    return np.minimum(np.maximum(u, lo), hi)

def norm2(x: NDArray) -> float:
    x = np.asarray(x, float).ravel()
    return float(np.sqrt(x @ x))

def cholesky_solve(A: NDArray, b: NDArray) -> NDArray:
    """Solve Ax=b for SPD A via Cholesky (stable)."""
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)

def defects_norm(
    ref: Trajectory,
    f_dt: Callable[[Vec, Vec], Vec],
) -> float:
    """Sum of squared one-step defects ‖f(X[k],U[k]) - X[k+1]‖^2 along the horizon."""
    dsum = 0.0
    for k in range(ref.N):
        d = f_dt(ref.X.Y[k], ref.U.Y[k],k) - ref.X.Y[k+1]
        dsum += float(d @ d)
    return dsum