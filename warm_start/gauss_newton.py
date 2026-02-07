from __future__ import annotations
import numpy as np
from models.continuous_model import ContinuousModel
from core.signal import Signal
from trajectory import Trajectory
from typing import Callable, Optional, Tuple, Dict, Any
from .utils import cholesky_solve

Vec = np.ndarray

def central_jacobian_u(f: Callable[[Vec, Vec], Vec],
                       x: Vec, u: Vec, h: Optional[Vec] = None) -> np.ndarray:
    """J_u ≈ ∂f/∂u at (x,u) via central differences; f: (x,u)->x_next."""
    x = np.asarray(x, float); u = np.asarray(u, float)
    nx, m = x.size, u.size
    J = np.zeros((nx, m))
    if h is None:
        h = 1e-6 * (1.0 + np.abs(u))
    for j in range(m):
        du = np.zeros_like(u); du[j] = h[j]
        fp = f(x, u + du)
        fm = f(x, u - du)
        J[:, j] = (fp - fm) / (2.0 * h[j])
    return J

def feedforward_step(
    f_hat_dt: Callable[[Vec, Vec], Vec],
    Xk_ref: Vec,
    Xkp1_ref: Vec,
    Uk_ref: Vec,
    Qff: np.ndarray,
    *,
    rho: float = 1e-3,                 # Tikhonov/LM damping
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    du_clip: Optional[float] = None,   # trust-region on step size (per-component)
    jac_u: Optional[Callable[[Vec, Vec], np.ndarray]] = None,  # supply analytic J_u if you have it
    return_info: bool = False,
) -> Tuple[Vec, Dict[str, Any]] | Vec:
    """
    One Gauss–Newton / Levenberg–Marquardt step around (Xk_ref, Uk_ref):
      minimize  || f_hat_dt(Xk_ref, u) - Xkp1_ref ||_Qff^2 + rho ||u - Uk_ref||^2

    Returns u_ff (and optional diagnostics).
    """
    Xk_ref = np.asarray(Xk_ref, float); Xkp1_ref = np.asarray(Xkp1_ref, float)
    Uk_ref = np.asarray(Uk_ref, float); Qff = np.asarray(Qff, float)

    # residual and Jacobian wrt u
    r = f_hat_dt(Xk_ref, Uk_ref) - Xkp1_ref                      # (nx,)
    J = jac_u(Xk_ref, Uk_ref) if jac_u is not None else central_jacobian_u(
        f_hat_dt, Xk_ref, Uk_ref
    )                                                            # (nx, m)

    # GN/LM normal equations
    H = J.T @ Qff @ J + rho * np.eye(J.shape[1])                 # (m,m)
    g = J.T @ Qff @ r                                            # (m,)

    # robust solve (Cholesky preferred; fall back to solve)
    try:
        du = -cholesky_solve(H, g)
    except np.linalg.LinAlgError:
        du = -np.linalg.solve(H + 1e-10 * np.eye(H.shape[0]), g)

    # trust-region clip
    if du_clip is not None:
        du = np.clip(du, -du_clip, du_clip)

    # box constraints
    u_ff = Uk_ref + du
    if u_min is not None: u_ff = np.maximum(u_ff, u_min)
    if u_max is not None: u_ff = np.minimum(u_ff, u_max)

    if return_info:
        info = {
            "r_norm": float(r.T @ Qff @ r),
            "du_norm": float(np.linalg.norm(du)),
            "cond_H": float(np.linalg.cond(H)),
        }
        return u_ff, info
    return u_ff

def warm_start_gauss_newton_U(
    traj: Trajectory,
    nom_sys: ContinuousModel,
    *,
    Qff: np.ndarray,
    rho: float = 1e-3,
    u_min=None, u_max=None,
    du_clip=0.05,
):
    X = traj.X.Y
    U = traj.U.Y.copy()
    N = U.shape[0]

    # wrap model into f_dt(x,u)
    f_dt = lambda x, u: nom_sys.f_dt(x, u, nom_sys.params)

    for k in range(N):
        U[k] = feedforward_step(
            f_hat_dt=f_dt,
            Xk_ref=X[k],
            Xkp1_ref=X[k+1],
            Uk_ref=U[k],
            Qff=Qff,
            rho=rho,
            u_min=u_min,
            u_max=u_max,
            du_clip=du_clip,
        )

    # wrap back into a Signal
    U_sig = Signal(
        t=traj.t[:-1],
        Y=U,
        dt=traj.dt,
        labels=traj.U.labels,
    )

    return traj.with_(U=U_sig)
