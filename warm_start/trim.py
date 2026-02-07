from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from models.continuous_model import ContinuousModel
from core.signal import Signal

Vec = np.ndarray

def _central_jacobian_z(
    sys: ContinuousModel,
    x: Vec,
    z: Vec,
    eps_alpha: float = 1e-5,
    eps_thr: float = 1e-3,
) -> np.ndarray:
    """
    J = ∂[dv, dgamma]/∂z evaluated at (x,z), z=[alpha, thr].
    Uses sys.f_ct(x, z, sys.params).
    """
    J = np.zeros((2, 2), dtype=float)
    eps = np.array([eps_alpha, eps_thr], float)

    for j in range(2):
        dz = np.zeros(2); dz[j] = eps[j]
        fp = sys.f_ct(x, z + dz, sys.params)[:2]
        fm = sys.f_ct(x, z - dz, sys.params)[:2]
        J[:, j] = (fp - fm) / (2 * eps[j])

    return J


def trim_nominal(
    sys: ContinuousModel,
    x: Vec,
    *,
    z0: Optional[Vec] = None,
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    max_iters: int = 25,
    tol: float = 1e-8,
    lm_lambda: float = 1e-2,
    backtrack: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
) -> Vec:

    v, gamma, h = map(float, x)

    if z0 is None:
        z = np.array([np.deg2rad(2.0), 0.6], float)
    else:
        z = np.array(z0, float)

    if u_min is None: u_min = np.array([np.deg2rad(-10.0), 0.0])
    if u_max is None: u_max = np.array([np.deg2rad(+10.0), 1.0])

    def residual(z):
        dv, dga, _ = sys.f_ct(np.array([v, gamma, h]), z, sys.params)
        return np.array([dv, dga], float)

    r = residual(z)

    for _ in range(max_iters):
        if np.linalg.norm(r) < tol:
            break

        J = _central_jacobian_z(sys, np.array([v, gamma, h]), z)
        H = J.T @ J + lm_lambda * np.eye(2)
        g = J.T @ r

        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(H) @ g

        accepted = False
        for a in backtrack:
            z_try = np.clip(z + a * step, u_min, u_max)
            r_try = residual(z_try)
            if np.linalg.norm(r_try) <= np.linalg.norm(r):
                z, r = z_try, r_try
                accepted = True
                break

        if not accepted:
            z = np.clip(z + 0.1 * step, u_min, u_max)
            r = residual(z)

    return z

def trim_rate(
    sys: ContinuousModel,
    x: Vec,
    vdot_des: float,
    gammadot_des: float,
    *,
    z0: Optional[Vec] = None,
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    max_iters: int = 20,
    tol: float = 1e-6,
    lm_lambda: float = 1e-2,
    backtrack: Tuple[float,...] = (1.0, 0.5, 0.25, 0.1),
    du_clip: Tuple[float,float] = (np.deg2rad(1.0), 0.1),
) -> Vec:

    x = np.asarray(x, float)

    if z0 is None:
        z = np.array([np.deg2rad(2.0), 0.6], float)
    else:
        z = np.array(z0, float)

    if u_min is None: u_min = np.array([np.deg2rad(-10.0), 0.15])
    if u_max is None: u_max = np.array([np.deg2rad(+10.0), 1.00])

    y_des = np.array([vdot_des, gammadot_des], float)

    def residual(z):
        dv, dga, _ = sys.f_ct(x, z, sys.params)
        return np.array([dv, dga], float) - y_des

    r = residual(z)

    for _ in range(max_iters):
        if np.linalg.norm(r) <= tol:
            break

        J = _central_jacobian_z(sys, x, z)
        H = J.T @ J + lm_lambda * np.eye(2)
        g = J.T @ r

        try:
            L = np.linalg.cholesky(H)
            step = -np.linalg.solve(L.T, np.linalg.solve(L, g))
        except np.linalg.LinAlgError:
            step = -np.linalg.solve(H + 1e-10*np.eye(2), g)

        step = np.clip(step, [-du_clip[0], -du_clip[1]], [du_clip[0], du_clip[1]])

        accepted = False
        for a in backtrack:
            z_try = np.clip(z + a * step, u_min, u_max)
            r_try = residual(z_try)
            if np.linalg.norm(r_try) <= np.linalg.norm(r) - 1e-12:
                z, r = z_try, r_try
                accepted = True
                break

        if not accepted:
            z_try = np.clip(z + 0.1 * step, u_min, u_max)
            r_try = residual(z_try)
            z, r = z_try, r_try

    return z


def warm_start_trim_U(
    traj,
    nom_sys: ContinuousModel,
    *,
    reuse_previous: bool = True,
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    trim: str = 'rate',
):
    N = traj.N
    U_trim = np.zeros_like(traj.U.Y)   # raw array of shape (N, nu)
    z_guess = None

    for k in range(N):
        xk = traj.X.Y[k]

        if trim == 'nominal':
            z = trim_nominal(
                nom_sys, xk,
                z0=z_guess,
                u_min=u_min,
                u_max=u_max
            )

        elif trim == 'rate':
            vdot_des     = (traj.X.Y[k+1,0] - traj.X.Y[k,0]) / traj.dt
            gammadot_des = (traj.X.Y[k+1,1] - traj.X.Y[k,1]) / traj.dt

            z = trim_rate(
                nom_sys, xk,
                vdot_des=vdot_des,
                gammadot_des=gammadot_des,
                z0=z_guess,
                u_min=u_min,
                u_max=u_max
            )

        U_trim[k] = z

        if reuse_previous:
            z_guess = z

    # Wrap into a Signal
    U_sig = Signal(
        t=traj.t[:-1],
        Y=U_trim,
        dt=traj.dt,
        labels=traj.U.labels
    )

    return U_sig
