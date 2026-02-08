# control/feedforward.py
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from core.utils import cholesky_solve, defects_norm  # tiny helper, optional
from core.structs import Trajectory
import differentiate

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

def feasible_control(
    ref: Trajectory,
    f_dt: Callable[[Vec, Vec], Vec],   # wrap your model & params: lambda x,u: f_nom_dt(x,u,p_nom)
    Qff: np.ndarray,                   # use your state weight Q, or a tuned version
    *,
    rho: float = 1e-3,
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    du_clip: Optional[float] = 0.05,
) -> Trajectory:
    """One forward pass of 1-step Gauss–Newton to warm-start U_ref."""
    X, U = ref.X.Y, ref.U.Y.copy()
    N = U.shape[0]
    for k in range(N):
        f_k = lambda x, u, k=k: f_dt(x, u, k)
        U[k] = feedforward_step(
            f_hat_dt=f_k,
            Xk_ref=X[k], Xkp1_ref=X[k+1], Uk_ref=U[k], Qff=Qff,
            rho=rho, u_min=u_min, u_max=u_max, du_clip=du_clip
        )
    return ref.with_(U=U)

def rollout_nominal(f_dt: Callable[[Vec, Vec], Vec], x0: Vec, U: np.ndarray) -> np.ndarray:
    X = np.zeros((U.shape[0] + 1, x0.size))
    X[0] = x0
    for k in range(U.shape[0]):
        X[k+1] = f_dt(X[k], U[k],k)
    return X

def feasible_control_with_rollout(
    ref: Trajectory,
    f_dt: Callable[[Vec, Vec], Vec],
    Qff: np.ndarray,
    *,
    rho: float = 1e-3,
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    du_clip: Optional[float] = 0.05,
) -> Trajectory:
    # pass 1: adjust U along fixed X
    ref1 = feasible_control(ref, f_dt, Qff, rho=rho, u_min=u_min, u_max=u_max, du_clip=du_clip)
    # pass 2: refresh X by rolling out nominal with the new U
    X_new = rollout_nominal(f_dt, ref.X.Y[0], ref1.U.Y)
    return ref1.with_(X=X_new)

# -------- Optional: an iterative wrapper (1–3 passes) with a simple stop rule --------

def feasible_control_iterative(
    ref: Trajectory,
    f_dt: Callable[[Vec, Vec], Vec],
    Qff: np.ndarray,
    *,
    max_iters: int = 3,
    tol_rel: float = 1e-2,
    rho: float = 1e-3,
    u_min: Optional[Vec] = None,
    u_max: Optional[Vec] = None,
    du_clip: Optional[float] = 0.05,
) -> Trajectory:
    """
    1–3 passes of GN + rollout. Stops early if defects improve < tol_rel.
    """
    cur = ref
    prev_cost = defects_norm(cur, f_dt)
    for _ in range(max_iters):
        cur = feasible_control_with_rollout(
            cur, f_dt, Qff, rho=rho, u_min=u_min, u_max=u_max, du_clip=du_clip
        )
        cost = defects_norm(cur, f_dt)
        if prev_cost <= 0 or (prev_cost - cost) / max(prev_cost, 1e-12) < tol_rel:
            break
        prev_cost = cost
    return cur



def make_Qff(dt, w_v=1.0, w_gam=10.0, w_h_per_sec=10.0):
    # h accumulates over dt; scale by (w_h_per_sec/dt)^2 so 1-step mismatch in h is “felt”.
    w_h = (w_h_per_sec / dt)**2
    return np.diag([w_v, w_gam, w_h])

def riccati_backward(A_seq, B_seq, Q, R, P_N):
    """Time-varying discrete LQR via backward Riccati.
    Returns K_seq (feedback gains) and P_seq (cost-to-go) of length N.
    """
    N = len(A_seq)
    n, m = A_seq[0].shape[0], B_seq[0].shape[1]
    P = P_N.copy()
    K_seq = [None]*N
    P_seq = [None]*(N+1)
    P_seq[N] = P_N.copy()
    for k in reversed(range(N)):
        A, B = A_seq[k], B_seq[k]
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)  # K = S^{-1} B^T P A
        K_seq[k] = K
        P = Q + A.T @ P @ (A - B @ K)
        P_seq[k] = P.copy()
    return K_seq, P_seq


def tvlqr_controller(traj_ref, f_model_dt, p_model, Q, R, P_N):
    """
    traj_ref: Trajectory with t, X_ref, U_ref, dt
    f_model_dt: callable (x,u) -> x_next using the model for TVLQR
    p_model: parameters (if needed, otherwise captured in closure)
    Q, R: LQR weights
    P_N: Terminal P
    Returns: list of feedback gains K[k] and maybe A,B for debugging
    """
    Xref = traj_ref.X
    Uref = traj_ref.U
    N    = Uref.shape[0]

    A_seq = []
    B_seq = []

    for k in range(N):
        xk = Xref[k]
        uk = Uref[k]
        # central-difference Jacobian on the discrete-time map
        Ak, Bk = differentiate.numeric_central_diff(f_model_dt, xk, uk, p_model) 
        A_seq.append(Ak)
        B_seq.append(Bk)

    K_seq, P_seq = riccati_backward(A_seq, B_seq, Q, R,  P_N)  
    return K_seq, A_seq, B_seq