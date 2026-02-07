# control/feedforward.py
from __future__ import annotations
import numpy as np
from typing import Callable

Vec = np.ndarray


# ------------------------------
# Reference & cost
# ------------------------------

def stage_cost(x, u, xref, Q, R):
    dx = x - xref
    return dx.T @ Q @ dx + u.T @ R @ u

def terminal_cost(x, xref, P):
    dx = x - xref
    return dx.T @ P @ dx

def rollout(f_dt: Callable[[Vec, Vec], Vec], x0: Vec, U: np.ndarray) -> np.ndarray:
    X = np.zeros((U.shape[0] + 1, x0.size))
    X[0] = x0
    for k in range(U.shape[0]):
        X[k+1] = f_dt(X[k], U[k])
    return X

def simulate_tv_lqr(
    f_dt, x0, Xref, Uref, K_seq, p, Q, R, P,
    *,
    alpha_bounds = (np.deg2rad(-5.0), np.deg2rad(+5.0)),
    thr_bounds   = (0.05, 1.0),
    du_max       = (np.deg2rad(2.0), 0.15),   # per-step rate limits
    v_floor      = 1.0,                       # m/s; keep dynamics sane
    stop_if_vlow = True
):
    """
    TV-LQR with input saturation + per-step rate limits.
    Returns:
      X  : (N+1,n) states
      Uc : (N,m)   commanded u = uref - K(x-xref)   (pre-clip)
      Ua : (N,m)   applied u after clip/rate-limit  (what plant sees)
      cost : total stage+terminal cost (with Ua)
    """
    N = len(Uref)
    x = x0.astype(float).copy()
    X  = [x.copy()]
    Uc = []
    Ua = []
    cost = 0.0

    a_lo, a_hi = alpha_bounds
    t_lo, t_hi = thr_bounds
    da_max, dt_max = du_max

    # initialize applied u with first u_ref (clipped)
    u_prev = np.clip(Uref[0].astype(float).copy(),
                     [a_lo, t_lo], [a_hi, t_hi])

    for k in range(N):
        xref = Xref[k]
        uref = Uref[k]
        x    = np.asarray(x,    float).ravel()
        xref = np.asarray(xref, float).ravel()
        uref = np.asarray(uref, float).ravel()

        Kk   = np.asarray(K_seq[k], float)


        e    = x - xref
        du   = - Kk @ e
        u_cmd = (uref + du).astype(float)

        # 2) rate limit wrt previously applied control
        du = u_cmd - u_prev
        du[0] = np.clip(du[0], -da_max, +da_max)
        du[1] = np.clip(du[1], -dt_max, +dt_max)
        u_rl = u_prev + du

        # 3) hard saturation
        u_applied = np.array([np.clip(u_rl[0], a_lo, a_hi),
                              np.clip(u_rl[1], t_lo, t_hi)], float)

        # cost with applied inputs
        cost += stage_cost(x, u_applied, xref, Q, R)

        # 4) propagate
        x = f_dt(x, u_applied, p)
        # guard against pathological slow-downs
        if x[0] < v_floor:
            x[0] = v_floor

        # log & prepare next step
        Uc.append(u_cmd.copy())
        Ua.append(u_applied.copy())
        X.append(x.copy())
        u_prev = u_applied

        if stop_if_vlow and x[0] <= v_floor:
            # optional: break early if speed collapsed
            # break
            pass

    cost += terminal_cost(X[-1], Xref[-1], P)
    return np.array(X), np.array(Ua), cost