
import numpy as np
from models import stage_cost, terminal_cost,eval_nominal,eval_true
from control import feedforward_step

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
        K    = K_seq[k]
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


def simulate_tv_lqr_with_ff(f_true_dt, x0, Xref, Uref, K_seq, p_true,
                            f_hat_dt, Qff, u_min, u_max,
                            du_clip=0.05, rho=1e-3):
    """
    Closed loop:
      u_k = u_ff[k] - K_k (x_k - Xref[k]),
      u_ff[k] = one-step inverse via f_hat_dt around (Xref[k], Uref[k]).
    Returns: X_true, U_applied, cost_true
    """
    N = Uref.shape[0]
    nx, nu = Xref.shape[1], Uref.shape[1]
    X = np.zeros((N+1, nx)); U = np.zeros((N, nu))
    X[0] = x0
    cost = 0.0
    for k in range(N):
        # 1-step feedforward (at reference)
        u_ff = feedforward_step(f_hat_dt, Xref[k], Xref[k+1], Uref[k],
                                Qff=Qff, rho=rho, u_min=u_min, u_max=u_max, du_clip=du_clip)
        # feedback
        e = X[k] - Xref[k]
        u = u_ff - K_seq[k] @ e
        u = np.clip(u, u_min, u_max)
        U[k] = u
        X[k+1] = f_true_dt(X[k], u, p_true)
    return X, U

def collect_residuals(f_nom_dt, f_true_dt, X, U, p_nom, p_true, return_nominals=False):
    """
    E[k] = x_{k+1}^{true} - f_nom_dt(X[k], U[k])
    Optionally also returns:
      X_nom_1step: nominal one-step predictions from true X_k
      X_nom_roll : nominal rollout under U (self-propagating)
    """
    X = np.asarray(X); U = np.asarray(U)
    N, nx = U.shape[0], X.shape[1]

    E = np.zeros((N, nx))

    if return_nominals:
        X_nom_1step = np.zeros((N+1, nx)); X_nom_1step[0] = X[0]
        #X_nom_roll  = np.zeros((N+1, nx)); X_nom_roll[0]  = X[0]

    for k in range(N):
        xk, uk = X[k], U[k]
        x_next_true = f_true_dt(xk, uk, p_true)
        x_next_nom1 = f_nom_dt(xk, uk, p_nom)               # 1-step from true X[k]
        if return_nominals:
            #x_next_nomr = f_nom_dt(X_nom_roll[k], uk, p_nom) # rollout from nominal X_nom_roll[k]
            X_nom_1step[k+1] = x_next_nom1
            #X_nom_roll[k+1]  = x_next_nomr
        E[k] = x_next_true - x_next_nom1

    if return_nominals:
        return E, X_nom_1step
    return E

