import numpy as np
from trajectory import Trajectory
from typing import Optional, Tuple, Callable
from models.continuous_model import ContinuousModel
from .signal import Signal

Vec = np.ndarray

def step_reference(
    x0: np.ndarray,
    N: int,
    dt: float,
    *,
    v_ref: Optional[float] = None,
    gamma_ref: Optional[float] = None,
    climb_rate: Optional[float] = None,
    nu: int = 2,
) -> Trajectory:
    """
    Simple constant (v,gamma) reference; altitude climbs linearly.
    If climb_rate is not given, enforce kinematic consistency:
      climb_rate = v_ref * sin(gamma_ref).
    """
    v0, ga0, h0 = map(float, x0)
    if v_ref is None:   v_ref = v0
    if gamma_ref is None: gamma_ref = ga0
    if climb_rate is None:
        climb_rate = v_ref * np.sin(gamma_ref)

    t = np.arange(N+1, dtype=float) * dt
    X = np.zeros((N+1, 3), dtype=float)
    U = np.zeros((N, nu), dtype=float)

    h = h0
    for k in range(N):
        X[k] = [v_ref, gamma_ref, h]
        h += climb_rate * dt
    X[N] = [v_ref, gamma_ref, h]

    return Trajectory(t=t, X=X, U=U)


# --- Helpers (same logic as before) ---

def _dgamma_ct(f_ct, x: Vec, alpha: float, thr: float, p: dict) -> float:
    return float(f_ct(x, np.array([alpha, thr], float), p)[1])

def _dv_ct(f_ct, x: Vec, alpha: float, thr: float, p: dict) -> float:
    return float(f_ct(x, np.array([alpha, thr], float), p)[0])

def _find_alpha_for_gamma_hold(
    f_ct, x: Vec, thr: float, p: dict,
    alpha_min: float, alpha_max: float, max_it: int = 40, tol: float = 1e-8
) -> float:
    aL, aU = float(alpha_min), float(alpha_max)
    gL = _dgamma_ct(f_ct, x, aL, thr, p)
    gU = _dgamma_ct(f_ct, x, aU, thr, p)
    if gL * gU < 0.0:
        for _ in range(max_it):
            am = 0.5 * (aL + aU)
            gm = _dgamma_ct(f_ct, x, am, thr, p)
            if abs(gm) < tol: return am
            if gL * gm <= 0.0:
                aU, gU = am, gm
            else:
                aL, gL = am, gm
        return 0.5 * (aL + aU)
    return aL if abs(gL) <= abs(gU) else aU

def _dv_max_with_gamma_hold(
    f_ct, x: Vec, p: dict,
    alpha_min: float, alpha_max: float, thr_max: float
) -> Tuple[float, float]:
    a_star = _find_alpha_for_gamma_hold(f_ct, x, thr=thr_max, p=p,
                                        alpha_min=alpha_min, alpha_max=alpha_max)
    dv = _dv_ct(f_ct, x, a_star, thr_max, p)
    return dv, a_star

def _thr_to_hold_speed_linear_in_thr(
    f_ct, x: Vec, alpha: float, p: dict,
    thr_min: float, thr_max: float
) -> float:
    dv0 = _dv_ct(f_ct, x, alpha, 0.0, p)
    dv1 = _dv_ct(f_ct, x, alpha, 1.0, p)
    a = (dv1 - dv0)
    thr = 0.0 if abs(a) < 1e-12 else -dv0 / a
    return float(np.clip(thr, thr_min, thr_max))

def _dgamma_max_with_speed_hold(
    f_ct, x: Vec, p: dict,
    alpha_min: float, alpha_max: float,
    thr_min: float, thr_max: float,
    n_alpha_grid: int = 25
) -> Tuple[float, float, float]:
    alphas = np.linspace(alpha_min, alpha_max, n_alpha_grid)
    best = (-np.inf, alphas[0], None)
    for a in alphas:
        thr = _thr_to_hold_speed_linear_in_thr(f_ct, x, a, p, thr_min, thr_max)
        dga = _dgamma_ct(f_ct, x, a, thr, p)
        if dga > best[0]:
            best = (dga, a, thr)
    return best  # (dgamma_max, alpha*, thr*)


def piecewise_reference(
    x0: Vec,
    dt: float,
    f_ct: Callable[[Vec, Vec, dict], Vec],
    p_nom: dict,
    *,
    gamma0: float,             # fixed gamma during Segment A
    v1_target: float,          # end-of-A speed
    gamma2_target: float,      # end-of-B gamma
    # Optional durations (will be stretched to feasibility if too short)
    T_A: Optional[float] = None,
    T_B: Optional[float] = None,
    # Safety margins (0<margin≤1) applied to capability
    max_dv_margin: float = 0.6,        # replaces eta_v
    max_gamma_margin: float = 0.6,     # replaces eta_g
    # Bounds/calc settings
    alpha_min_deg: float = -6.0,
    alpha_max_deg: float = +8.0,
    thr_min: float = 0.15,
    thr_max: float = 1.00,
    n_alpha_grid: int = 25,
    min_T_A: float = 1.0,
    min_T_B: float = 1.0,
) -> Trajectory:
    """
    Two segments:
      A) Accelerate from v0 to v1_target at fixed γ=γ0.
         dv_max computed with dγ=0; we use a_v = max_dv_margin * dv_max.
      B) Increase γ from γ0 to γ2_target at fixed v=v1.
         dγ_max computed with dv=0; we use dγ* = max_gamma_margin * dγ_max.

    If T_A / T_B provided and < feasible minima, they are auto-stretched.
    Returns a Trajectory with U=zeros
    """
    assert 0.0 < max_dv_margin <= 1.0 and 0.0 < max_gamma_margin <= 1.0

    alpha_min = np.deg2rad(alpha_min_deg)
    alpha_max = np.deg2rad(alpha_max_deg)

    v0, ga0, h0 = map(float, x0)
    if v1_target <= v0:
        raise ValueError("v1_target must exceed initial v0.")
    if gamma2_target <= gamma0:
        raise ValueError("gamma2_target must exceed gamma0.")
    ga0 = gamma0  # enforce constant-γ during Segment A

    # --- Segment A capability & duration ---
    xA = np.array([v0, gamma0, h0], float)
    dv_max, _ = _dv_max_with_gamma_hold(f_ct, xA, p_nom, alpha_min, alpha_max, thr_max)
    if dv_max <= 1e-9:
        raise ValueError("Segment A: dv_max ≤ 0; cannot accelerate at this gamma.")
    a_v = max_dv_margin * dv_max
    TA_min = (v1_target - v0) / max(a_v, 1e-12)
    TA = max(min_T_A, TA_min if T_A is None else max(T_A, TA_min))
    NA = int(np.ceil(TA / dt))

    # Build Segment A states
    t_list = [0.0]
    X_list = [np.array([v0, gamma0, h0], float)]
    for k in range(NA):
        t = (k+1) * dt
        v = v0 + (v1_target - v0) * min(t/TA, 1.0)  # linear-in-time ramp
        ga = gamma0
        h  = X_list[-1][2] + X_list[-1][0] * np.sin(ga) * dt
        X_list.append(np.array([v, ga, h], float))
        t_list.append(t)

    v1 = X_list[-1][0]
    h1 = X_list[-1][2]

    # --- Segment B capability & duration ---
    xB = np.array([v1, gamma0, h1], float)
    dga_max, _, _ = _dgamma_max_with_speed_hold(
        f_ct, xB, p_nom, alpha_min, alpha_max, thr_min, thr_max, n_alpha_grid=n_alpha_grid
    )
    if dga_max <= 1e-9:
        raise ValueError("Segment B: dgamma_max ≤ 0; cannot increase gamma at this speed.")

    dga_star = max_gamma_margin * dga_max
    TB_min = (gamma2_target - gamma0) / max(dga_star, 1e-12)
    TB = max(min_T_B, TB_min if T_B is None else max(T_B, TB_min))
    NB = int(np.ceil(TB / dt))

    for k in range(NB):
        t = t_list[-1] + dt
        tau = min((k+1) * dt, TB)
        ga = gamma0 + (gamma2_target - gamma0) * (tau / TB)
        v = v1
        h = X_list[-1][2] + v * np.sin(X_list[-1][1]) * dt
        X_list.append(np.array([v, ga, h], float))
        t_list.append(t)

    # Assemble
    t_arr = np.asarray(t_list, float)
    X_arr = np.vstack(X_list)
    U_arr = np.zeros((X_arr.shape[0]-1, 2), float)  # fill later (trim/feasibility)
    return Trajectory(t=t_arr, X=X_arr, U=U_arr)

def level_flight_accelerate_reference(
    x0,
    dt: float,
    f_ct: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
    p_nom: dict,
    *,
    v_target: float = 300.0,
    gamma_ref: Optional[float] = 0.0,
    max_dv_margin: float = 0.6,
    alpha_min_deg: float = -6.0,
    alpha_max_deg: float = +8.0,
    thr_max: float = 1.0,
    min_T: float = 1.0,
) -> "Trajectory":
    """
    Single-segment accelerate reference:
      - Start from x0 = [v0, gamma0, h0]
      - Accelerate v: v0 -> v_target
      - Keep gamma ≈ gamma_ref (default 0 rad, level)
      - h updated by dh = v sin(gamma) dt (so ~constant if gamma_ref≈0)
      - U is left zero; to be filled later by trim / feasible-control routines.

    The segment duration T is chosen from the nominal max dv capability:
      dv_max = max dv/dt at (x0) with gamma-hold, thr=thr_max, alpha in [min,max]
      a_des  = max_dv_margin * dv_max
      T      = (v_target - v0) / a_des

    Args:
        x0: initial state [v, gamma, h]
        dt: time step
        f_ct, p_nom: nominal continuous-time model and params
        v_target: final speed
        gamma_ref: desired (fixed) gamma during the segment
        max_dv_margin: safety factor in (0,1]; use <1 to stay inside capability
        alpha_min_deg, alpha_max_deg: AoA bounds used for dv_max calc
        thr_max: throttle used in dv_max calc (typically 1.0)
        min_T: minimum duration floor

    Returns:
        Trajectory(t, X, U)
    """
    x0 = np.asarray(x0, float)
    v0, gamma0, h0 = x0

    if v_target <= v0:
        raise ValueError("v_target must be greater than initial v0 for acceleration segment.")

    if gamma_ref is None:
        gamma_ref = float(gamma0)
    else:
        gamma_ref = float(gamma_ref)

    if not (0.0 < max_dv_margin <= 1.0):
        raise ValueError("max_dv_margin must be in (0,1].")

    alpha_min = np.deg2rad(alpha_min_deg)
    alpha_max = np.deg2rad(alpha_max_deg)

    # --- compute max dv capability at start state with gamma hold ---
    x_cap = np.array([v0, gamma_ref, h0], float)
    dv_max, a_star = _dv_max_with_gamma_hold(
        f_ct, x_cap, p_nom,
        alpha_min=alpha_min, alpha_max=alpha_max,
        thr_max=thr_max
    )

    if dv_max <= 1e-9:
        raise ValueError("accelerate_reference: dv_max <= 0; cannot accelerate at this condition.")

    a_des = max_dv_margin * dv_max
    T_raw = (v_target - v0) / max(a_des, 1e-12)
    T = max(min_T, T_raw)

    N = int(np.ceil(T / dt))
    T = N * dt  # snap to grid

    # --- build trajectory ---
    t = np.linspace(0.0, T, N+1)
    X = np.zeros((N+1, 3), float)
    X[0] = np.array([v0, gamma_ref, h0], float)

    for k in range(N):
        tau = (k+1) * dt / T
        if tau > 1.0:
            tau = 1.0
        v = v0 + (v_target - v0) * tau
        ga = gamma_ref
        # simple kinematic integration for h using previous state
        h = X[k, 2] + X[k, 0] * np.sin(X[k, 1]) * dt
        X[k+1] = np.array([v, ga, h], float)

    U = np.zeros((N, 2), float)  # to be filled by trim / controller later

    # assuming you have a Trajectory dataclass or similar:
    return Trajectory(t=t, X=X, U=U)

def dvmax_vs_speed(f_ct, p, h=2000.0, gamma_ref=0.0,
                   v_min=140.0, v_max=320.0, nv=61,
                   alpha_min_deg=-6.0, alpha_max_deg=8.0, thr_max=1.0):
    v_grid = np.linspace(v_min, v_max, nv)
    dv_max = np.zeros_like(v_grid)
    alpha_star = np.zeros_like(v_grid)
    alpha_min = np.deg2rad(alpha_min_deg)
    alpha_max = np.deg2rad(alpha_max_deg)
    for i, v in enumerate(v_grid):
        x = np.array([v, gamma_ref, h], float)
        # your existing helper: find alpha* that makes dgamma≈0, then compute dv at thr_max
        a_star = _find_alpha_for_gamma_hold(f_ct, x, thr=thr_max, p=p,
                                            alpha_min=alpha_min, alpha_max=alpha_max)
        dv = _dv_ct(f_ct, x, a_star, thr_max, p)
        dv_max[i] = dv
        alpha_star[i] = a_star
    return v_grid, dv_max, alpha_star

def accel_time_from_dvmax(v_grid, dvmax_grid, v0, v1, eta=0.6, eps=1e-3):
    # assumes v_grid ascending, dvmax on same grid
    import numpy as np
    v0, v1 = float(v0), float(v1)
    if v1 <= v0: return 0.0
    # interpolate dvmax(v)
    dv = np.linspace(v0, v1, 800)
    dvmax = np.interp(dv, v_grid, dvmax_grid)
    a_des = eta * np.maximum(dvmax, eps)  # avoid divide-by-zero
    dt = np.trapz(1.0 / a_des, dv)
    return float(dt)

# --- capability helpers (use your existing ones if available) ---

def _dv_ct(f_ct, x, alpha, thr, p):
    # returns dv/dt from your nominal CT model at (x,alpha,thr)
    dv, dga, dh = f_ct(x, np.array([alpha, thr], float), p)
    return float(dv)

def _find_alpha_for_gamma_hold(f_ct, x, thr, p, alpha_min, alpha_max, tol=1e-8, iters=30):
    # Newton/bisection hybrid to solve dgamma(x,alpha,thr)=0
    a_lo, a_hi = float(alpha_min), float(alpha_max)
    a = 0.5*(a_lo + a_hi)
    for _ in range(iters):
        dv, dga, dh = f_ct(x, np.array([a, thr], float), p)
        if abs(dga) < tol: break
        # central slope wrt alpha
        h = 1e-5
        _, dga_p, _ = f_ct(x, np.array([a+h, thr], float), p)
        _, dga_m, _ = f_ct(x, np.array([a-h, thr], float), p)
        J = (dga_p - dga_m)/(2*h)
        if J == 0:
            # fall back to bisection
            if dga > 0: a_hi = a
            else: a_lo = a
            a = 0.5*(a_lo + a_hi)
        else:
            a_new = a - dga/J
            a = float(np.clip(a_new, a_lo, a_hi))
    return a

def _dv_max_with_gamma_hold(f_ct, x, p, alpha_min, alpha_max, thr_max) -> Tuple[float, float]:
    a_star = _find_alpha_for_gamma_hold(f_ct, x, thr=thr_max, p=p,
                                        alpha_min=alpha_min, alpha_max=alpha_max)

    dv = _dv_ct(f_ct, x, a_star, thr_max, p)
    return dv, a_star

def _dv_min_with_gamma_hold(f_ct, x, p, alpha_min, alpha_max, thr_min) -> Tuple[float, float]:
    # “most negative” dv at gamma-hold using minimum throttle → decel capability
    a_star = _find_alpha_for_gamma_hold(f_ct, x, thr=thr_min, p=p,
                                        alpha_min=alpha_min, alpha_max=alpha_max)
    dv = _dv_ct(f_ct, x, a_star, thr_min, p)
    return dv, a_star  # dv should be ≤ 0

# --- the reference builder you asked for ---

def accelerate_with_push_over_reference(
    x0: np.ndarray,
    dt: float,
    nom_sys: ContinuousModel,
    *,
    v_switch: float = 270.0,
    v_peak: float = 320.0,
    v_final: float = 300.0,
    gamma_push_deg: float = -1.5,
    gamma_level_deg: float = 0.0,
    max_dv_margin: float = 0.6,
    max_ddv_margin: float = 0.8,
    decel_margin: float = 0.8,
    alpha_min_deg: float = -6.0,
    alpha_max_deg: float = +8.0,
    thr_min: float = 0.15,
    thr_max: float = 1.00,
    min_T: float = 1.0,
) -> Trajectory:

    """
    Segment A: level (gamma=0)  v0 -> v_switch using a_des = max_dv_margin * dv_max(level)
    Segment B: push-over (gamma=gamma_push) v_switch -> v_peak using a_des = max_ddv_margin * dv_max(push)
    Segment C: level (gamma=0)  v_peak -> v_final using decel a_des = decel_margin * |dv_min(level)|

    Returns a Trajectory with U=zeros (to be filled by trim/feasibility later).
    """
    v0, ga0, h0 = map(float, x0)
    g_level  = np.deg2rad(gamma_level_deg)
    g_push   = np.deg2rad(gamma_push_deg)
    a_min    = np.deg2rad(alpha_min_deg)
    a_max    = np.deg2rad(alpha_max_deg)

    def choose_gamma_push_for_positive_dv(
        f_ct, x_start, p_nom,
        gamma_deg_candidates=(-3.0, -2.5, -2.0, -1.5, -1.0, -0.5),
        alpha_min_deg=-6.0, alpha_max_deg=8.0, thr_max=1.0
    ):
        a_min = np.deg2rad(alpha_min_deg)
        a_max = np.deg2rad(alpha_max_deg)
        best = None
        for gdeg in gamma_deg_candidates:
            g = np.deg2rad(gdeg)
            x = np.array([x_start[0], g, x_start[2]], float)  # hold h from start of segment
            dv_max, _ = _dv_max_with_gamma_hold(nom_sys.f_ct, x, nom_sys.params, a_min, a_max, thr_max)
            if dv_max > 0.0:
                best = (g, dv_max)
                break
        return best

    def _build_segment_linear_v(v_start, v_end, gamma_const, T):
        N = int(np.ceil(max(T, min_T) / dt))
        T = N * dt
        t = np.linspace(0.0, T, N+1)
        X = np.zeros((N+1, 3))
        X[0] = np.array([v_start, gamma_const, X_list[-1][2] if X_list else h0])
        for k in range(N):
            tau = (k+1)/N
            v   = v_start + (v_end - v_start) * tau
            ga  = gamma_const
            h   = X[k,2] + X[k,0] * np.sin(X[k,1]) * dt
            X[k+1] = np.array([v, ga, h])
        return t[1:], X[1:]

    # containers
    t_list = [0.0]
    X_list = [np.array([v0, g_level, h0], float)]

    # ---- Segment A: level accel to v_switch ----
    xA = np.array([v0, g_level, h0], float)
    dvA_max, _ = _dv_max_with_gamma_hold( nom_sys.f_ct, xA, nom_sys.params, a_min, a_max, thr_max )
    if dvA_max <= 1e-9: raise ValueError("No level accel capability at start.")
    aA = max_dv_margin * dvA_max
    TA = (v_switch - v0) / max(aA, 1e-12)
    tA, XA = _build_segment_linear_v(v0, v_switch, g_level, TA)
    t_list += (t_list[-1] + tA).tolist()
    X_list += XA.tolist()

    # ---- Segment B: push-over accel to v_peak ----
    xB0 = X_list[-1].copy(); 
    res = choose_gamma_push_for_positive_dv(nom_sys.f_ct, xB0, nom_sys.params,
                                        gamma_deg_candidates=(-3.0,-2.5,-2.0,-1.5,-1.0),
                                        alpha_min_deg=-8.0, alpha_max_deg=10.0, thr_max=1.0)
    if res is None:
        raise ValueError("Even at -3 deg, no positive dv capability. Lower altitude or v_peak.")
    gamma_push, dvB_max = res
    xB = X_list[-1].copy(); 
    xB[1] = gamma_push
    dvB_max, _ = _dv_max_with_gamma_hold(nom_sys.f_ct, xB, nom_sys.params, a_min, a_max, thr_max)
    if dvB_max <= 1e-9: raise ValueError("No push-over accel capability.")
    aB = max_ddv_margin * dvB_max
    TB = (v_peak - v_switch) / max(aB, 1e-12)
    tB, XB = _build_segment_linear_v(v_switch, v_peak, g_push, TB)
    t_list += (t_list[-1] + tB).tolist()
    X_list += XB.tolist()

    # ---- Segment C: level-out & decel to v_final (low throttle capability) ----
    xC = X_list[-1].copy(); xC[1] = g_level
    dvC_min, _ = _dv_min_with_gamma_hold(nom_sys.f_ct, xC, nom_sys.params, a_min, a_max, thr_min)  # ≤ 0
    if dvC_min >= -1e-9:
        # if we cannot decelerate meaningfully, just hold v_final == v_peak
        v_final = v_peak
        TC = min_T
    else:
        aC = decel_margin * abs(dvC_min)
        TC = (v_peak - v_final) / max(aC, 1e-12)
    tC, XC = _build_segment_linear_v(v_peak, v_final, g_level, TC)
    t_list += (t_list[-1] + tC).tolist()
    X_list += XC.tolist()

    # Assemble Trajectory
    t_arr = np.asarray(t_list, float)
    X_arr = np.vstack(X_list)
    U_arr = np.zeros((X_arr.shape[0]-1, 2), float)
    X_sig = Signal(t=t_arr, Y=X_arr, labels=["v", "gamma", "h"]) 
    U_sig = Signal(t=t_arr[:-1], Y=U_arr, labels=["alpha", "throttle"])
    return Trajectory(X=X_sig, U=U_sig)

