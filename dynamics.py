
import numpy as np
from core.structs import DynEval
# ------------------------------
# Aerodynamics & point-mass model
# ------------------------------

import numpy as np

def isa_troposphere(h):
    """
    Simple ISA up to ~11 km.
    Returns: rho [kg/m^3], a [m/s] (speed of sound)
    """
    # constants
    T0 = 288.15      # K
    p0 = 101325.0    # Pa
    L  = 0.0065      # K/m (lapse rate)
    R  = 287.05      # J/(kg K)
    g0 = 9.80665     # m/s^2
    gamma_air = 1.4

    h_clamped = np.clip(h, 0.0, 11000.0)
    T = T0 - L * h_clamped
    p = p0 * (T/T0) ** (g0/(R*L))
    rho = p / (R * T)
    a = np.sqrt(gamma_air * R * T)
    return rho, a

def sincos_small_safe(x):
    # Stable small-angle handling (used in nominal model)
    return x, 1.0 - 0.5*x*x  # sin≈x, cos≈1 - x^2/2



# Nominal evaluator (used by f_nom_ct)
def eval_nominal(x, u, p) -> DynEval:
    v, gamma, h = x
    alpha, thr = u
    thr = float(np.clip(thr, 0.0, 1.0))

    rho, a = isa_troposphere(h)
    q = 0.5 * rho * v**2
    M = max(v / a, 1e-6)

    CL0   = p.get('CL0', 0.15)
    CLa   = p.get('CL', 4.0)
    CD0   = p.get('CD0', 0.02)
    eta_ind = p.get('eta', 0.05)
    S     = p['S']; m = p['m']; g = p['g']

    CL = CL0 + CLa * alpha
    CD = CD0 + eta_ind * CL**2
    L  = q * S * CL
    D  = q * S * CD

    T0 = p.get('T0_nom', 20000.0)
    t1 = p.get('T1_nom', -2000.0)
    T  = thr * (T0 + t1 * M)

    s_a, c_a = sincos_small_safe(alpha)
    s_g, c_g = sincos_small_safe(gamma)

    dv     = ( T * c_a - D - m * g * s_g ) / m
    dgamma = ( L + T * s_a - m * g * c_g ) / (m * max(v, 1e-3))
    dh     = v * s_g

    return DynEval(xdot=np.array([dv, dgamma, dh]), CL=CL, CD=CD, L=L, D=D, T=T,
                   q=q, M=M, rho=rho)

# True evaluator (used by f_true_ct)
def eval_true(x, u, p) -> DynEval:
    """
    'True' model where the ONLY difference from eval_nominal
    is that CD has Mach-dependent CD0(M) and k(M).
    Lift, thrust, trig, etc. are identical to the nominal model.
    """
    v, gamma, h = x
    alpha, thr = u
    thr = float(np.clip(thr, 0.0, 1.0))

    # Atmosphere + Mach
    rho, a = isa_troposphere(h)
    q = 0.5 * rho * v**2
    M = max(v / a, 1e-6)

    S = p['S']; m = p['m']; g = p['g']

    # --- LIFT: same as nominal ---
    CL0_nom = p.get('CL0_nom', 0.15)
     # Mach-dependent lift slope
    CLa0 = p.get('CL_alpha0', 4.5)      # base slope
    a1   = p.get('CLa_a1', 0.35)        # amplitude
    M0   = p.get('CLa_M0', 0.8)         # center Mach
    wM   = p.get('CLa_w', 0.12)         # width

    CLaM = CLa0 * (1.0 + a1 * np.tanh((M - M0) / max(wM, 1e-6)))

    CL = CL0_nom + CLaM * alpha

    # --- DRAG: Mach-dependent CD0(M) and k(M) (this is the ONLY difference) ---
    # CD0(M) = cd0_0 + cd0_1 * tanh((M - M1)/wD)
    CD0_nom = p.get('CD0_nom', 0.02)       # same baseline as nominal
    dCD0    = p.get('dCD0_max', 0.01)     # extra drag at high Mach
    M1      = p.get('CD0_M1', 0.85)
    wD      = p.get('CD0_w', 0.15)
    # high M: CD0_true ~ CD0_nom + dCD0
    CD0M = CD0_nom + 0.5 * dCD0 * (1.0 + np.tanh((M - M1)/max(wD, 1e-6)))

    # k(M) = k0 * (1 + b1 * M)
    k0 = p.get('k0', 0.045)
    b1 = p.get('k_M_slope', 0.15)
    kM = k0 * (1.0 + b1 * M)

    CD = CD0M + kM * CL**2

    L = q * S * CL
    D = q * S * CD

    # --- THRUST: same as nominal ---
    T0_nom = p.get('T0_nom', 20000.0)
    T1_nom = p.get('T1_nom', -2000.0)
    T = thr * (T0_nom + T1_nom * M)

    # --- KINEMATICS/DYNAMICS: same as nominal (small-angle-safe trig) ---
    s_a, c_a = sincos_small_safe(alpha)
    s_g, c_g = sincos_small_safe(gamma)

    dv     = ( T * c_a - D - m * g * s_g ) / m
    dgamma = ( L + T * s_a - m * g * c_g ) / (m * max(v, 1e-3))
    dh     = v * s_g

    return DynEval(xdot=np.array([dv, dgamma, dh]),
                   CL=CL, CD=CD, L=L, D=D, T=T,
                   q=q, M=M, rho=rho)

def f_nom_ct(x, u, p):
    return eval_nominal(x, u, p).xdot

def f_true_ct(x, u, p):
    return eval_true(x, u, p).xdot

