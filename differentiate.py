
import numpy as np
from dynamics import isa_troposphere

def numeric_foward_diff(f, x, u, p, eps=1e-5):
    x = np.asarray(x).copy()
    u = np.asarray(u).copy()
    n = x.size
    m = u.size
    f0 = f(x, u, p)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for ii in range(n):
        dx = np.zeros_like(x)
        dx[ii] = eps
        A[:, ii] = (f(x+dx, u, p) - f0) / eps
    for jj in range(m):
        du = np.zeros_like(u)
        du[jj] = eps
        B[:, jj] = (f(x, u+du, p) - f0) / eps
    return A, B

def numeric_central_diff(f, x, u, p, hx=None, hu=None):
    x = np.array(x, dtype=float).copy()
    u = np.array(u, dtype=float).copy()
    n, m = x.size, u.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # scale-aware steps (good defaults)
    if hx is None: hx = 1e-6 * (1.0 + np.abs(x))
    if hu is None: hu = 1e-6 * (1.0 + np.abs(u))

    for ii in range(n):
        dx = np.zeros_like(x); dx[ii] = hx[ii]
        fp = f(x + dx, u, p)
        fm = f(x - dx, u, p)
        A[:, ii] = (fp - fm) / (2.0 * hx[ii])

    for jj in range(m):
        du = np.zeros_like(u); du[jj] = hu[jj]
        fp = f(x, u + du, p)
        fm = f(x, u - du, p)
        B[:, jj] = (fp - fm) / (2.0 * hu[jj])

    return A, B


def analytic_jacobian_ct(x, u, p):
    v, ga, h = x
    al, thr   = u
    S, m, g   = p['S'], p['m'], p['g']
    CL0       = p.get('CL0_nom', 0.15)
    CLa       = p.get('CL_alpha_nom', 4.0)
    CD0       = p.get('CD0_nom', 0.03)
    k_ind     = p.get('k_nom', 0.05)
    T0        = p.get('T0_nom', 20000.0)
    T1        = p.get('T1_nom', -2000.0)

    # ISA
    rho, a = isa_troposphere(h)
    # d rho / d h = rho * (L R - g) / (R T)
    # recover T_air from a: a^2 = gamma R T  ->  T = a^2/(gamma R)
    gamma_air = 1.4
    Rgas = 287.05
    L = 0.0065
    T_air = a*a/(gamma_air*Rgas)
    drho_dh = rho * (L*Rgas - g) / (Rgas * T_air)

    # M and derivatives
    M = max(v/a, 1e-9)
    dM_dv = 1.0 / a
    da_dh = - (gamma_air*Rgas*L) / (2.0*a)          # da/dh
    dM_dh = -v/(a*a) * da_dh                        # = v*(gamma R L)/(2 a^3)

    # Aero
    q  = 0.5 * rho * v*v
    CL = CL0 + CLa * al
    CD = CD0 + k_ind * CL*CL
    Lf = q * S * CL
    Df = q * S * CD

    # Thrust
    T  = thr * (T0 + T1*M)
    dT_dthr = (T0 + T1*M)
    dT_dv   = thr * T1 * dM_dv
    dT_dh   = thr * T1 * dM_dh

    # Small-angle trig and derivatives
    sin_al, cos_al = al, (1.0 - 0.5*al*al)
    sin_ga, cos_ga = ga, (1.0 - 0.5*ga*ga)
    dcosal_dal = -al
    dsinal_dal =  1.0
    dsinga_dga =  1.0
    dcosga_dga = -ga

    # Helpful partials for L, D
    # q_v = dq/dv, q_h = dq/dh
    q_v = rho * v
    q_h = 0.5 * v*v * drho_dh
    # dCL/dalpha = CLa ; no v,h dependence in nominal
    dD_dv   = q_v * S * CD                 # CD depends only on alpha
    dD_dal  = q * S * (2.0 * k_ind * CL * CLa)
    dD_dh   = q_h * S * CD
    dL_dv   = q_v * S * CL
    dL_dal  = q * S * CLa
    dL_dh   = q_h * S * CL

    # ===== A_c = df/dx =====
    # dv/dv, dv/dga, dv/dh
    dvdv  = ( dT_dv * cos_al - dD_dv ) / m
    dvdga = ( - m * g * dsinga_dga ) / m          # = -g
    dvdh  = ( dT_dh * cos_al - dD_dh ) / m

    # dgamma/dx uses N = (L + T*sin(alpha) - m g cos(gamma)) / (m v)
    N      = (Lf + T*sin_al - m*g*cos_ga)
    inv_mv = 1.0 / (m * max(v, 1e-6))
    dN_dv  = dL_dv + dT_dv * sin_al               # cos(gamma) term has no v
    dN_dga = ( -m*g * dcosga_dga )                # = m*g*ga
    dN_dh  = dL_dh + dT_dh * sin_al

    dgdv  = - N * (1.0 / (m * max(v,1e-6)**2)) + inv_mv * dN_dv
    dgdga = inv_mv * dN_dga                       # = (g*ga)/v
    dgdh  = inv_mv * dN_dh

    # dh/dx
    dhdv  = sin_ga                                 # ≈ ga
    dhdga = v
    dhdh  = 0.0

    A_c = np.array([
        [dvdv,  dvdga, dvdh],
        [dgdv,  dgdga, dgdh],
        [dhdv,  dhdga, dhdh],
    ])

    # ===== B_c = df/du =====
    # dv/du
    dvdal  = ( T * dcosal_dal - dD_dal ) / m      # = ( -T*al - dD/dal )/m
    dvdthr = ( dT_dthr * cos_al ) / m

    # dgamma/du
    dN_dal  = dL_dal + T*dsinal_dal               # + dT/dal * sin(al) but T has no alpha in nominal
    dN_dthr = dT_dthr * sin_al
    dgal    = inv_mv * dN_dal
    dgthr   = inv_mv * dN_dthr

    # dh/du = 0
    B_c = np.array([
        [dvdal,  dvdthr],
        [dgal,   dgthr],
        [0.0,    0.0   ],
    ])
    return A_c, B_c

def analytic_discrete(x, u, p, dt):
    # For Euler-discretized x_{k+1} = x_k + dt * f_ct(x,u):
    A_c, B_c = analytic_jacobian_ct(x, u, p)
    n = len(x)
    A_d = np.eye(n) + dt * A_c
    B_d = dt * B_c
    return A_d, B_d
