
import numpy as np

def estimate_L_nom(f_nom_dt, X_samp, U_samp, p_nom, eps=1e-5):
    """Crude Lipschitz estimate for x->f_nom_dt(x,u) by finite differences over samples."""
    Ls = []
    for x in X_samp:
        for u in U_samp:
            fx = f_nom_dt(x, u, p_nom)
            for i in range(len(x)):
                dx = np.zeros_like(x); dx[i]=eps
                fx2 = f_nom_dt(x+dx, u, p_nom)
                Ls.append(np.linalg.norm(fx2-fx, ord=2)/eps)
    return float(np.max(Ls)) if Ls else 0.0

def estimate_L_residual(W, phi, X_samp, U_samp, eps=1e-5):
    """Crude Lipschitz estimate for x->W^T phi(x,u)."""
    Ls = []
    for x in X_samp:
        for u in U_samp:
            fx = phi(x,u) @ W
            for i in range(len(x)):
                dx = np.zeros_like(x); dx[i]=eps
                fx2 = phi(x+dx,u) @ W
                Ls.append(np.linalg.norm(fx2-fx, ord=2)/eps)
    return float(np.max(Ls)) if Ls else 0.0

def k_step_prediction_bound(delta0_norm, eps1, L, k):
    if abs(L-1.0) < 1e-9:
        return delta0_norm + k*eps1
    else:
        return (L**k)*delta0_norm + eps1*(L**k - 1.0)/(L-1.0)

def finite_horizon_cost_gap_bound(delta_seq_norms, L_ell, L_Vf):
    N = len(delta_seq_norms)-1
    return L_ell*sum(delta_seq_norms[:N]) + L_Vf*delta_seq_norms[-1]
