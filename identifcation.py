
import numpy as np
from models import isa_troposphere

def poly_features(xu, deg=2):
    """Polynomial features up to degree `deg` on concatenated [x;u]."""
    xu = np.asarray(xu)
    d = xu.size
    feats = [1.0]
    feats.extend(xu.tolist())
    if deg >= 2:
        for i in range(d):
            for j in range(i, d):
                feats.append(xu[i]*xu[j])
    if deg >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(xu[i]*xu[j]*xu[k])
    return np.array(feats, dtype=float)

def batch_features(X, U, deg=2):
    N = X.shape[0]
    Phi = np.vstack([poly_features(np.hstack([X[k], U[k]]), deg) for k in range(N)])
    return Phi

def ridge_regression(Phi, Y, lam=1e-4):
    """Solve W in min ||Phi W - Y||^2 + lam ||W||^2, returns W."""
    # Phi: N x d, Y: N x n_out, W: d x n_out
    d = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(d)
    B = Phi.T @ Y
    W = np.linalg.solve(A, B)
    return W

def fit_residual_model(X, U, E_next, deg=2, lam=1e-4):
    """Fit residual e(x,u) ~ W^T phi(x,u).
    Returns (W, featurizer) where featurizer is a callable (x,u)->phi."""
    Phi = batch_features(X, U, deg=deg)
    W = ridge_regression(Phi, E_next, lam=lam)
    def phi(x, u):
        return poly_features(np.hstack([x, u]), deg=deg)
    return W, phi

def predict_residual(W, phi, x, u):
    return phi(x, u) @ W  # (d,) @ (d x n) -> (n,)

def _rbf_ard(Z1, Z2, ls):
    # Z1: (N1,d), Z2: (N2,d), ls: (d,) positive
    D = (Z1[:, None, :] - Z2[None, :, :]) / ls
    return np.exp(-0.5 * np.sum(D*D, axis=2))

def _perdim_median_ls(Z):
    # ARD lengthscales via per-dimension median |diff|
    N, d = Z.shape
    ls = np.empty(d)
    for i in range(d):
        zi = Z[:, i][:, None]
        md = np.median(np.abs(zi - zi.T))
        ls[i] = md if md > 1e-8 else 1.0
    return ls

def feature_stack(X, U, p=None, keep_2d=True):
    """
    Build physics-informed features from states X and controls U.

    X: (N, dx) or (dx,) with columns [v, gamma, h]
    U: (N, du) or (du,) with columns [alpha, thr]
    p: unused placeholder (kept for API compatibility)
    keep_2d: if False and X,U are 1D, return a 1D feature vector

    Returns:
        Z: (N, F) if batched, otherwise (1, F) unless keep_2d=False -> (F,)
    """
    X = np.asarray(X)
    U = np.asarray(U)

    # Detect single-sample inputs and upgrade to 2D
    was_1d = (X.ndim == 1) and (U.ndim == 1)
    if X.ndim == 1:
        X = X[None, :]
    if U.ndim == 1:
        U = U[None, :]

    # Basic shape checks (optional but helpful)
    if X.shape[0] != U.shape[0]:
        raise ValueError(f"Batch size mismatch: X has {X.shape[0]} rows, U has {U.shape[0]} rows.")
    if X.shape[1] < 3:
        raise ValueError(f"X must have at least 3 columns [v, gamma, h]; got {X.shape[1]}.")
    if U.shape[1] < 2:
        raise ValueError(f"U must have at least 2 columns [alpha, thr]; got {U.shape[1]}.")

    # Unpack
    v     = X[:, 0]
    gamma = X[:, 1]
    h     = X[:, 2]
    alpha = U[:, 0]
    thr   = U[:, 1]

    # ISA: prefer vectorized call if available; fallback to np.vectorize
    try:
        rho, a = isa_troposphere(h)            # ideally returns arrays
    except Exception:
        rho, a = np.vectorize(isa_troposphere)(h)

    # Aerodynamics & convenience terms
    q = 0.5 * rho * v**2
    M = np.maximum(v / a, 1e-6)               # avoid divide-by-zero

    # Angles
    s_a, c_a = np.sin(alpha), np.cos(alpha)
    s_g, c_g = np.sin(gamma), np.cos(gamma)

    # Feature stack
    Z = np.column_stack([
        X, U, M, q, s_a, c_a, s_g, c_g,
        alpha**2, M**2, np.sqrt(q)            # light nonlinears
    ])

    # If caller passed 1D inputs, optionally squeeze back to 1D
    if was_1d and not keep_2d:
        Z = Z[0]

    return Z


class RKHSResidual:
    def __init__(self, Ztr, alpha, ls, mu, sig, Ymean, diag_info):
        self.Ztr = Ztr          # normalized train inputs
        self.alpha = alpha      # (N, nx)
        self.ls = ls            # (d,)
        self.mu = mu            # (d,)
        self.sig = sig          # (d,)
        self.Ymean = Ymean      # (1, nx)
        self.diag_info = diag_info  # dict: {"cond":..., "eigmin":..., "eigmax":...}

    def predict(self, X, U,p_nom):
        Zq = feature_stack(X, U, p_nom)
        Zq = (Zq - self.mu) / self.sig
        Kq = _rbf_ard(Zq, self.Ztr, self.ls)
        return Kq @ self.alpha + self.Ymean  # (Nq, nx)

    def predict_one(self, x, u, p_nom):
        # 1) Features: ensure (1, F), not (1,1,F)
        Zq = feature_stack(x, u, p_nom, keep_2d=True)   # (1, F)

        # 2) Standardize
        Zq = (Zq - self.mu) / self.sig                  # (1, F)

        # 3) Shape sanity checks (helpful when debugging)
        F = Zq.shape[1]
        if self.Ztr.shape[1] != F:
            raise ValueError(f"Feature dim mismatch: Ztr has {self.Ztr.shape[1]}, Zq has {F}.")
        Ntr = self.Ztr.shape[0]
        if self.alpha.shape[0] != Ntr:
            raise ValueError(f"Alpha rows {self.alpha.shape[0]} != Ntr {Ntr} (dual weights expected).")

        # 4) Kernel block and prediction
        Kq = _rbf_ard(Zq, self.Ztr, self.ls)            # (1, Ntr)
        if Kq.shape[1] != Ntr:
            raise ValueError(f"Kq has {Kq.shape[1]} cols; expected {Ntr}. Check Zq shape (should be 2D).")

        y = Kq @ self.alpha                              # (1, Dy)

        # 5) Add mean and return 1D if Dy>1
        return (y + self.Ymean)[0]

def fit_residual_rkhs(Xtr, Utr, Ytr, p_nom, lam=1e-3, ls=None):
    """
    Kernel ridge on z=[x;u] with ARD RBF. Returns RKHSResidual model.
    """
    Xtr = np.asarray(Xtr); Utr = np.asarray(Utr); Ytr = np.asarray(Ytr)
    Ztr_raw = feature_stack(Xtr, Utr, p_nom)               # (N,d)
    mu = Ztr_raw.mean(axis=0); sig = Ztr_raw.std(axis=0) + 1e-8
    Ztr = (Ztr_raw - mu) / sig
    if ls is None:
        ls = _perdim_median_ls(Ztr)
    K = _rbf_ard(Ztr, Ztr, ls)                    # (N,N)
    Ymean = Ytr.mean(axis=0, keepdims=True)       # intercept via mean-centering
    Y = Ytr - Ymean
    Kreg = K + lam*np.eye(K.shape[0])
    alpha = np.linalg.solve(Kreg, Y)

    try:
        eigvals = np.linalg.eigvalsh(Kreg)
        eigmin = float(eigvals.min())
        eigmax = float(eigvals.max())
        cond   = float(eigmax / max(eigmin, 1e-18))
    except np.linalg.LinAlgError:
        eigmin = np.nan; eigmax = np.nan; cond = np.nan

    diag_info = {"cond": cond, "eigmin": eigmin, "eigmax": eigmax}
    return RKHSResidual(Ztr, alpha, ls, mu, sig, Ymean,diag_info)
