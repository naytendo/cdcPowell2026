# learning/rkhs_residual_model.py

import numpy as np
from .residual_model import ResidualModel


def rbf_kernel(Z1, Z2, lengthscale=1.0):
    """
    RBF kernel between two sets of points.
    Z1: (N1, d)
    Z2: (N2, d)
    Returns: (N1, N2)
    """
    Z1 = np.atleast_2d(Z1)
    Z2 = np.atleast_2d(Z2)
    diff = Z1[:, None, :] - Z2[None, :, :]
    sqdist = np.sum(diff**2, axis=-1)
    return np.exp(-0.5 * sqdist / (lengthscale**2))


class RKHSResidualModel(ResidualModel):
    """
    RKHS residual model using kernel ridge regression:
        e_hat(z) = K(z, Z_train) @ Alpha
    where Alpha = (K + lambda I)^(-1) E.
    """

    def __init__(self, kernel=rbf_kernel, lengthscale=1.0, reg=1e-6):
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.reg = reg

        self.Z_train = None   # (N, d)
        self.Alpha = None     # (N, nx)

    def fit(self, X, U, E):
        """
        X: (N, nx)
        U: (N, nu)
        E: (N, nx)
        """
        Z = np.hstack([X, U])          # (N, d)
        self.Z_train = Z

        K = self.kernel(Z, Z, lengthscale=self.lengthscale)  # (N, N)
        N = K.shape[0]
        K_reg = K + self.reg * np.eye(N)

        # Solve (K + λI) Alpha = E  → Alpha: (N, nx)
        self.Alpha = np.linalg.solve(K_reg, E)

    def predict(self, x, u):
        """
        x: (nx,)
        u: (nu,)
        returns e_hat: (nx,)
        """
        if self.Z_train is None or self.Alpha is None:
            raise RuntimeError("RKHSResidualModel must be fit() before predict().")

        z = np.concatenate([x, u])[None, :]  # (1, d)
        k = self.kernel(z, self.Z_train, lengthscale=self.lengthscale)  # (1, N)
        return k @ self.Alpha  # (1, nx) → (nx,)
