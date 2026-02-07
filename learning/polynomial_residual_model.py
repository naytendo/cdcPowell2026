# learning/polynomial_residual_model.py

import numpy as np
from itertools import combinations_with_replacement
from .residual_model import ResidualModel


class PolynomialResidualModel(ResidualModel):
    """
    Polynomial residual model:
        e_hat = W * phi(x,u)
    where phi(x,u) contains all monomials of x and u up to degree d.
    """

    def __init__(self, degree):
        self.degree = degree
        self.W = None
        self.feature_indices = None   # list of tuples indicating which variables multiply

    def _build_feature_indices(self, dim):
        """
        Build all monomial index combinations up to the given degree.
        dim = nx + nu
        """
        idx = []
        for d in range(self.degree + 1):
            for combo in combinations_with_replacement(range(dim), d):
                idx.append(combo)
        return idx

    def _phi(self, x, u):
        """
        Construct polynomial feature vector phi(x,u).
        """
        z = np.concatenate([x, u])  # shape (nx+nu,)
        feats = []
        for combo in self.feature_indices:
            if len(combo) == 0:
                feats.append(1.0)  # constant term
            else:
                val = 1.0
                for i in combo:
                    val *= z[i]
                feats.append(val)
        return np.array(feats)

    def fit(self, X, U, E):
        """
        Fit W using least squares:
            E ≈ Phi * W
        """
        N, nx = X.shape
        nu = U.shape[1]

        # Build feature index list once
        dim = nx + nu
        self.feature_indices = self._build_feature_indices(dim)

        # Build design matrix Phi
        Phi = np.array([self._phi(x, u) for x, u in zip(X, U)])  # shape (N, n_features)

        # Solve least squares for each output dimension
        # W has shape (n_features, nx)
        self.W, _, _, _ = np.linalg.lstsq(Phi, E, rcond=None)

    def predict(self, x, u):
        """
        Predict residual e_hat = W^T * phi(x,u)
        """
        phi = self._phi(x, u)  # shape (n_features,)
        return phi @ self.W    # shape (nx,)
