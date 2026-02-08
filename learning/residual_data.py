from dataclasses import dataclass
import numpy as np
from core.signal import Signal

@dataclass(frozen=True, slots=True)
class ResidualData:
    """
    Residual data for a trajectory:
      - Xk:  state signal at times t_k
      - Uk:  input signal at times t_k
      - Ek1: one-step prediction error signal at times t_k
    """
    Xk: Signal     # shape (N, nx)
    Uk: Signal     # shape (N, nu)
    Ek1: Signal    # shape (N, nx)

    @property
    def N(self):
        return self.Xk.N

    def as_training_data(self):
        """
        Returns arrays suitable for regression:
            X_data: (N, nx)
            U_data: (N, nu)
            E_data: (N, nx)
        """
        return self.Xk.Y, self.Uk.Y, self.Ek1.Y

    def l2_norm(self):
        """
        L2 norm of each residual vector:
            ||E_k||_2 for k = 0..N-1
        Shape: (N,)
        """
        return np.linalg.norm(self.Ek1.Y, axis=1)

    def max_norm(self):
        """
        Infinity norm of each residual vector:
            max_i |E_k[i]| for k = 0..N-1
        Shape: (N,)
        """
        return np.max(np.abs(self.Ek1.Y), axis=1)

    def mean(self):
        """
        Mean absolute value of each residual vector:
            (1/nx) * sum_i |E_k[i]|
        Shape: (N,)
        """
        return np.mean(np.abs(self.Ek1.Y), axis=1)
