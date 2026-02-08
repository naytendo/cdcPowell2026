
import numpy as np
from models.discrete_model import DiscreteModel

class TVLQR:
    def __init__(self, model, Q, R, Qf=None,
                 du_max=None, u_min=None, u_max=None):
        self.model = model
        self.Q = Q
        self.R = R
        self.Qf = Qf if Qf is not None else Q

        # rate limits (per component)
        self.du_max = du_max   # e.g. np.array([deg2rad(2), 0.15])
        # saturation limits
        self.u_min = u_min
        self.u_max = u_max

        # store previous applied control
        self.u_prev = None


    def compute_gains(self, traj):
        X_ref = traj.X.Y
        U_ref = traj.U.Y
        N = traj.N

        # Linearize along trajectory
        A_sig, B_sig = self.model.linearize(traj)
        A = A_sig.Y
        B = B_sig.Y

        nx = X_ref.shape[1]
        nu = U_ref.shape[1]

        # Riccati recursion
        P = [None] * (N + 1)
        K = [None] * N

        P[N] = self.Qf

        for k in reversed(range(N)):
            Ak = A[k]
            Bk = B[k]

            S = self.R + Bk.T @ P[k+1] @ Bk
            K[k] = np.linalg.solve(S, Bk.T @ P[k+1] @ Ak)

            P[k] = self.Q + Ak.T @ P[k+1] @ (Ak - Bk @ K[k])

        self.K = K
        self.X_ref = X_ref
        self.U_ref = U_ref
        self.N = N

    def control(self, x, k):
        # raw TVLQR control
        u = self.U_ref[k] - self.K[k] @ (x.flatten() - self.X_ref[k])

        # initialize previous control on first call
        if self.u_prev is None:
            self.u_prev = u.copy()

        # ---- rate limits ----
        if self.du_max is not None:
            du = u - self.u_prev
            du = np.clip(du, -self.du_max, self.du_max)
            u = self.u_prev + du

        # ---- saturation ----
        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)

        # update memory
        self.u_prev = u.copy()

        return u
