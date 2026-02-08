from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from core.signal import Signal
from trajectory import Trajectory
from .continuous_model import ContinuousModel
from simulate import rollout

ArrF = NDArray[np.floating]

EPS = 1e-6

@dataclass
class DiscreteModel:
    cont_sys: ContinuousModel
    t: np.ndarray                 # full time vector
    method: str = "euler"

    def __post_init__(self):
        # compute dt[k] = t[k+1] - t[k]
        self.dt = np.diff(self.t)
        self.f_dt = self._build_discrete_map()

    def _build_discrete_map(self):
        f_ct = self.cont_sys.get_derivative
        dt_vec = self.dt

        if self.method == "euler":
            def f_dt(x, u, k):
                h = dt_vec[k]
                return x + h * f_ct(x, u)
            return f_dt

        elif self.method == "rk4":
            def f_dt(x, u, k):
                h = dt_vec[k]
                k1 = f_ct(x, u)
                k2 = f_ct(x + 0.5*h*k1, u)
                k3 = f_ct(x + 0.5*h*k2, u)
                k4 = f_ct(x + h*k3, u)
                return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            return f_dt

        else:
            raise ValueError(f"Unknown discretization method: {self.method}")

    # ---------------------------------------------------------
    # One-step propagation using stored f_dt
    # ---------------------------------------------------------
    def step(self, x: ArrF, u: ArrF, k: int) -> ArrF:
        return self.f_dt(x, u, k)


    # ---------------------------------------------------------
    # Rollout: Signal + Signal → Signal
    # ---------------------------------------------------------
    def rollout_state(self, X: Signal, U: Signal) -> Signal:
        X_arr = X.Y
        U_arr = U.Y
        N = U_arr.shape[0]

        X_sim = np.zeros_like(X_arr)
        X_sim[0] = X_arr[0]

        for k in range(N):
            X_sim[k+1] = self.step(X_sim[k], U_arr[k], k)

        return Signal(t=X.t, Y=X_sim, labels=X.labels)



    # ---------------------------------------------------------
    # Rollout: Trajectory → Trajectory
    # ---------------------------------------------------------
    def rollout_trajectory(self, traj: Trajectory) -> Trajectory:
        X_sim = self.rollout_state(traj.X, traj.U)
        return Trajectory(X=X_sim, U=traj.U)
    
    def linearize(self, traj: Trajectory):
        """
        Compute discrete-time Jacobians A_k = df_dt/dx, B_k = df_dt/du
        along a trajectory.

        Returns:
            A_signal: Signal with shape (N, nx, nx)
            B_signal: Signal with shape (N, nx, nu)
        """
        X = traj.X.Y          # (N+1, nx)
        U = traj.U.Y          # (N, nu)
        t = traj.U.t          # Jacobians defined at input times

        N = traj.U.N
        nx = self.cont_sys.nx
        nu = self.cont_sys.nu

        A = np.zeros((N, nx, nx))
        B = np.zeros((N, nx, nu))

        # Finite-difference Jacobians
        for k in range(N):
            xk = X[k]
            uk = U[k]

            # baseline
            f0 = self.step(xk, uk, k)

            # ---- A_k = df/dx ----
            for i in range(nx):
                dx = np.zeros(nx)
                dx[i] = EPS
                f_plus = self.step(xk + dx, uk, k)
                A[k, :, i] = (f_plus - f0) / EPS

            # ---- B_k = df/du ----
            for j in range(nu):
                du = np.zeros(nu)
                du[j] = EPS
                f_plus = self.step(xk, uk + du, k)
                B[k, :, j] = (f_plus - f0) / EPS

        # Wrap in Signals
        A_signal = Signal(t=t, Y=A, labels=None)
        B_signal = Signal(t=t, Y=B, labels=None)

        return A_signal, B_signal
