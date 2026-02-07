from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ..core.signal import Signal
from ..trajectory import Trajectory
from .continuous_model import ContinuousModel
from simulate import rollout

ArrF = NDArray[np.floating]

EPS = 1e-6

@dataclass
class DiscreteModel:
    cont_sys: ContinuousModel
    dt: float
    method: str = "euler"   # or "rk4"

    def __post_init__(self):
        # Build and store the discrete-time update function
        self.f_dt = self._build_discrete_map()

    # ---------------------------------------------------------
    # Build f_dt(x,u,p) based on chosen method
    # ---------------------------------------------------------
    def _build_discrete_map(self):
        f_ct = self.cont_sys.get_derivative
        dt = self.dt

        if self.method == "euler":
            def f_dt(x, u, p):
                return x + dt * f_ct(x, u)
            return f_dt

        elif self.method == "rk4":
            def f_dt(x, u, p):
                h = dt
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
    def step(self, x: ArrF, u: ArrF) -> ArrF:
        return self.f_dt(x, u, self.cont_sys.params)

    # ---------------------------------------------------------
    # Rollout: Signal + Signal → Signal
    # ---------------------------------------------------------
    def rollout_state(self, X: Signal, U: Signal) -> Signal:
        X_arr = X.Y
        U_arr = U.Y

        # Wrap self.step into a 2-argument function
        def f(x, u):
            return self.step(x, u)

        X_sim = rollout(f, X_arr[0], U_arr)

        return Signal(t=X.t, Y=X_sim, dt=X.dt, labels=X.labels)


    # ---------------------------------------------------------
    # Rollout: Trajectory → Trajectory
    # ---------------------------------------------------------
    def rollout_trajectory(self, traj: Trajectory) -> Trajectory:
        X_sim = self.rollout_state(traj.X, traj.U)
        return Trajectory(X=X_sim, U=traj.U, dt=traj.dt)
    
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
        dt = traj.dt

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
            f0 = self.step(xk, uk)

            # ---- A_k = df/dx ----
            for i in range(nx):
                dx = np.zeros(nx)
                dx[i] = EPS
                f_plus = self.step(xk + dx, uk)
                A[k, :, i] = (f_plus - f0) / EPS

            # ---- B_k = df/du ----
            for j in range(nu):
                du = np.zeros(nu)
                du[j] = EPS
                f_plus = self.step(xk, uk + du)
                B[k, :, j] = (f_plus - f0) / EPS

        # Wrap in Signals
        A_signal = Signal(t=t, Y=A, dt=dt, labels=None)
        B_signal = Signal(t=t, Y=B, dt=dt, labels=None)

        return A_signal, B_signal
