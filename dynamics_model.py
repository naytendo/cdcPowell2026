from dataclasses import dataclass
from typing import Callable
import numpy as np
import differentiate
from core.structs import Trajectory

ArrF = np.ndarray

@dataclass(frozen=True)
class DynamicsModel:
    name: str
    f_ct: Callable[[ArrF, ArrF, dict], ArrF]   # xdot = f_ct(x,u,p)
    params: dict
    dt: float


    def linearize_dt(self, traj: Trajectory):
        """
        Finite-difference linearization of the discrete map x+dt*f_ct(x,u,p).
        """
        """
        traj: Trajectory with t, X_ref, U_ref, dt
        f_model_dt: callable (x,u) -> x_next using the model for TVLQR
        p_model: parameters (if needed, otherwise captured in closure)
        Q, R: LQR weights
        P_N: Terminal P
        Returns: list of feedback gains K[k] and maybe A,B for debugging
        """
        X = traj.X
        U = traj.U
        dt   = traj.dt
        N    = U.shape[0]
        f_dt = self.euler_discretize(self.f_ct, dt)

        A_seq = []
        B_seq = []

        for k in range(N):
            xk = X[k]
            uk = U[k]
            # central-difference Jacobian on the discrete-time map
            Ak, Bk = differentiate.numeric_central_diff(f_dt, xk, uk, self.params) 
            A_seq.append(Ak)
            B_seq.append(Bk)
        return A_seq, B_seq
    
    def euler_discretize(self):
        def f_dt(x, u, p):
            return x + self.dt * self.f_ct(x, u, p)
        return f_dt