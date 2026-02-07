# mpc_controller.py
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp

from models.discrete_model import DiscreteModel
from signal import Signal
from trajectory import Trajectory

ArrF = NDArray[np.floating]


@dataclass
class LinearMPC:
    model: DiscreteModel
    N: int                 # horizon length
    Q: ArrF                # (nx, nx)
    R: ArrF                # (nu, nu)
    Qf: ArrF               # (nx, nx)
    u_min: Optional[ArrF] = None  # (nu,)
    u_max: Optional[ArrF] = None  # (nu,)

    def _build_nominal_trajectory(
        self,
        x0: ArrF,
        U_last: Optional[ArrF] = None,
    ) -> Trajectory:
        """
        Build a simple nominal trajectory by rolling out the model
        from x0 with either:
          - last MPC solution's inputs, or
          - zero inputs if none provided.
        """
        nx = self.model.cont_sys.nx
        nu = self.model.cont_sys.nu
        dt = self.model.dt

        if U_last is not None:
            U_nom = U_last.copy()
        else:
            U_nom = np.zeros((self.N, nu))

        # time vector
        t = np.linspace(0.0, self.N * dt, self.N + 1)

        # build signals
        X0 = np.zeros((self.N + 1, nx))
        X0[0] = x0

        X_sig = Signal(t=t, Y=X0, dt=dt)
        U_sig = Signal(t=t[:-1], Y=U_nom, dt=dt)

        traj_nom = Trajectory(X=X_sig, U=U_sig, dt=dt)
        # roll out to get consistent nominal states
        traj_nom = self.model.rollout(traj_nom)
        return traj_nom

    def _linearize_along(self, traj: Trajectory):
        """
        Compute linearization:
            x_{k+1} ≈ x̄_{k+1} + A_k (x_k - x̄_k) + B_k (u_k - ū_k)
        and affine term:
            c_k = x̄_{k+1} - A_k x̄_k - B_k ū_k
        """
        X_bar = traj.X.Y      # (N+1, nx)
        U_bar = traj.U.Y      # (N,   nu)
        A_sig, B_sig = self.model.linearize(traj)
        A = A_sig.Y           # (N, nx, nx)
        B = B_sig.Y           # (N, nx, nu)

        N, nx, nu = traj.U.N, self.model.cont_sys.nx, self.model.cont_sys.nu
        c = np.zeros((N, nx))

        for k in range(N):
            x_bar_k = X_bar[k]
            u_bar_k = U_bar[k]
            x_bar_kp1 = X_bar[k+1]
            Ak = A[k]
            Bk = B[k]
            c[k] = x_bar_kp1 - Ak @ x_bar_k - Bk @ u_bar_k

        return A, B, c, X_bar, U_bar

    def solve(
        self,
        x0: ArrF,
        X_ref: Optional[ArrF] = None,   # shape (N+1, nx) or None
        U_ref: Optional[ArrF] = None,   # shape (N, nu) or None
        U_last: Optional[ArrF] = None,  # previous optimal inputs (for warm start)
    ) -> ArrF:
        """
        Solve the MPC QP and return the optimal first input u0*.
        """
        nx = self.model.cont_sys.nx
        nu = self.model.cont_sys.nu

        # 1) nominal trajectory
        traj_nom = self._build_nominal_trajectory(x0, U_last)
        A, B, c, X_bar, U_bar = self._linearize_along(traj_nom)
        N = self.N

        # references: if None, track nominal
        if X_ref is None:
            X_ref = X_bar
        if U_ref is None:
            U_ref = U_bar

        # 2) decision variables: δx, δu
        dx = cp.Variable((N+1, nx))
        du = cp.Variable((N,   nu))

        cost = 0
        constr = []

        # initial condition: x0 = x̄0 + δx0  =>  δx0 = x0 - x̄0
        constr += [dx[0, :] == x0 - X_bar[0]]

        # dynamics: δx_{k+1} = A_k δx_k + B_k δu_k + c_k - c_k  (affine part handled by ref)
        # but easier to work in absolute x,u in the cost:
        for k in range(N):
            constr += [
                dx[k+1, :] == A[k] @ dx[k, :] + B[k] @ du[k, :] + c[k] - c[k]
            ]

        # 3) cost
        for k in range(N):
            xk = X_bar[k] + dx[k, :]
            uk = U_bar[k] + du[k, :]

            x_ref_k = X_ref[k]
            u_ref_k = U_ref[k]

            cost += cp.quad_form(xk - x_ref_k, self.Q)
            cost += cp.quad_form(uk - u_ref_k, self.R)

        # terminal cost
        xN = X_bar[N] + dx[N, :]
        x_ref_N = X_ref[N]
        cost += cp.quad_form(xN - x_ref_N, self.Qf)

        # 4) input constraints (if provided)
        if self.u_min is not None:
            constr += [U_bar[k] + du[k, :] >= self.u_min for k in range(N)]
        if self.u_max is not None:
            constr += [U_bar[k] + du[k, :] <= self.u_max for k in range(N)]

        # 5) solve QP
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"MPC QP solve failed with status {prob.status}")

        du_opt = du.value
        u0 = U_bar[0] + du_opt[0]
        return u0
    
    def control(self, x, k, ref=None):
        return self.solve(x)

