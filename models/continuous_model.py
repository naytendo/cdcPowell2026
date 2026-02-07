from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
from numpy.typing import NDArray
from core.structs import Trajectory
from core.signal import Signal
from core.structs import DynEval, DynEvalSignals
from typing import Optional
from warm_start.dispatcher import warm_start_U

ArrF = NDArray[np.floating]


@dataclass
class ContinuousModel:
    nx: int
    nu: int
    params: Dict
    f_ct: Callable[[ArrF, ArrF, Dict], ArrF]   # dx/dt = f(x,u,p)
    dyn_eval: Optional[Callable[[np.ndarray, np.ndarray, dict], DynEval]] = None

    # ---------------------------------------------------------
    # Single-step derivative
    # ---------------------------------------------------------
    def get_derivative(self, x: ArrF, u: ArrF) -> ArrF:
        return self.f_ct(x, u, self.params)

    # ---------------------------------------------------------
    # Derivative signal along a trajectory
    # ---------------------------------------------------------
    def derivative_signal(self, traj: Trajectory) -> Signal:
        """
        Compute dX/dt along a trajectory (X(t), U(t)).
        Returns a DerivativeSignal with same time vector and dt.
        """
        X = traj.X.Y          # (N+1, nx)
        U = traj.U.Y          # (N, nu)
        t = traj.X.t
        dt = traj.dt
        N = traj.U.N

        dXdt = np.zeros_like(X)

        for k in range(N + 1):
            # For the last state, reuse the last input
            u_k = U[k] if k < N else U[-1]
            dXdt[k] = self.get_derivative(X[k], u_k)

        return Signal(t=t, Y=dXdt, dt=dt)
    
    def eval_trajectory(self, traj: Trajectory) -> DynEvalSignals:
        X = traj.X.Y
        U = traj.U.Y
        t = traj.X.t
        dt = traj.dt
        N = traj.U.N

        # allocate arrays
        xdot = np.zeros_like(X)
        CL   = np.zeros(N+1)
        CD   = np.zeros(N+1)
        L    = np.zeros(N+1)
        D    = np.zeros(N+1)
        T    = np.zeros(N+1)
        q    = np.zeros(N+1)
        M    = np.zeros(N+1)
        rho  = np.zeros(N+1)

        for k in range(N+1):
            u_k = U[k] if k < N else U[-1]
            dyn = self.dyn_eval(X[k], u_k, self.params)

            xdot[k] = dyn.xdot
            CL[k]   = dyn.CL
            CD[k]   = dyn.CD
            L[k]    = dyn.L
            D[k]    = dyn.D
            T[k]    = dyn.T
            q[k]    = dyn.q
            M[k]    = dyn.M
            rho[k]  = dyn.rho

        return DynEvalSignals(
            xdot = Signal(t=t, Y=xdot, dt=dt),
            CL   = Signal(t=t, Y=CL,   dt=dt),
            CD   = Signal(t=t, Y=CD,   dt=dt),
            L    = Signal(t=t, Y=L,    dt=dt),
            D    = Signal(t=t, Y=D,    dt=dt),
            T    = Signal(t=t, Y=T,    dt=dt),
            q    = Signal(t=t, Y=q,    dt=dt),
            M    = Signal(t=t, Y=M,    dt=dt),
            rho  = Signal(t=t, Y=rho,  dt=dt),
        )
    
    def warm_start(self, traj, **kwargs): 
        return warm_start_U(traj, self, **kwargs)
