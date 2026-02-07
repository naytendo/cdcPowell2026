from dataclasses import dataclass, replace
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from .signal import Signal

# ---- Array aliases ----
ArrF = NDArray[np.floating]
# ---- Common data containers ----

@dataclass(frozen=True)
class ResidualData:
    tk: ArrF #(N, )
    Xk: ArrF   # (N, nx)
    Uk: ArrF   # (N, nu)
    Ek1: ArrF  # (N, nx)  e_{k+1} = x_{k+1}^true - f_nom(x_k,u_k)

@dataclass(frozen=True, slots=True)
class Trajectory:
    """
    Discrete-time reference over a horizon:
      - t:   (N+1,) time stamps [s]
      - X:   (N+1,nx) reference states
      - U:   (N,nu)   reference inputs (often zeros; can be feasibilized later)
      - dt:  scalar step
    """
    t: ArrF
    X: ArrF
    U: ArrF
    dt: float

    @property
    def N(self) -> int:          # number of control intervals
        return self.U.shape[0]

    @property
    def nx(self) -> int:
        return self.X.shape[1]

    @property
    def nu(self) -> int:
        return self.U.shape[1]

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])
    
    def with_(self, *, t=None, X=None, U=None, dt=None, kind=None) -> "Trajectory":
        """Return a new Trajectory with selected fields replaced."""
        return replace(
            self,
            t=self.t if t is None else t,
            X=self.X if X is None else X,
            U=self.U if U is None else U,
            dt=self.dt if dt is None else float(dt)
        )
    
    def get_residuals(
        self,
        f_model_dt: Callable[[ArrF, ArrF, dict], ArrF],
        p_model: dict
    ) -> ResidualData:
        """
        Compute one-step residuals of a given discrete-time model along this trajectory.

        Ek1[k] = X[k+1] - f_model_dt(X[k], U[k], p_model)
        for k = 0..N-1.

        Typically, 'self' is a true or reference trajectory, and f_model_dt is your
        nominal or learned model.
        """
        X = self.X
        U = self.U
        N = U.shape[0]

        Xk  = X[:-1].copy()       # (N, nx)
        Uk  = U.copy()            # (N, nu)
        Ek1 = np.zeros_like(Xk)   # (N, nx)

        for k in range(N):
            xk = X[k]
            uk = U[k]
            xkp1_model = f_model_dt(xk, uk, p_model)
            Ek1[k] = X[k+1] - xkp1_model

        t_k = self.t[:-1].copy()  # time stamps aligned with Xk, Uk

        return ResidualData(tk=t_k, Xk=Xk, Uk=Uk, Ek1=Ek1)


@dataclass(frozen=True, slots=True)
class AeroTrace:
    """Aero coefficients/forces along a run (evaluated at the same (X[k],U[k]))."""
    t: ArrF              # (N,)
    CL_nom: ArrF         # (N,)
    CD_nom: ArrF         # (N,)
    L_nom:  ArrF         # (N,)
    D_nom:  ArrF         # (N,)
    T_nom:  ArrF         # (N,)

    CL_true: ArrF        # (N,)
    CD_true: ArrF        # (N,)
    L_true:  ArrF        # (N,)
    D_true:  ArrF        # (N,)
    T_true:  ArrF        # (N,)

    q: ArrF              # (N,) dynamic pressure (same for nom/true here)
    M: ArrF              # (N,) Mach (same here)
    alpha: ArrF          # (N,) angle of attack used
    thr: ArrF            # (N,) throttle used

    # convenient deltas
    @property
    def dCL(self) -> ArrF: return self.CL_true - self.CL_nom
    @property
    def dCD(self) -> ArrF: return self.CD_true - self.CD_nom
    @property
    def dL(self)  -> ArrF: return self.L_true  - self.L_nom
    @property
    def dD(self)  -> ArrF: return self.D_true  - self.D_nom
    @property
    def dT(self)  -> ArrF: return self.T_true  - self.T_nom

@dataclass(frozen=True)
class DynEval:
    xdot: np.ndarray   # [dv, dgamma, dh]
    CL: float
    CD: float
    L:  float
    D:  float
    T:  float
    q:  float
    M:  float
    rho: float

@dataclass(frozen=True, slots=True)
class DefectTrace:
    t: ArrF                     # (N,)
    d: ArrF                     # (N, nx)   d_k = f(X[k],U[k]) - X[k+1]
    norms: ArrF                 # (N,)

@dataclass(frozen=True)
class DynEvalSignals:
    xdot: Signal
    CL: Signal
    CD: Signal
    L: Signal
    D: Signal
    T: Signal
    q: Signal
    M: Signal
    rho: Signal
