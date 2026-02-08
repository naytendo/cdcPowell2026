# analysis/residuals.py

import numpy as np
from .residual_data import ResidualData
from core.signal import Signal

def collect_residuals(model_nom, model_true, traj):
    """
    Compute one-step prediction residuals between a nominal and true model
    along a given trajectory.

    Residual:
        E[k] = x_{k+1}^{true} - f_nom(x_k, u_k)

    Parameters
    ----------
    model_nom : DiscreteModel
        The nominal model with a .step(x,u) method.
    model_true : DiscreteModel
        The true model with a .step(x,u) method.
    traj : Trajectory
        Contains X (states), U (inputs), t (time), dt (step size).

    Returns
    -------
    ResidualData
        Structured residual dataset with Signals:
            Xk  : (N, nx) states at time k
            Uk  : (N, nu) inputs at time k
            Ek1 : (N, nx) one-step residuals
    """

    X = traj.X.Y          # shape (N+1, nx)
    U = traj.U.Y          # shape (N,   nu)
    t = traj.X.t          # shape (N+1,)
    N = traj.U.N

    Xk  = X[:-1]          # (N, nx)
    Uk  = U               # (N, nu)
    Ek1 = np.zeros_like(Xk)

    for k in range(N):
        xk = X[k]
        uk = U[k]

        xkp1_true = model_true.step(xk, uk, k)
        xkp1_nom  = model_nom.step(xk, uk, k)

        Ek1[k] = xkp1_true - xkp1_nom

    # Wrap into Signals
    sig_Xk  = Signal(t=t[:-1], Y=Xk, labels=traj.X.labels)
    sig_Uk  = Signal(t=t[:-1], Y=Uk, labels=traj.U.labels)
    sig_Ek1 = Signal(t=t[:-1], Y=Ek1, labels=traj.X.labels)

    return ResidualData(Xk=sig_Xk, Uk=sig_Uk, Ek1=sig_Ek1)
