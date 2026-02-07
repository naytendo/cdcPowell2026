from dataclasses import dataclass, replace
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from core.signal import Signal
from learning.residual_data import ResidualData
import matplotlib.pyplot as plt

ArrF = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class Trajectory:
    X: Signal     # state signal (N+1, nx)
    U: Signal     # input signal (N, nu)
    dt: float

    # -------------------------
    # Basic properties
    # -------------------------
    @property
    def t(self) -> ArrF:
        return self.X.t

    @property
    def N(self) -> int:
        return self.U.N

    @property
    def nx(self) -> int:
        return self.X.dim

    @property
    def nu(self) -> int:
        return self.U.dim

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    # -------------------------
    # Validation
    # -------------------------
    def __post_init__(self):
        if self.X.N != self.U.N + 1:
            raise ValueError(
                f"X has {self.X.N} samples but U has {self.U.N}. "
                "Expected X.N = U.N + 1."
            )
        if not np.isclose(self.X.dt, self.dt):
            raise ValueError("X.dt must equal Trajectory.dt")
        if not np.isclose(self.U.dt, self.dt):
            raise ValueError("U.dt must equal Trajectory.dt")

    # -------------------------
    # Replacement helper
    # -------------------------
    def with_(self, *, X=None, U=None, dt=None) -> "Trajectory":
        return replace(
            self,
            X=self.X if X is None else X,
            U=self.U if U is None else U,
            dt=self.dt if dt is None else float(dt),
        )

    # -------------------------
    # Residuals
    # -------------------------
    def get_residuals(
        self,
        f_model_dt: Callable[[ArrF, ArrF, dict], ArrF],
        p_model: dict
    ) -> ResidualData:

        X = self.X.Y
        U = self.U.Y
        t = self.X.t
        dt = self.dt
        N = self.U.N

        Xk  = X[:-1]          # (N, nx)
        Uk  = U               # (N, nu)
        Ek1 = np.zeros_like(Xk)

        for k in range(N):
            xk = X[k]
            uk = U[k]
            xkp1_model = f_model_dt(xk, uk, p_model)
            Ek1[k] = X[k+1] - xkp1_model

        # Build signals
        sig_Xk  = Signal(t=t[:-1], Y=Xk,  dt=dt, labels=self.X.labels)
        sig_Uk  = Signal(t=t[:-1], Y=Uk,  dt=dt, labels=self.U.labels)
        sig_Ek1 = Signal(t=t[:-1], Y=Ek1, dt=dt, labels=self.X.labels)

        return ResidualData(Xk=sig_Xk, Uk=sig_Uk, Ek1=sig_Ek1)


    # -------------------------
    # State lookup
    # -------------------------
    def get_states(self, t=None, method="nearest") -> ArrF:
        if t is None:
            return self.X.Y
        return self.X.at(t, method=method)

    # -------------------------
    # Input lookup
    # -------------------------
    def get_input(self, t=None, method="nearest") -> ArrF:
        if t is None:
            return self.U.Y
        return self.U.at(t, method=method)

    # -------------------------
    # Plotting
    # -------------------------
    def plot(self, *, show=True, save_prefix=None, gamma_in_deg=True):
        tX = self.X.t
        X  = self.X.Y
        U  = self.U.Y
        tU = self.U.t

        # ---- States ----
        fig1, axs = plt.subplots(self.nx, 1, figsize=(10, 2.5*self.nx), sharex=True)
        if self.nx == 1:
            axs = [axs]

        for i in range(self.nx):
            y = X[:, i].copy()
            label = f"x[{i}]"

            if i == 1:  # gamma
                label = "gamma [deg]" if gamma_in_deg else "gamma [rad]"
                if gamma_in_deg:
                    y = np.rad2deg(y)

            axs[i].plot(tX, y)
            axs[i].set_ylabel(label)
            axs[i].grid(alpha=0.3)

        axs[-1].set_xlabel("time [s]")
        fig1.tight_layout()

        if save_prefix:
            fig1.savefig(f"{save_prefix}_states.png", dpi=160)

        # ---- Inputs ----
        if self.nu > 0:
            fig2, axs_u = plt.subplots(self.nu, 1, figsize=(10, 2.5*self.nu), sharex=True)
            if self.nu == 1:
                axs_u = [axs_u]

            for j in range(self.nu):
                uj = U[:, j]
                lbl = f"u[{j}]"

                if j == 0:  # alpha
                    lbl = "alpha [deg]"
                    uj = np.rad2deg(uj)
                if j == 1:  # throttle
                    lbl = "throttle [-]"

                axs_u[j].plot(tU, uj)
                axs_u[j].set_ylabel(lbl)
                axs_u[j].grid(alpha=0.3)

            axs_u[-1].set_xlabel("time [s]")
            fig2.tight_layout()

            if save_prefix:
                fig2.savefig(f"{save_prefix}_inputs.png", dpi=160)

        if show:
            plt.show()
        else:
            plt.close("all")
