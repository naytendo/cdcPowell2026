from dataclasses import dataclass
from typing import Sequence, Optional
from trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Trajectories:
    trajs: Sequence[Trajectory]

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajs[idx]

    # ---------------------------------------------------------
    # Internal helper for plotting a single trajectory
    # ---------------------------------------------------------
    @staticmethod
    def _plot_traj_on_axes(
        traj: Trajectory,
        ax_states,
        ax_inputs,
        *,
        label: str = "",
        gamma_in_deg: bool = True,
    ):
        # Extract signals
        tX = traj.X.t
        X  = traj.X.Y
        U  = traj.U.Y
        tU = traj.U.t

        nx = traj.nx
        nu = traj.nu

        # ---- States: assume [v, gamma, h] ----
        if nx >= 1:
            ax_states[0].plot(tX, X[:, 0], label=label)

        if nx >= 2:
            y = X[:, 1]
            if gamma_in_deg:
                y = np.rad2deg(y)
            ax_states[1].plot(tX, y, label=label)

        if nx >= 3:
            ax_states[2].plot(tX, X[:, 2], label=label)

        # ---- Inputs: assume [alpha, throttle] ----
        if nu >= 1:
            ax_inputs[0].plot(tU, np.rad2deg(U[:, 0]), label=label)

        if nu >= 2:
            ax_inputs[1].plot(tU, U[:, 1], label=label)

    # ---------------------------------------------------------
    # Public plotting method
    # ---------------------------------------------------------
    def plot(
        self,
        *,
        labels: Optional[Sequence[str]] = None,
        gamma_in_deg: bool = True,
        show: bool = True,
        save_prefix: Optional[str] = None,
    ) -> None:

        if labels is None:
            labels = [f"traj_{i}" for i in range(len(self.trajs))]

        # Shared axes
        fig_s, axs_s = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig_u, axs_u = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        # State labels
        axs_s[0].set_ylabel("v [m/s]")
        axs_s[1].set_ylabel("gamma [deg]" if gamma_in_deg else "gamma [rad]")
        axs_s[2].set_ylabel("h [m]")
        for ax in axs_s:
            ax.grid(alpha=0.3)
        axs_s[-1].set_xlabel("time [s]")

        # Input labels
        axs_u[0].set_ylabel("alpha [deg]")
        axs_u[1].set_ylabel("throttle [-]")
        for ax in axs_u:
            ax.grid(alpha=0.3)
        axs_u[-1].set_xlabel("time [s]")

        # Plot each trajectory
        for traj, lbl in zip(self.trajs, labels):
            self._plot_traj_on_axes(
                traj,
                axs_s,
                axs_u,
                label=lbl,
                gamma_in_deg=gamma_in_deg,
            )

        # Legends
        axs_s[0].legend(loc="best")
        axs_u[0].legend(loc="best")

        fig_s.tight_layout()
        fig_u.tight_layout()

        if save_prefix:
            fig_s.savefig(f"{save_prefix}_states.png", dpi=160)
            fig_u.savefig(f"{save_prefix}_inputs.png", dpi=160)

        if show:
            plt.show()
        else:
            plt.close(fig_s)
            plt.close(fig_u)
