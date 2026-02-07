import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence
# If you're inside a package, use: from .types import Trajectory
from core.structs import Trajectory, ResidualData

def plot_tvlqr_run(X_true, Xref, U_applied, Uref, E, dt,
                   state_labels=None, control_labels=None, save=True, prefix="tvlqr"):
    """
    X_true: (Tx+1, nx) or (Tx, nx)
    Xref:   (Tx+1, nx) or (Tx, nx)
    U_applied: (Tu, nu)
    Uref:      (Tu, nu)
    E:      (Tu, nx) residuals from collect_residuals (model error per step)
    dt:     timestep [s]
    """
    X_true = np.asarray(X_true); Xref = np.asarray(Xref)
    U_applied = np.asarray(U_applied); Uref = np.asarray(Uref)
    E = np.asarray(E)

    nx = X_true.shape[1]
    nu = U_applied.shape[1]

    # default labels (edit if your ordering differs)
    if state_labels is None:
        state_labels = [r"$v$ [m/s]", r"$\gamma$ [rad]", r"$h$ [m]"][:nx]
    if control_labels is None:
        control_labels = [r"$\alpha$ [rad]", "throttle"][:nu]

    # --- time vectors (align lengths robustly) ---
    Tx = min(X_true.shape[0], Xref.shape[0])
    Tu = min(U_applied.shape[0], Uref.shape[0], E.shape[0])
    t_x = np.arange(Tx) * dt
    t_u = np.arange(Tu) * dt

    # --- Figure 1: States true vs ref ---
    fig1, axes1 = plt.subplots(nx, 1, figsize=(8, 2.4*nx), sharex=True)
    axes1 = np.atleast_1d(axes1)
    for i in range(nx):
        axes1[i].plot(t_x, X_true[:Tx, i], label="true", linewidth=2)
        axes1[i].plot(t_x, Xref[:Tx, i],  label="ref",  linestyle="--")
        axes1[i].set_ylabel(state_labels[i])
        axes1[i].grid(True, alpha=0.3)
        axes1[i].legend()
    axes1[-1].set_xlabel("time [s]")
    fig1.suptitle("States: true vs reference")
    fig1.tight_layout()
    if save: fig1.savefig(f"{prefix}_states.png", dpi=160)

    # --- Figure 2: Inputs applied vs ref ---
    fig2, axes2 = plt.subplots(nu, 1, figsize=(8, 2.4*nu), sharex=True)
    axes2 = np.atleast_1d(axes2)
    for j in range(nu):
        axes2[j].plot(t_u, U_applied[:Tu, j], label="applied", linewidth=2)
        axes2[j].plot(t_u, Uref[:Tu, j],      label="ref", linestyle="--")
        axes2[j].set_ylabel(control_labels[j])
        axes2[j].grid(True, alpha=0.3)
        axes2[j].legend()
    axes2[-1].set_xlabel("time [s]")
    fig2.suptitle("Inputs: applied vs reference")
    fig2.tight_layout()
    if save: fig2.savefig(f"{prefix}_inputs.png", dpi=160)

    # --- Figure 3: Residuals (model error) ---
    # E[k] = f_true_dt(x_k,u_k) - f_nom_dt(x_k,u_k)
    fig3, axes3 = plt.subplots(nx+1, 1, figsize=(8, 2.4*(nx+1)), sharex=True)
    axes3 = np.atleast_1d(axes3)
    for i in range(nx):
        axes3[i].plot(t_u, E[:Tu, i], label=fr"$E_{{{i}}}$", linewidth=1.8)
        axes3[i].set_ylabel(state_labels[i].replace(" [", " Δ["))
        axes3[i].grid(True, alpha=0.3)
    # residual 2-norm
    norms = np.linalg.norm(E[:Tu, :], axis=1)
    axes3[-1].plot(t_u, norms, label=r"$\|E_k\|_2$", linewidth=2)
    axes3[-1].set_ylabel(r"$\|E_k\|_2$")
    axes3[-1].set_xlabel("time [s]")
    for ax in axes3: ax.grid(True, alpha=0.3)
    fig3.suptitle("One-step residuals (true - nominal)")
    fig3.tight_layout()
    if save: fig3.savefig(f"{prefix}_residuals.png", dpi=160)

    plt.show()

# viz.py
import numpy as np
import matplotlib.pyplot as plt

def plot_true_vs_nominal(
    X_true,
    X_nom_1step=None,
    X_nom_rollout=None,
    U=None,                 # (N, nu) applied inputs; optional
    E=None,                 # (N, nx) one-step residuals; optional
    dt=0.05,
    prefix="nominal",
    state_labels=None,
    control_labels=None,
    show=True,
    save=True,
):
    X_true = np.asarray(X_true)
    nx = X_true.shape[1]

    if X_nom_1step is not None:
        X_nom_1step = np.asarray(X_nom_1step)
    if X_nom_rollout is not None:
        X_nom_rollout = np.asarray(X_nom_rollout)
    if U is not None:
        U = np.asarray(U)
        nu = U.shape[1]
    if E is not None:
        E = np.asarray(E)

    if state_labels is None:
        state_labels = [r"$v$ [m/s]", r"$\gamma$ [rad]", r"$h$ [m]"][:nx]
    if control_labels is None and U is not None:
        control_labels = [r"$\alpha$ [rad]", "throttle"][:U.shape[1]]

    # ---- time axes (states vs inputs) ----
    Tx_candidates = [X_true.shape[0]]
    if X_nom_1step   is not None: Tx_candidates.append(X_nom_1step.shape[0])
    if X_nom_rollout is not None: Tx_candidates.append(X_nom_rollout.shape[0])
    Tx = min(Tx_candidates)
    t_x = np.arange(Tx) * dt

    if U is not None: 
        Tu = U.shape[0]
        t_u = np.arange(Tu) * dt
    if E is not None:
        Tu_E = E.shape[0]
        if U is None:
            t_u = np.arange(Tu_E) * dt
            Tu = Tu_E
        else:
            Tu = min(Tu, Tu_E)
            t_u = np.arange(Tu) * dt

    # ---- Figure 1: States (true vs nominal) ----
    fig1, axes1 = plt.subplots(nx, 1, figsize=(9, 2.5*nx), sharex=True)
    axes1 = np.atleast_1d(axes1)
    for i in range(nx):
        axes1[i].plot(t_x, X_true[:Tx, i], label="true", lw=2)
        if X_nom_rollout is not None:
            axes1[i].plot(t_x, X_nom_rollout[:Tx, i], ":", label="nom-roll", lw=1.7)
        if X_nom_1step is not None:
            axes1[i].plot(t_x, X_nom_1step[:Tx, i], "-.", label="nom-1step", lw=1.7)
        axes1[i].set_ylabel(state_labels[i])
        axes1[i].grid(alpha=0.3)
    axes1[-1].set_xlabel("time [s]")
    axes1[0].legend(ncol=3, fontsize=9)
    fig1.suptitle("States: true vs nominal")
    fig1.tight_layout()
    if save: fig1.savefig(f"{prefix}_states_nomonly.png", dpi=160)

    # ---- Figure 2: Inputs (optional) ----
    if U is not None:
        fig2, axes2 = plt.subplots(U.shape[1], 1, figsize=(9, 2.5*U.shape[1]), sharex=True)
        axes2 = np.atleast_1d(axes2)
        for j in range(U.shape[1]):
            axes2[j].plot(t_u, U[:Tu, j], lw=2)
            axes2[j].set_ylabel(control_labels[j] if control_labels else f"u{j}")
            axes2[j].grid(alpha=0.3)
        axes2[-1].set_xlabel("time [s]")
        fig2.suptitle("Inputs (applied)")
        fig2.tight_layout()
        if save: fig2.savefig(f"{prefix}_inputs_nomonly.png", dpi=160)

    # ---- Figure 3: Residuals (optional) ----
    if E is not None:
        fig3, axes3 = plt.subplots(nx+1, 1, figsize=(9, 2.5*(nx+1)), sharex=True)
        axes3 = np.atleast_1d(axes3)
        for i in range(nx):
            axes3[i].plot(t_u, E[:Tu, i], lw=1.8)
            axes3[i].set_ylabel(state_labels[i].replace(" [", " Δ["))
            axes3[i].grid(alpha=0.3)
        norms = np.linalg.norm(E[:Tu, :], axis=1)
        axes3[-1].plot(t_u, norms, lw=2)
        axes3[-1].set_ylabel(r"$\|E_k\|_2$")
        axes3[-1].set_xlabel("time [s]")
        for ax in axes3: ax.grid(alpha=0.3)
        fig3.suptitle("One-step residuals: true − nominal")
        fig3.tight_layout()
        if save: fig3.savefig(f"{prefix}_residuals_nomonly.png", dpi=160)

    if show: plt.show()
    else: plt.close('all')



from core.structs import AeroTrace

def plot_aero_trace(trace: AeroTrace, prefix: str | None = None, show: bool = False):
    t = trace.t

    # Coefficients
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(t, trace.CL_true, label='C_L true')
    plt.plot(t, trace.CL_nom,  '--', label='C_L nom')
    plt.xlabel('time [s]'); plt.ylabel('C_L'); plt.title('Lift coefficient'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t, trace.CD_true, label='C_D true')
    plt.plot(t, trace.CD_nom,  '--', label='C_D nom')
    plt.xlabel('time [s]'); plt.ylabel('C_D'); plt.title('Drag coefficient'); plt.legend()
    plt.tight_layout()
    if prefix: plt.savefig(f"{prefix}_CL_CD.png", dpi=160)

    # Differences (and optionally AoA/Mach context)
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t, trace.dCL); ax1.set_ylabel('ΔC_L')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.plot(t, trace.dCD); ax2.set_ylabel('ΔC_D')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    ax3.plot(t, trace.alpha, label='α [rad]')
    ax3.plot(t, trace.M,     label='Mach')
    ax3.set_xlabel('time [s]'); ax3.legend()
    plt.tight_layout()
    if prefix: plt.savefig(f"{prefix}_dCL_dCD.png", dpi=160)

    # Differences (and optionally AoA/Mach context)
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t, trace.T_nom); ax1.set_ylabel('Thr nom')
    ax1.plot(t, trace.T_nom); ax1.set_ylabel('Thr true')
    ax3.set_xlabel('time [s]'); ax3.legend()
    plt.tight_layout()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.plot(t, trace.L_nom); ax2.set_ylabel('L_nom')
    ax2.plot(t, trace.L_true); ax2.set_ylabel('L_true')
    ax3.set_xlabel('time [s]'); ax3.legend()
    plt.tight_layout()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    ax3.plot(t, trace.D_nom, label='D nom')
    ax3.plot(t, trace.D_true,     label='D true')
    ax3.set_xlabel('time [s]'); ax3.legend()
    plt.tight_layout()
    if prefix: plt.savefig(f"{prefix} delForces.png", dpi=160)
    plt.show()



def plot_trajectory(
    traj: Trajectory,
    *,
    show: bool = True,
    save_prefix: Optional[str] = None,
    gamma_in_deg: bool = True,
):
    """
    Plot a Trajectory (states and inputs) over time.

    Args:
        traj: Trajectory (t: (N+1,), X: (N+1,nx), U: (N,nu), dt)
        show: whether to show figures (plt.show) or close them (for batch runs).
        save_prefix: if provided, saves figures as f"{save_prefix}_states.png" and
                     f"{save_prefix}_inputs.png".
        gamma_in_deg: plot gamma in degrees if True, radians if False.
    """
    tX = traj.t                         # (N+1,)
    X  = traj.X.Y                        # (N+1, nx)
    U  = traj.U.Y                        # (N, nu)
    tU = traj.t[:-1] if len(traj.t) > 1 else traj.t[:0]  # align inputs with intervals

    # ---- States: assume [v, gamma, h] if nx >= 3; otherwise, plot generically.
    fig1 = plt.figure(figsize=(10, 7))
    nrows = min(3, X.shape[1])
    for i in range(X.shape[1]):
        ax = plt.subplot(nrows, 1, min(i+1, nrows))
        y = X[:, i].copy()
        label = f"x[{i}]"
        if i == 0: label = "v [m/s]"
        if i == 1:
            label = "gamma [deg]" if gamma_in_deg else "gamma [rad]"
            if gamma_in_deg:
                y = np.rad2deg(y)
        if i == 2: label = "h [m]"
        ax.plot(tX, y)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if i == nrows-1:
            ax.set_xlabel("time [s]")
        if i+1 >= nrows: break
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_states.png", dpi=160)

    # ---- Inputs
    if U.size > 0:
        fig2 = plt.figure(figsize=(10, 5))
        nrows_u = min(3, U.shape[1]) if U.ndim == 2 else 1
        for j in range(U.shape[1]):
            ax = plt.subplot(nrows_u, 1, min(j+1, nrows_u))
            uj = U[:, j]
            lbl = f"u[{j}]"
            # Friendly labels if (alpha, throttle)
            if U.shape[1] >= 1 and j == 0:
                lbl = "alpha [deg]"
                uj = np.rad2deg(uj)
            if U.shape[1] >= 2 and j == 1:
                lbl = "throttle [-]"
            ax.plot(tU, uj)
            ax.set_ylabel(lbl)
            ax.grid(alpha=0.3)
            if j == nrows_u-1:
                ax.set_xlabel("time [s]")
            if j+1 >= nrows_u: break
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_inputs.png", dpi=160)

    if show:
        plt.show()
    else:
        plt.close("all")


def _plot_traj_on_axes(
    traj,
    ax_states,
    ax_inputs,
    *,
    label: str = "",
    gamma_in_deg: bool = True,
):
    tX = traj.t
    X  = traj.X.Y
    U  = traj.U.Y
    nx = X.shape[1]
    tU = traj.t[:-1]

    # states: assume [v, gamma, h]
    # v
    if nx >= 1:
        ax_states[0].plot(tX, X[:, 0], label=label)
    # gamma
    if nx >= 2:
        y = X[:, 1]
        if gamma_in_deg:
            y = np.rad2deg(y)
        ax_states[1].plot(tX, y, label=label)
    # h
    if nx >= 3:
        ax_states[2].plot(tX, X[:, 2], label=label)

    # inputs: assume [alpha, throttle]
    if U.size > 0:
        nu = U.shape[1]
        if nu >= 1:
            ax_inputs[0].plot(tU, np.rad2deg(U[:, 0]), label=label)
        if nu >= 2:
            ax_inputs[1].plot(tU, U[:, 1], label=label)


def plot_trajectories(
    trajs: Sequence,
    *,
    labels: Optional[Sequence[str]] = None,
    gamma_in_deg: bool = True,
    show: bool = True,
    save_prefix: Optional[str] = None,
):
    if labels is None:
        labels = [f"traj_{i}" for i in range(len(trajs))]

    # make shared figures/axes
    fig_s, axs_s = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig_u, axs_u = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # pretty labels/grids
    axs_s[0].set_ylabel("v [m/s]")
    axs_s[1].set_ylabel("gamma [deg]" if gamma_in_deg else "gamma [rad]")
    axs_s[2].set_ylabel("h [m]")
    for ax in axs_s:
        ax.grid(alpha=0.3)
    axs_s[-1].set_xlabel("time [s]")

    axs_u[0].set_ylabel("alpha [deg]")
    axs_u[1].set_ylabel("throttle [-]")
    for ax in axs_u:
        ax.grid(alpha=0.3)
    axs_u[-1].set_xlabel("time [s]")

    # plot all trajs on the same axes
    for traj, lbl in zip(trajs, labels):
        _plot_traj_on_axes(
            traj,
            axs_s,
            axs_u,
            label=lbl,
            gamma_in_deg=gamma_in_deg,
        )

    # one legend on top state plot + on top input plot
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

def plot_residual_components(res: ResidualData, dt: float, state_labels=None):
    N, nx = res.Ek1.shape
    t_k1 = np.arange(1, N+1) * dt

    if state_labels is None:
        state_labels = [f"x{i}" for i in range(nx)]

    for i, label in enumerate(state_labels):
        plt.figure()
        plt.plot(t_k1, res.Ek1[:, i])
        plt.xlabel("time [s]")
        plt.ylabel(f"e_{i+1} in {label}")
        plt.title(f"One-step residual in {label}")
        plt.grid(True)
        plt.show()

def plot_residual_norm(res: ResidualData, label: str):
    err_norm = np.linalg.norm(res.Ek1, axis=1)
    plt.plot(res.t_k + (res.t_k[1] - res.t_k[0]), err_norm, label=label)

def plot_residual_components_multi(
    res_list: Sequence["ResidualData"],
    dt: float,
    labels: Sequence[str] | None = None,
    state_labels: Sequence[str] | None = None,
):
    """
    For each state component i:
      plot e_{k+1,i} for all residual datasets on the same figure.

    res_list     : list of ResidualData
    dt           : timestep
    labels       : labels for each ResidualData (same length as res_list)
    state_labels : labels for state components (length = nx)
    """
    if not res_list:
        return

    N, nx = res_list[0].Ek1.shape

    if labels is None:
        labels = [f"model{i}" for i in range(len(res_list))]
    assert len(labels) == len(res_list), "labels must match res_list length"

    if state_labels is None:
        state_labels = [f"x{i}" for i in range(nx)]
    assert len(state_labels) == nx, "state_labels must match state dimension"

    t_k1 = np.arange(1, N+1) * dt

    for i, state_lbl in enumerate(state_labels):
        plt.figure()

        for res, lab in zip(res_list, labels):
            plt.plot(t_k1, res.Ek1[:, i], label=lab)

        plt.xlabel("time [s]")
        plt.ylabel(f"e in {state_lbl}")
        plt.title(f"One-step residual in {state_lbl}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()