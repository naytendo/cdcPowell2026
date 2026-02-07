from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

ArrF = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class Signal:
    t: ArrF          # shape (N,)
    Y: ArrF          # shape (N, dim)
    dt: float
    labels: tuple[str, ...] | None = None

    # -----------------------------
    # Basic properties
    # -----------------------------
    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def dim(self) -> int:
        return self.Y.shape[1]

    # -----------------------------
    # Indexing and slicing
    # -----------------------------
    def __getitem__(self, idx):
        """Return a sliced Signal."""
        return Signal(
            t=self.t[idx],
            Y=self.Y[idx],
            dt=self.dt,
            labels=self.labels,
        )

    # -----------------------------
    # Interpolation
    # -----------------------------
    def at(self, t_query: float | ArrF, *, method: str = "interp") -> ArrF:
        """
        Return the signal value(s) at time t_query.

        Parameters
        ----------
        t_query : float or array-like
            Time(s) at which to evaluate the signal.
        method : {"interp", "nearest"}
            - "interp": linear interpolation across each dimension.
            - "nearest": return the sample closest in time.

        Returns
        -------
        Yq : array
            Interpolated or nearest-neighbor values.
            Shape (dim,) for scalar t_query, or (len(t_query), dim) for array input.
        """
        t_query = np.atleast_1d(t_query)

        if method == "nearest":
            # nearest neighbor index for each query time
            idxs = np.array([np.argmin(np.abs(self.t - tq)) for tq in t_query])
            Yq = self.Y[idxs]

        elif method == "interp":
            # linear interpolation for each dimension
            Yq = np.vstack([
                np.interp(t_query, self.t, self.Y[:, j])
                for j in range(self.dim)
            ]).T

        else:
            raise ValueError("method must be 'interp' or 'nearest'")

        # unwrap scalar input
        return Yq[0] if Yq.shape[0] == 1 else Yq


    # -----------------------------
    # Plotting
    # -----------------------------
    def plot(self, *, show=True, save_prefix=None):
        fig, axs = plt.subplots(self.dim, 1, figsize=(10, 2.5*self.dim), sharex=True)

        if self.dim == 1:
            axs = [axs]

        for i in range(self.dim):
            axs[i].plot(self.t, self.Y[:, i])
            label = self.labels[i] if self.labels else f"Y[{i}]"
            axs[i].set_ylabel(label)
            axs[i].grid(alpha=0.3)

        axs[-1].set_xlabel("time [s]")
        fig.tight_layout()

        if save_prefix:
            fig.savefig(f"{save_prefix}.png", dpi=160)

        if show:
            plt.show()
        else:
            plt.close(fig)
