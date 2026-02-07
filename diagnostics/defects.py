import numpy as np
from core.structs import Trajectory
from dataclasses import dataclass
from typing import Callable

Vec = np.ndarray

@dataclass(frozen=True, slots=True)
class DefectTrace:
    t: np.ndarray           # (N,)
    d: np.ndarray           # (N, nx)
    norms: np.ndarray       # (N,)

def compute_defects(traj: Trajectory, f_dt: Callable[[Vec,Vec],Vec]) -> DefectTrace:
    N, nx = traj.N, traj.nx
    d = np.empty((N, nx), dtype=float)
    for k in range(N):
        d[k] = f_dt(traj.X.Y[k], traj.U.Y[k]) - traj.X.Y[k+1]
    norms = np.linalg.norm(d, axis=1)
    return DefectTrace(t=traj.t[:-1], d=d, norms=norms)