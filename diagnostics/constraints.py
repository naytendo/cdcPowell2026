import numpy as np
from typing import Dict
from core.structs import Trajectory
from dataclasses import dataclass
from typing import Optional, Dict, Any

Vec = np.ndarray

@dataclass(frozen=True, slots=True)
class FeasibilityConstraints:
    # All are optional; unset = “no constraint”
    u_min: Optional[Vec] = None          # (m,)
    u_max: Optional[Vec] = None          # (m,)
    du_max: Optional[Vec] = None         # (m,) per-step rate limit on |U[k]-U[k-1]|
    x_min: Optional[Vec] = None          # (nx,)
    x_max: Optional[Vec] = None          # (nx,)
    xN_target: Optional[Vec] = None      # (nx,) desired terminal state
    xN_tol: float = 0.0                  # tolerance on terminal state (2-norm)



def check_constraints(traj: Trajectory, c: FeasibilityConstraints) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    # Input box
    if c.u_min is not None:
        viol = np.maximum(0.0, c.u_min - traj.U) #c.u_min - traj.U > 0 if traj.U < c.u_min
        info["u_min_violation_max"] = float(np.max(viol)) if viol.size else 0.0
    if c.u_max is not None:
        viol = np.maximum(0.0, traj.U - c.u_max) #traj.U - c.u_msx > 0 if traj.U > c.u_max
        info["u_max_violation_max"] = float(np.max(viol)) if viol.size else 0.0

    # Input rate limits
    if c.du_max is not None and traj.N > 1:
        dU = np.diff(traj.U, axis=0)
        viol = np.maximum(0.0, np.abs(dU) - c.du_max)
        info["du_violation_max"] = float(np.max(viol)) if viol.size else 0.0

    # State box
    if c.x_min is not None:
        viol = np.maximum(0.0, c.x_min - traj.X) # same logic for trhe states # but its a vector???
        info["x_min_violation_max"] = float(np.max(viol)) if viol.size else 0.0
    if c.x_max is not None:
        viol = np.maximum(0.0, traj.X - c.x_max)
        info["x_max_violation_max"] = float(np.max(viol)) if viol.size else 0.0

    # Terminal target
    if c.xN_target is not None:
        term_err = traj.X[-1] - c.xN_target
        info["terminal_error_norm"] = float(np.linalg.norm(term_err))
        info["terminal_within_tol"] = (info["terminal_error_norm"] <= c.xN_tol)

    return info