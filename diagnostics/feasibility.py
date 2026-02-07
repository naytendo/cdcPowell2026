import numpy as np
from typing import Dict
from core.structs import Trajectory
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from .constraints import check_constraints
from .defects import compute_defects

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
        viol = np.maximum(0.0, c.u_min - traj.U.Y) #c.u_min - traj.U > 0 if traj.U < c.u_min
        info["u_min_violation_max"] = float(np.max(viol)) if viol.size else 0.0
    if c.u_max is not None:
        viol = np.maximum(0.0, traj.U.Y - c.u_max) #traj.U - c.u_msx > 0 if traj.U > c.u_max
        info["u_max_violation_max"] = float(np.max(viol)) if viol.size else 0.0

    # Input rate limits
    if c.du_max is not None and traj.N > 1:
        dU = np.diff(traj.U.Y, axis=0)
        viol = np.maximum(0.0, np.abs(dU) - c.du_max)
        info["du_violation_max"] = float(np.max(viol)) if viol.size else 0.0

    # State box
    if c.x_min is not None:
        viol = np.maximum(0.0, c.x_min - traj.X.Y) # same logic for trhe states # but its a vector???
        info["x_min_violation_max"] = float(np.max(viol)) if viol.size else 0.0
    if c.x_max is not None:
        viol = np.maximum(0.0, traj.X.Y - c.x_max)
        info["x_max_violation_max"] = float(np.max(viol)) if viol.size else 0.0

    # Terminal target
    if c.xN_target is not None:
        term_err = traj.X.Y[-1] - c.xN_target
        info["terminal_error_norm"] = float(np.linalg.norm(term_err))
        info["terminal_within_tol"] = (info["terminal_error_norm"] <= c.xN_tol)

    return info

def feasibility_report(
    traj: Trajectory,
    f_dt: Callable[[Vec,Vec],Vec],
    constraints: Optional[FeasibilityConstraints] = None,
    *,
    defect_tol: float = 1e-3,            # tolerance on per-step defect norm
    defect_metric: str = "p95",          # "max" | "mean" | "median" | "p95"
) -> Dict[str, Any]:
    """Return a dict of metrics + overall booleans for feasibility."""
    cons = constraints or FeasibilityConstraints()
    rep: Dict[str, Any] = {}

    # Dynamic defects
    D = compute_defects(traj, f_dt)
    rep["defect_max"]   = float(np.max(D.norms))
    rep["defect_mean"]  = float(np.mean(D.norms))
    rep["defect_median"]= float(np.median(D.norms))
    rep["defect_p95"]   = float(np.percentile(D.norms, 95))
    rep["defect_trace"] = D  # you can remove this if you only want scalars

    metric_value = rep[f"defect_{defect_metric}"]
    rep["dynamics_feasible"] = (metric_value <= defect_tol)

    # Constraints
    cinf = check_constraints(traj, cons)
    rep.update(cinf)

    # Aggregate booleans for constraints that were specified
    cons_flags = []
    if "u_min_violation_max" in rep: cons_flags.append(rep["u_min_violation_max"] <= 0.0)
    if "u_max_violation_max" in rep: cons_flags.append(rep["u_max_violation_max"] <= 0.0)
    if "du_violation_max"   in rep: cons_flags.append(rep["du_violation_max"]   <= 0.0)
    if "x_min_violation_max" in rep: cons_flags.append(rep["x_min_violation_max"] <= 0.0)
    if "x_max_violation_max" in rep: cons_flags.append(rep["x_max_violation_max"] <= 0.0)
    if "terminal_within_tol" in rep: cons_flags.append(bool(rep["terminal_within_tol"]))

    rep["constraints_feasible"] = (all(cons_flags) if cons_flags else True)
    rep["feasible"] = bool(rep["dynamics_feasible"] and rep["constraints_feasible"])
    return rep

def feasibility_score(rep: Dict[str,Any], w_defect=1.0, w_viols=1.0) -> float:
    """
    A soft score: lower is better. 0 means perfectly feasible.
    Combines the chosen defect metric and max constraint violations.
    """
    score = w_defect * rep["defect_p95"]
    for k in ("u_min_violation_max","u_max_violation_max","du_violation_max",
              "x_min_violation_max","x_max_violation_max"):
        if k in rep:
            score += w_viols * float(rep[k])
    if "terminal_error_norm" in rep:
        # add only the *excess* over tolerance
        tol = rep.get("xN_tol", 0.0) if isinstance(rep, FeasibilityConstraints) else 0.0
        score += max(0.0, rep["terminal_error_norm"] - tol)
    return float(score)

