from .trim import warm_start_trim_U
from .gauss_newton import warm_start_gauss_newton_U

def warm_start_U(
    traj,
    model,
    *,
    method="trim",
    trim="rate",
    Qff=None,
    rho=1e-3,
    u_min=None,
    u_max=None,
    du_clip=0.05,
    reuse_previous=True,
):
    if method == "trim":
        return warm_start_trim_U(
            traj, model,
            reuse_previous=reuse_previous,
            u_min=u_min,
            u_max=u_max,
            trim=trim,
        )

    elif method == "gauss_newton":
        return warm_start_gauss_newton_U(
            traj, model,
            Qff=Qff,
            rho=rho,
            u_min=u_min,
            u_max=u_max,
            du_clip=du_clip,
        )

    raise ValueError(f"Unknown warm-start method: {method}")
