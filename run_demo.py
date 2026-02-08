
import numpy as np
import matplotlib.pyplot as plt
from dynamic_models import f_true_ct, f_nom_ct, eval_nominal, eval_true
from simulate import stage_cost, terminal_cost
from core.structs import Trajectory
from models.continuous_model import ContinuousModel
from models.discrete_model import DiscreteModel

from visuals import plot_tvlqr_run, plot_aero_trace, plot_trajectory, plot_trajectories, plot_residual_components_multi
from diagnostics.feasibility import FeasibilityConstraints, feasibility_score, feasibility_report
from warm_start.dispatcher import warm_start_U
from control import feasible_control_iterative, make_Qff
from learning.residuals import collect_residuals
from core.reference_builder import accelerate_with_push_over_reference
from learning.polynomial_residual_model import PolynomialResidualModel
from learning.rkhs_residual_model import RKHSResidualModel
from models.learned_model import ResidualAugmentedModel
from controller.tvlqr_controller import TVLQR
np.set_printoptions(precision=3, suppress=True)
from simulator import Simulator

def main():
    # ---------------- Params ----------------
    dt = 0.05
    
    
    P_N = np.diag([1.0, 5.0, 0.1])

    # Flight params (true and nominal with mismatch)
    p_nom = dict(rho=1.225, S=16.2, m=1200.0, g=9.81,
                 CL0=0.15, CL_alpha=4.0, CD0=0.02, eta=0.05)
    p_true = dict(rho=1.225, S=16.2, m=1200.0, g=9.81,
                  CL0=0.2, CL_alpha=5.0, CD0=0.02, eta0=0.04, eta_M_slope = 0.12)
    
    nom_model = ContinuousModel(
        nx=3,
        nu=2,
        params=p_nom,
        f_ct=f_nom_ct,
        dyn_eval=eval_nominal
    )

    true_model = ContinuousModel(
        nx=3,
        nu=2,
        params=p_nom,
        f_ct=f_true_ct,
        dyn_eval=eval_true
    )


    u_min = np.array([np.deg2rad(-10), 0.0])
    u_max = np.array([np.deg2rad(+10), 1.0])
    x0 = np.array([150.0, 0.0, 2000.0])
    dt = 0.05


    ref = accelerate_with_push_over_reference(
        x0, dt, nom_model,
        v_switch=270.0, v_peak=320.0, v_final=300.0,
        gamma_push_deg=-1.5,       # tweak: -1.0° .. -2.0°
        max_dv_margin=0.6,
        max_ddv_margin=0.8,
        decel_margin=0.8,
        alpha_min_deg=-6, alpha_max_deg=8,
        thr_min=0.15, thr_max=1.0,
    )

    # constraints (examples)
    du_max = np.array([np.deg2rad(2.0), 0.35])
    cons = FeasibilityConstraints(
        u_min=u_min, u_max=u_max,
        xN_target=ref.X.Y[-1], xN_tol=2.0  # e.g., 2 m terminal tolerance
    )
    nom_discrete_model = DiscreteModel(nom_model, t = ref.t, method = "euler")
    true_discrete_model = DiscreteModel(true_model, t = ref.t, method = "euler")
    
    f_true = true_discrete_model.f_dt
    f_nom = nom_discrete_model.f_dt


    # evaluate a trajectory (raw ref, feasible ref, rollout, etc.)

    
    U0 = warm_start_U(ref, nom_model, method = "trim", u_min=u_min, u_max=u_max,trim='rate')
    ref = ref.with_(U=U0) 
    

    ref = nom_discrete_model.rollout_trajectory(ref)

    rep = feasibility_report(ref, f_nom, cons, defect_tol=1e-3, defect_metric="p95")
    print("Feasible?", rep["feasible"])
    print(f"defect_p95={rep['defect_p95']:.3e}, u_max_violation={rep.get('u_max_violation_max',0):.3e}")

    # optional scalar score (for model/method bake-off)
    print("feasibility_score:", feasibility_score(rep))

    plot_trajectory(ref, show=True, save_prefix="results/ref")
    
    dt_avg = np.mean(np.diff(ref.t)) 
    Q = make_Qff(dt_avg, w_v=1.0, w_gam=10.0, w_h_per_sec=5.0)


    # optional scalar score (for model/method bake-off)
    print("feasibility_score:", feasibility_score(rep))
    # or iterative:
    ref_feas = feasible_control_iterative(ref, f_nom, Qff=Q, u_min=u_min, u_max=u_max, max_iters=2)
    rep = feasibility_report(ref_feas, f_nom, cons, defect_tol=1e-3, defect_metric="p95")
    print("Feasible?", rep["feasible"])
    print(f"defect_p95={rep['defect_p95']:.3e}, u_max_violation={rep.get('u_max_violation_max',0):.3e}")


    # --------------- TV-LQR gains -----------
    # tolerances you care about
    dv_tol     = 2.0      # m/s
    dgamma_tol = 0.01     # rad  (~0.57 deg)
    dh_tol     = 8.0      # m over the horizon

    Q = np.diag([
        1.0/(dv_tol**2),           # v weight
        8.0/(dgamma_tol**2),       # gamma weight (8x boost)
        1.0/(dh_tol**2)            # h weight (small per-step)
    ])

    # inputs: keep small unless you see chattering
    R = np.diag([np.deg2rad(1.0)**-2,   # alpha penalty ~ 1 / (1 deg)^2
                (0.15)**-2])          # throttle penalty for ~15% change`
    P_N = Q

    # --------------- Nominal control on true dynamics (collect residuals) ---------------
    # 1. Build controller
    tvlqr = TVLQR(nom_discrete_model, Q, R)
    tvlqr.compute_gains(ref_feas)

    # 2. Build simulator
    sim = Simulator(true_discrete_model)

    # 3. Run closed-loop simulation
    ref_true = sim.simulate_tvlqr(x0, tvlqr)

    # Collect training data
    res = collect_residuals(nom_discrete_model, true_discrete_model, ref_true)
    X_data, U_data, E_data = res.as_training_data()

    # Fit polynomial residual model
    res_model = PolynomialResidualModel(degree=2)
    res_model.fit(X_data, U_data, E_data)

    # Build hybrid model
    discrete_model_poly = ResidualAugmentedModel(nom_discrete_model, res_model)

    tvlqr_poly = TVLQR(discrete_model_poly, Q, R)
    tvlqr_poly.compute_gains(ref_feas)


    # Use in simulation or control
    sim_poly = Simulator(discrete_model_poly)
    ref_true_poly = sim_poly.simulate_tvlqr(x0, tvlqr_poly)

    res_poly = collect_residuals(discrete_model_poly, true_discrete_model, ref_true_poly)


    # ... after you have X_true and U_applied from simulating on the true plant ...
    

    ### Residual Augmented model with just RKHS
    
    # 1. Collect residuals
    # 2. Fit RKHS residual model
    rkhs = RKHSResidualModel(lengthscale=0.5, reg=1e-4)
    rkhs.fit(X_data, U_data, E_data)

    # 3. Build hybrid model
    discrete_model_rkhs = ResidualAugmentedModel(nom_discrete_model, rkhs)

    # 4. Use in simulation / control
    sim_rkhs = Simulator(discrete_model_rkhs)

    tvlqr_rkhs = TVLQR(discrete_model_rkhs, Q, R)
    tvlqr_rkhs.compute_gains(ref_feas)
    ref_true_rkhs = sim_rkhs.simulate_tvlqr(x0, tvlqr_rkhs)


    res_rkhs = collect_residuals(discrete_model_rkhs, true_discrete_model, ref_true_rkhs)

    ## Next step: Do a residual model with RKHS after first fitting and collecting and simulation residual augmented model with polynomial model.

    X2_data, U2_data, E2_data = res_poly.as_training_data()

    res_rkhs_model = RKHSResidualModel(lengthscale=0.1, reg=1e-1)
    res_rkhs_model.fit(X2_data, U2_data, E2_data)

    discrete_model_hybrid = ResidualAugmentedModel(discrete_model_poly, res_rkhs_model)

    tvlqr_hybrid = TVLQR(discrete_model_hybrid, Q, R)
    tvlqr_hybrid.compute_gains(ref_feas)
    ref_true_hybrid = sim_rkhs.simulate_tvlqr(x0, tvlqr_hybrid)

    res_hybrid = collect_residuals(discrete_model_hybrid, true_discrete_model, ref_true_hybrid)


    plt.figure(figsize=(10,6))

    plt.plot(res_poly.Ek1.t, res_poly.l2_norm(), label="Polynomial Residual")
    plt.plot(res_rkhs.Ek1.t, res_rkhs.l2_norm(), label="RKHS Residual")
    plt.plot(res_hybrid.Ek1.t, res_hybrid.l2_norm(), label="Hybrid (Poly + RKHS) Residual")

    plt.xlabel("Time")
    plt.ylabel("L2 Norm of Residual")
    plt.title("Residual Comparison Across Models")
    plt.legend()
    plt.grid(True)
    plt.show()

    # print(f"Accumulated absolute cost gap (placeholder Lipschitz constants): {cost_gap+term_gap:.3f}")
    # print(f"Simple bound using delta bounds (with L_ell=L_Vf=1): {bound_cost:.3f}")

    # print("Saved plot: prediction_bound.png")

if __name__ == '__main__':
    main()
