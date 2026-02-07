
import numpy as np
import matplotlib.pyplot as plt
from models import f_true_ct, f_nom_ct, eval_nominal, eval_true
from simulate import stage_cost, terminal_cost
from core.structs import Trajectory
from models.continuous_model import ContinuousModel
from models.discrete_model import DiscreteModel

from visuals import plot_tvlqr_run, plot_aero_trace, plot_trajectory, plot_trajectories, plot_residual_components_multi
from diagnostics.feasibility import FeasibilityConstraints, feasibility_score, feasibility_report
from warm_start.dispatcher import warm_start_U
from control import feasible_control_iterative, make_Qff, tvlqr_controller
from simulate_old import simulate_tv_lqr
from learning import collect_residuals
from identifcation import fit_residual_model, predict_residual, batch_features, fit_residual_rkhs
from bounds import finite_horizon_cost_gap_bound
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
    nom_discrete_model = DiscreteModel(nom_model, dt = dt, method = "euler")
    true_discrete_model = DiscreteModel(true_model, dt = dt, method = "euler")
    f_nom_dt =  nom_discrete_model.f_dt
    f_true_dt = true_discrete_model.f_dt
    f_nom = lambda x,u: f_nom_dt(x,u,p_nom)
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
    
    Q = make_Qff(ref.dt, w_v=1.0, w_gam=10.0, w_h_per_sec=5.0)


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

    X2, U2, E2 = res_poly.as_training_data()

    res_rkhs2 = RKHSResidualModel(lengthscale=0.5, reg=1e-4)
    res_rkhs2.fit(X2, U2, E2)

    discrete_model_hybrid = ResidualAugmentedModel(discrete_model_poly, res_rkhs)

    tvlqr_hybrid = TVLQR(discrete_model_hybrid, Q, R)
    tvlqr_hybrid.compute_gains(ref_feas)
    ref_true_hybrid = sim_rkhs.simulate_tvlqr(x0, tvlqr_hybrid)




    # 4) epsilon_1 for the *hybrid* model
    hybrid_eps1 = float(np.max(np.linalg.norm(Eval - Eval_hybrid, axis=1)))
    print(f"Hybrid (poly+RKHS) epsilon_1: {hybrid_eps1:.4f}")

    # --------------- Lipschitz-ish constants ---------------
    # Sample a small grid around validation to estimate L
    Xs = Xval + 0.01*np.random.randn(*Xval.shape)
    Us = Uval + 0.01*np.random.randn(*Uval.shape)
    # L_nom = estimate_L_nom(f_nom_dt, Xs, Us, p_nom, eps=1e-5)
    # L_res = estimate_L_residual(W, phi, Xs, Us, eps=1e-5)
    # L = L_nom + L_res
    # print(f"Estimated L_nom={L_nom:.3f}, L_residual={L_res:.3f}, L_total≈{L:.3f}")

    # --------------- Multi-step prediction check ---------------
    # Build augmented predictor
    # --- learned predictors ---------------------------------
    def f_hat_poly_dt(x, u, p_nom):
        return f_nom_dt(x, u, p_nom) + (phi(x, u) @ W)

    def f_hat_hybrid_dt(x, u, p_nom):
        return f_nom_dt(x, u, p_nom) +(phi(x, u) @ W) + rkhs.predict_one(x, u,p_nom)
    
    def f_hat_rkhs_dt(x, u, p_nom):
        return f_nom_dt(x, u, p_nom) + rkhs.predict_one(x, u,p_nom)
    

    #One-step prediction sanity check
    err_nom   = []
    err_poly  = []
    err_hybrid = []

    err_nom_state   = []   # optional, per-component if you want
    err_poly_state  = []
    err_hybrid_state = []

    for k in range(len(ref_true.U)):
        xk   = ref_true.X[k]
        uk   = ref_true.U[k]
        xkp1_true = ref_true.X[k+1]          # or f_true_dt(xk, uk, p_true)

        # Nominal prediction
        xkp1_nom = f_nom_dt(xk, uk, p_nom)

        # Polynomial-corrected prediction
        xkp1_poly = f_hat_poly_dt(xk, uk,p_nom)

        # Polynomial + RKHS (hybrid) prediction
        xkp1_hybrid = f_hat_hybrid_dt(xk, uk,p_nom)

        # L2 norm errors
        err_nom.append(np.linalg.norm(xkp1_true - xkp1_nom))
        err_poly.append(np.linalg.norm(xkp1_true - xkp1_poly))
        err_hybrid.append(np.linalg.norm(xkp1_true - xkp1_hybrid))

        # Optional: store per-state-component error
        err_nom_state.append(xkp1_true - xkp1_nom)
        err_poly_state.append(xkp1_true - xkp1_poly)
        err_hybrid_state.append(xkp1_true - xkp1_hybrid)

    err_nom   = np.array(err_nom)
    err_poly  = np.array(err_poly)
    err_hybrid = np.array(err_hybrid)

    err_nom_state   = np.vstack(err_nom_state)      # shape (N, nx)
    err_poly_state  = np.vstack(err_poly_state)
    err_hybrid_state = np.vstack(err_hybrid_state)

    t = np.arange(len(ref_true.U)) * dt

    


    K_seq_poly, A_seq_poly, B_seq_poly = tvlqr_controller(ref_nom, f_hat_poly_dt, p_nom, Q, R, P_N)
    K_seq_hybrid,A_seq_hybrid, B_seq_hybrid = tvlqr_controller(ref_nom, f_hat_poly_dt, p_nom, Q, R, P_N)

    # --------------- Polynomial residual estimate control on true dynamics (collect residuals) ---------------
    X_true_poly, U_applied_poly, cost_true = simulate_tv_lqr(f_true_dt, x0, ref_nom.X, ref_nom.U, K_seq_poly, p_true, Q, R, P_N)
    ref_true_poly = Trajectory(t=ref_nom.t, X=X_true_poly, U=U_applied_poly, dt=float(dt))
    # plot_tvlqr_run(
    #     ref_true_poly.X, ref_nom.X, ref_true_poly.U, ref_nom.U, E, dt,
    #     state_labels=[r"$v$ [m/s]", r"$gamma$ [rad]", r"$h$ [m]"],
    #     control_labels=[r"$alpha$ [rad]", "throttle"],
    #     save=True, prefix="demo"
    #  )
    
    res_poly = ref_true_poly.get_residuals(f_hat_poly_dt,p_nom)
    # --------------- Hybrid residual estimate control on true dynamics (collect residuals) ---------------
    X_true_hybrid, U_applied_hybrid, cost_true = simulate_tv_lqr(f_true_dt, x0, ref_nom.X, ref_nom.U, K_seq_hybrid, p_true, Q, R, P_N)
    ref_true_hybrid = Trajectory(t=ref_nom.t, X=X_true_hybrid, U=U_applied_hybrid, dt=float(dt))
    
    # plot_tvlqr_run(
    #     ref_true_hybrid.X, ref_nom.X, ref_true_hybrid.U, ref_nom.U, E, dt,
    #     state_labels=[r"$v$ [m/s]", r"$gamma$ [rad]", r"$h$ [m]"],
    #     control_labels=[r"$alpha$ [rad]", "throttle"],
    #     save=True, prefix="demo"
    #  )
    res_hybrid = ref_true_hybrid.get_residuals(f_hat_hybrid_dt,p_nom)
    ## we need to plot comparison of each of the runs. Right now. It's all on plotted out three times
    resList = [res_poly, res_hybrid]
    plot_residual_components_multi(res_list=resList, dt= dt, labels= ['poly','hybrid'])


    # Rollout one-step predictions from validation initial state
    kmax = min(20, len(Uval))

    # start from the same validation point
    x0_val = [220, 0, 2000]

    x_true_seq  = [x0_val.copy()]
    x_hat_poly  = [x0_val.copy()]
    x_hat_hybrid  = [x0_val.copy()]


    delta_poly  = [0.0]
    delta_hybrid  = [0.0]

    # if you still want a bound, use the MODEL-SPECIFIC eps1
    bounds_poly = [0.0]
    bounds_hybrid = [0.0]

    delta0 = 0.0  # we start from same x



    for k in range(kmax):
        u_k = Uval[k]

        # true next
        x_true_next = f_true_dt(x_true_seq[-1], u_k, p_true)
        x_true_seq.append(x_true_next)

        # poly predictor
        x_poly_next = f_hat_poly_dt(x_hat_poly[-1], u_k,p_nom)
        x_hat_poly.append(x_poly_next)
        err_poly = np.linalg.norm(x_true_next - x_poly_next)
        delta_poly.append(err_poly)
        # bound (if you keep the old global L)
        # bounds_poly.append(k_step_prediction_bound(delta0, poly_eps1, L, k+1))

        # hybrid predictor
        x_hybrid_next = f_hat_hybrid_dt(x_hat_hybrid[-1], u_k,p_nom)
        x_hat_hybrid.append(x_hybrid_next)
        err_hybrid = np.linalg.norm(x_true_next - x_hybrid_next)
        delta_hybrid.append(err_hybrid)
        # bounds_hybrid.append(k_step_prediction_bound(delta0, hybrid_eps1, L, k+1))

    t = np.arange(len(x_true_seq), dtype=float) * dt

    trajTrue = Trajectory(t = t, X = np.array(x_true_seq), U = Uval[:(len(x_true_seq)-1)],  dt = dt)
    trajEstPoly = Trajectory(t = t, X = np.array(x_hat_poly), U = Uval[:(len(x_true_seq)-1)],  dt = dt)
    trajEstHybrid = Trajectory(t = t, X = np.array(x_hat_hybrid), U = Uval[:(len(x_true_seq)-1)],  dt = dt)
    plot_trajectories([trajTrue, trajEstPoly, trajEstHybrid], show=True)

    # Plot
    plt.figure()
    plt.plot(range(kmax+1), delta_poly, marker='o', label='actual |delta_{t+k}|')
    plt.plot(range(kmax+1), delta_hybrid, marker='o', label='actual |delta_{t+k}|')
    #plt.plot(range(kmax+1), bounds, marker='x', linestyle='--', label='bound')
    plt.xlabel('k steps')
    plt.ylabel('prediction error norm')
    plt.title('Multi-step prediction: actual vs bound')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('prediction_bound.png', dpi=160)

    # --------------- Simple cost comparison (nominal vs augmented prediction) ---------------
    # Re-run controller but evaluate stage cost vs reference using the same u on true plant
    # (For a fuller MPC, you'd re-solve with f_hat inside the optimizer. Here we compare predictors.)
    cost_gap = 0.0
    for k in range(len(Uval)):
        cost_true_k = stage_cost(x_true_seq[k], Uval[k], Xref[k], Q, R)
        cost_hat_k  = stage_cost(x_hat_seq[k],  Uval[k], Xref[k], Q, R)
        cost_gap += abs(cost_true_k - cost_hat_k)
    term_gap = abs(terminal_cost(x_true_seq[-1], Xref[len(Uval)], P_N)
                   - terminal_cost(x_hat_seq[-1],  Xref[len(Uval)], P_N))
    bound_cost = finite_horizon_cost_gap_bound(bounds, L_ell=1.0, L_Vf=1.0)  # placeholders
    print(f"Accumulated absolute cost gap (placeholder Lipschitz constants): {cost_gap+term_gap:.3f}")
    print(f"Simple bound using delta bounds (with L_ell=L_Vf=1): {bound_cost:.3f}")

    print("Saved plot: prediction_bound.png")

if __name__ == '__main__':
    main()
