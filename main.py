import time
import sys
import numpy as np
from src.visualization.mpc_panel import show_mpc_panel
from src.logger import save_log, save_pdf
from src.config import PhysicalParams, MPCTuning, ConstraintLimits, DisturbanceConfig


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def build_config(params):
    """
    Constructs structured dataclass configuration objects from the GUI panel dict.
    """
    physics = PhysicalParams()  # Immutable physical constants use defaults
    
    tuning = MPCTuning(
        dt=params["dt"],
        Q=np.diag(params["q_diag"]),
        R=np.array([[params["r_val"]]]),
        S=np.diag(params["s_diag"]),
        Np=params["Np"],
        auto_dare=params.get("auto_dare", True),
    )
    
    constraints = ConstraintLimits(
        u_max=params["u_max"],
        x_max=params["x_max"],
        q_max=params["q_max"],
        xdot_max=params["xdot_max"],
        qdot_max=params["qdot_max"],
        du_max=params["du_max"],
        rho_slack=params["rho_slack"],
    )
    
    disturbance = DisturbanceConfig(
        noise_mean=params["noise_mean"],
        noise_peak=params["noise_peak"],
        random_seed=params["random_seed"],
        disturbances=params["disturbances"],
    )
    
    return physics, tuning, constraints, disturbance


def run_simulation(gui_params):

    physics, tuning, constraints, disturbance = build_config(gui_params)
    
    from src.physics.plant import CartPolePlant
    from src.visualization.gui import CartPoleGUI
    from src.controllers.mpc_core import MPCController
    from src.reference.governor import ReferenceGovernor
    
    dt = tuning.dt
    t_final = gui_params["t_final"]
    total_steps = int(t_final / dt)
    

    plant = CartPolePlant(state_init=gui_params["x0"], physics=physics)
    gui = CartPoleGUI(physics=physics)
    gui.show_ghost = gui_params["ghost_trajectory"]
    

    controller = MPCController(physics=physics, tuning=tuning, constraints=constraints)
    
    tracking_mode = gui_params["tracking_mode"]
    tracking_val = gui_params["tracking_val"]
    

    u_prev = 0.0
    

    t_history = np.zeros(total_steps)
    states_history = np.zeros((total_steps, 4))
    forces_history = np.zeros(total_steps)
    u_mpc_history = np.zeros(total_steps)
    u_ff_history = np.zeros(total_steps)
    solve_time_history = np.zeros(total_steps)
    x_d_history = np.zeros((total_steps, 4))
    f_noise_history = np.zeros(total_steps)
    f_dist_history = np.zeros(total_steps)
    f_ext_history = np.zeros(total_steps)
    
    terminal_tol = np.array([0.01, 0.017, 0.05, 0.05])
    settle_time = None
    settle_step = None
    

    noise_mean = disturbance.noise_mean
    noise_peak = disturbance.noise_peak
    disturbances = disturbance.disturbances
    rng_seed = disturbance.random_seed
    rng = np.random.default_rng(rng_seed)
    

    gov = ReferenceGovernor(mode=tracking_mode, physics=physics, q_max=constraints.q_max)
    gov.target_val = tracking_val
    gov.start_pos = gui_params["x0"][0]
    gov.start_vel = gui_params["x0"][2]
    

    print(f"\n--- Starting {tracking_mode} Mode ({t_final}s, Np={tuning.Np}, dt={dt}) ---")
    startTime_real = time.time()
    
    for step in range(total_steps):
        current_time = step * dt
        

        x_actual = plant.state
        states_history[step] = x_actual
        

        x_ref, u_ff = gov.step_reference(current_time)
            
        x_d_history[step] = x_ref
        

        error = x_actual - x_ref
        u_prev_error = u_prev - u_ff
        e_augmented = np.concatenate([error, [u_prev_error]])
        

        start_solve = time.time()
        delta_u_optimal, ghost_states = controller.compute_action(e_augmented, current_reference=x_ref)
        solve_t = time.time() - start_solve
        solve_time_history[step] = solve_t
        

        u_total = u_prev + delta_u_optimal
        u_mpc_history[step] = delta_u_optimal
        u_ff_history[step] = u_ff
        forces_history[step] = u_total
        t_history[step] = current_time
        

        if settle_time is None:
            if np.all(np.abs(x_actual - x_ref) <= terminal_tol):
                settle_time = current_time
                settle_step = step
        

        f_noise = noise_mean + rng.uniform(-noise_peak, noise_peak)
        f_dist = sum(d["force"] for d in disturbances
                     if d["t_start"] <= current_time < d["t_end"])
        f_ext = f_noise + f_dist
        
        f_noise_history[step] = f_noise
        f_dist_history[step] = f_dist
        f_ext_history[step] = f_ext
        

        plant.step(u_total + f_ext, dt)
        u_prev = u_total
        

        gui.render(
            plant.state, current_time,
            force=u_total,
            target_state=x_ref,
            settle_time=settle_time,
            ghost_states=ghost_states,
            f_noise=f_noise,
            f_dist=f_dist
        )

        time.sleep(0.005)

    real_time_taken = time.time() - startTime_real
    

    final_state = states_history[-1]
    x0 = states_history[0]
    
    print(f"\n{'='*60}")
    print(f"Simulation Finished! {total_steps} steps in {real_time_taken:.2f}s real time.")
    print(f"Terminal State: x={final_state[0]:.4f}m, q={np.degrees(final_state[1]):.2f}°, "
          f"dx={final_state[2]:.4f}m/s, dq={final_state[3]:.4f}r/s")
    print(f"{'='*60}")
    

    x_error0 = x0[0] - x_d_history[0, 0]
    x_sign = np.sign(x_error0) if x_error0 != 0 else 1.0
    q_error0 = x0[1] - x_d_history[0, 1]
    q_sign = np.sign(q_error0) if q_error0 != 0 else 1.0
    
    x_opposite = (states_history[:, 0] - x_d_history[:, 0]) * (-x_sign)
    q_opposite = (states_history[:, 1] - x_d_history[:, 1]) * (-q_sign)
    x_overshoot = max(np.max(x_opposite), 0.0) if len(x_opposite) > 0 else 0.0
    q_overshoot = max(np.max(q_opposite), 0.0) if len(q_opposite) > 0 else 0.0
    
    x_peak = np.max(np.abs(states_history[:, 0] - x_d_history[:, 0]))
    q_peak = np.max(np.abs(states_history[:, 1] - x_d_history[:, 1]))
    u_peak = np.max(np.abs(forces_history))
    energy = np.sum(forces_history**2) * dt
    cost = sum(
        ((s - xd) @ tuning.Q @ (s - xd) + f * tuning.R[0, 0] * f) * dt
        for s, xd, f in zip(states_history, x_d_history, forces_history)
    )
    
    print(f"\n--- Metrics ---")
    settle_str = f"{settle_time:.3f} s" if settle_time is not None else "N/A"
    print(f"  Settle Time (ts):   {settle_str}")
    print(f"  Peak X Error:       {x_peak:.4f} m")
    print(f"  Peak Q Error:       {np.degrees(q_peak):.2f} °")
    print(f"  Peak Force:         {u_peak:.2f} N")
    print(f"  Control Energy:     {energy:.4f}")
    

    soft_limits = [constraints.x_max, constraints.q_max, constraints.xdot_max, constraints.qdot_max]
    soft_labels = ['x', 'q', 'x_dot', 'q_dot']
    any_soft_violation = False
    
    print(f"\n--- Constraints ---")
    print(f"  [Hard] F_max = {u_peak:.2f} / {constraints.u_max:.1f} N")
    for i, (lbl, limit) in enumerate(zip(soft_labels, soft_limits)):
        peak = np.max(np.abs(states_history[:, i]))
        if peak > limit:
            any_soft_violation = True
            print(f"  [Soft] |{lbl}|_max = {peak:.4f} / {limit} (VIOLATION! slack triggered)")
        else:
            print(f"  [Soft] |{lbl}|_max = {peak:.4f} / {limit}")
            


    if gui_params["save_log"]:
        actual_s_diag = controller.S.diagonal().tolist() if hasattr(controller, 'S') else gui_params["s_diag"]
        
        log_data = {
            **gui_params,
            "s_diag": actual_s_diag,
            "settle_time": settle_time,
            "x_peak": x_peak,
            "q_peak": q_peak,
            "x_overshoot": x_overshoot,
            "theta_overshoot": q_overshoot,
            "u_peak": u_peak,
            "energy": energy,
            "cost": cost,
            "t_history": t_history,
            "states_history": states_history,
            "forces_history": forces_history,
            "u_mpc_history": u_mpc_history,
            "u_ff_history": u_ff_history,
            "solve_time_history": solve_time_history,
            "x_d_history": x_d_history,
            "f_noise_history": f_noise_history,
            "f_dist_history": f_dist_history,
            "f_ext_history": f_ext_history,
        }
        filepath = save_log(log_data)
        print(f"\n[Log Saved]: {filepath}")
        pdf_path = save_pdf(log_data, filepath)
        print(f"[PDF Saved]: {pdf_path}")


if __name__ == "__main__":
    print("Initializing GUI Parameter Panel...")
    gui_params = show_mpc_panel()
    
    if gui_params is not None:
        run_simulation(gui_params)
    else:
        print("Panel closed. Simulation aborted.")
