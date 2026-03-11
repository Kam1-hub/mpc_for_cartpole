"""Headless test for PDF export from logger."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import src.config as config


config.dt = 0.02
config.Np = 10
config.Q = np.diag([100, 500, 1, 1])
config.R = np.array([[0.1]])
config.S = np.diag([200, 1000, 1, 1])
config.u_max = 10.0
config.x_max = 2.4
config.q_max = 0.5
config.xdot_max = 5.0
config.qdot_max = 10.0
config.du_max = 5.0
config.rho_slack = 1e4

from src.physics.plant import CartPolePlant
from src.controllers.mpc_core import MPCController
from src.reference.governor import ReferenceGovernor
from src.logger import save_log, save_pdf


def test_pdf_export():
    dt = config.dt
    total_steps = 50
    plant = CartPolePlant(state_init=[0.0, 0.1, 0.0, 0.0])
    controller = MPCController()
    gov = ReferenceGovernor(mode="Static")
    gov.target_val = 0.0

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

    u_prev = 0.0
    rng = np.random.default_rng(42)

    for step in range(total_steps):
        current_time = step * dt
        x_actual = plant.state
        x_ref, u_ff = gov.step_reference(current_time)

        error = x_actual - x_ref
        u_prev_error = u_prev - u_ff
        e_augmented = np.concatenate([error, [u_prev_error]])

        delta_u, _ = controller.compute_action(e_augmented, current_reference=x_ref)
        u_total = u_prev + delta_u

        f_noise = rng.uniform(-0.05, 0.05)
        plant.step(u_total + f_noise, dt)

        t_history[step] = current_time
        states_history[step] = x_actual
        forces_history[step] = u_total
        u_mpc_history[step] = delta_u
        u_ff_history[step] = u_ff
        x_d_history[step] = x_ref
        f_noise_history[step] = f_noise

        u_prev = u_total

    log_data = {
        "tracking_mode": "Static",
        "t_final": total_steps * dt,
        "dt": dt,
        "x0": [0.0, 0.1, 0.0, 0.0],
        "tracking_val": 0.0,
        "q_diag": config.Q.diagonal().tolist(),
        "r_val": config.R[0, 0],
        "s_diag": config.S.diagonal().tolist(),
        "Np": config.Np,
        "u_max": config.u_max,
        "x_max": config.x_max,
        "q_max": config.q_max,
        "xdot_max": config.xdot_max,
        "qdot_max": config.qdot_max,
        "du_max": config.du_max,
        "rho_slack": config.rho_slack,
        "noise_mean": 0.0,
        "noise_peak": 0.05,
        "random_seed": 42,
        "disturbances": [],
        "ghost_trajectory": True,
        "save_log": True,
        "settle_time": 0.5,
        "x_peak": 0.1,
        "q_peak": 0.1,
        "x_overshoot": 0.0,
        "theta_overshoot": 0.0,
        "u_peak": 5.0,
        "energy": 1.0,
        "cost": 2.0,
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


    xlsx_path = save_log(log_data)
    assert os.path.isfile(xlsx_path), f"XLSX not created: {xlsx_path}"
    print(f"[PASS] XLSX created: {xlsx_path}  ({os.path.getsize(xlsx_path)} bytes)")


    pdf_path = save_pdf(log_data, xlsx_path)
    assert os.path.isfile(pdf_path), f"PDF not created: {pdf_path}"

    pdf_size = os.path.getsize(pdf_path)
    assert pdf_size > 1000, f"PDF too small ({pdf_size} bytes), likely corrupt"
    print(f"[PASS] PDF created: {pdf_path}  ({pdf_size} bytes)")


    assert pdf_path == xlsx_path.replace(".xlsx", ".pdf"), "PDF path doesn't match XLSX naming"
    print(f"[PASS] Naming consistent: same base name, different extension")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_pdf_export()
