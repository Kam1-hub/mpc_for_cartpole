"""Phase 3 headless verification test — noise and disturbances."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from src.physics.plant import CartPolePlant
from src.controllers.mpc_core import MPCController
from src.reference.governor import ReferenceGovernor
from src.config import dt, Q, R, x_max, q_max, xdot_max, qdot_max, u_max

TOTAL_STEPS = 600   # ~10 seconds
TERMINAL_TOL = np.array([0.01, 0.017, 0.05, 0.05])

def run_scenario(name, noise_mean, noise_peak, disturbances, seed=42):
    rng = np.random.default_rng(seed)
    plant = CartPolePlant(state_init=[0.0, 0.1, 0.0, 0.0])
    controller = MPCController()
    gov = ReferenceGovernor(mode="Static")
    gov.target_val = 0.0

    u_prev = 0.0
    states = np.zeros((TOTAL_STEPS, 4))
    forces = np.zeros(TOTAL_STEPS)
    settle_time = None

    for step in range(TOTAL_STEPS):
        t = step * dt
        x_actual = plant.state
        states[step] = x_actual

        x_ref, u_ff = gov.step_reference(t)
        error = x_actual - x_ref
        u_prev_error = u_prev - u_ff
        e_aug = np.concatenate([error, [u_prev_error]])

        delta_u, _ = controller.compute_action(e_aug, current_reference=x_ref)
        u_total = u_prev + delta_u
        forces[step] = u_total


        if settle_time is None and np.all(np.abs(x_actual - x_ref) <= TERMINAL_TOL):
            settle_time = t


        f_noise = noise_mean + rng.uniform(-noise_peak, noise_peak)
        f_dist = sum(d["force"] for d in disturbances if d["t_start"] <= t < d["t_end"])
        plant.step(u_total + f_noise + f_dist, dt)
        u_prev = u_total

        if step % 100 == 0:
            print(f"  [{name}] step {step:4d}  x={x_actual[0]:+.3f}  "
                  f"q={np.degrees(x_actual[1]):+.1f}°  u={u_total:+.2f}N  "
                  f"f_ext={f_noise + f_dist:+.3f}N")

    final = states[-1]
    x_peak_err = np.max(np.abs(states[:, 0]))
    q_peak_err = np.max(np.abs(states[:, 1]))
    u_peak = np.max(np.abs(forces))

    return {
        "name": name,
        "final": final,
        "settle_time": settle_time,
        "x_peak": x_peak_err,
        "q_peak": q_peak_err,
        "u_peak": u_peak,
        "survived": q_peak_err < np.pi / 4,  # Pole didn't fall over (< 45°)
    }


def print_result(r):
    print(f"\n{'='*60}")
    print(f"Scenario: {r['name']}")
    print(f"{'='*60}")
    f = r["final"]
    print(f"  Final: x={f[0]:+.4f}m  q={np.degrees(f[1]):+.2f}°  "
          f"dx={f[2]:+.4f}m/s  dq={f[3]:+.4f}r/s")
    settle_str = f"{r['settle_time']:.3f}s" if r["settle_time"] else "N/A"
    print(f"  Settle Time:  {settle_str}")
    print(f"  |x|_max:      {r['x_peak']:.4f}m  (limit {x_max})")
    print(f"  |q|_max:      {np.degrees(r['q_peak']):.2f}°  (limit {np.degrees(q_max):.1f}°)")
    print(f"  |F|_max:      {r['u_peak']:.2f}N  (limit {u_max}N)")
    print(f"  Survived:     {'YES ✓' if r['survived'] else 'NO ✗'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Headless Test — Robustness Under External Forces")
    print("=" * 60)


    r1 = run_scenario(
        name="Noise Only (Mean=0, Peak=0.5N)",
        noise_mean=0.0, noise_peak=0.5,
        disturbances=[]
    )
    print_result(r1)


    r2 = run_scenario(
        name="Wind Gust (5N @ 3.0s-3.5s)",
        noise_mean=0.0, noise_peak=0.0,
        disturbances=[{"t_start": 3.0, "t_end": 3.5, "force": 5.0}]
    )
    print_result(r2)


    r3 = run_scenario(
        name="Combined: Noise + Overlapping Gusts",
        noise_mean=0.1, noise_peak=0.3,
        disturbances=[
            {"t_start": 2.0, "t_end": 2.5, "force": 3.0},
            {"t_start": 2.3, "t_end": 3.0, "force": -4.0},   # Overlap → net -1N
            {"t_start": 6.0, "t_end": 6.2, "force": 8.0},    # Strong impulse
        ]
    )
    print_result(r3)


    all_survived = all(r["survived"] for r in [r1, r2, r3])
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL SCENARIOS SURVIVED ✓' if all_survived else 'SOME SCENARIOS FAILED ✗'}")
    print(f"{'='*60}")
