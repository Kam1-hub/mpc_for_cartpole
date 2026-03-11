"""Phase 2 headless verification test."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from src.physics.plant import CartPolePlant
from src.controllers.mpc_core import MPCController
from src.reference.governor import ReferenceGovernor
from src.config import dt, Q, R

plant = CartPolePlant(state_init=[0.0, 0.1, 0.0, 0.0])
controller = MPCController()
u_prev = 0.0
total_steps = 600
states_history = np.zeros((total_steps, 4))
forces_history = np.zeros(total_steps)
x_d_history = np.zeros((total_steps, 4))
terminal_tol = np.array([0.01, 0.017, 0.05, 0.05])
settle_time = None
settle_step = None

for step in range(total_steps):
    current_time = step * dt
    x_actual = plant.state
    states_history[step] = x_actual

    if current_time < 3.0:
        gov = ReferenceGovernor(mode="Static")
        gov.target_val = 0.0
        x_ref, u_ff = gov.step_reference(current_time)
    elif current_time < 6.0:
        gov = ReferenceGovernor(mode="Cruise")
        gov.start_pos = 0.0
        gov.target_val = 0.15
        x_ref, u_ff = gov.step_reference(current_time - 3.0)
    else:
        gov = ReferenceGovernor(mode="Accel")
        gov.start_pos = 0.45
        gov.start_vel = 0.15
        gov.target_val = -0.15
        x_ref, u_ff = gov.step_reference(current_time - 6.0)

    x_d_history[step] = x_ref
    error = x_actual - x_ref
    u_prev_error = u_prev - u_ff
    e_augmented = np.concatenate([error, [u_prev_error]])
    delta_u_optimal, _ = controller.compute_action(e_augmented, current_reference=x_ref)
    u_total = u_prev + delta_u_optimal
    forces_history[step] = u_total

    if settle_time is None:
        if np.all(np.abs(x_actual - x_ref) <= terminal_tol):
            settle_time = current_time
            settle_step = step

    plant.step(u_total, dt)
    u_prev = u_total


    if step % 100 == 0:
        print(f"  step {step}/{total_steps}  x={x_actual[0]:+.3f}  q={np.degrees(x_actual[1]):+.1f}deg  "
              f"x_ref={x_ref[0]:+.3f}  u={u_total:+.2f}N")

final_state = states_history[-1]
x0 = states_history[0]

print(f"\n{'='*60}")
print(f"Phase 2 Dynamic Tracking Report")
print(f"{'='*60}")
print(f"Total: {total_steps} steps ({total_steps * dt:.1f}s)")
print(f"Final state: x={final_state[0]:+.4f}m  q={np.degrees(final_state[1]):+.2f}deg  "
      f"xdot={final_state[2]:+.4f}m/s  qdot={final_state[3]:+.4f}rad/s")
print(f"Final target: x={x_d_history[-1,0]:+.4f}m  q={np.degrees(x_d_history[-1,1]):+.2f}deg  "
      f"xdot={x_d_history[-1,2]:+.4f}m/s")

if settle_time is not None:
    print(f"Settle time: {settle_time:.3f}s (step {settle_step})")
else:
    print("Settle time: NOT REACHED")

x_peak = np.max(np.abs(states_history[:, 0] - x_d_history[:, 0]))
q_peak = np.max(np.abs(states_history[:, 1] - x_d_history[:, 1]))
print(f"Peak tracking error: x={x_peak:.4f}m  q={np.degrees(q_peak):.2f}deg")
print(f"Peak control force: {np.max(np.abs(forces_history)):.2f}N")
print(f"Control energy: {np.sum(forces_history**2) * dt:.4f}")

from src.config import x_max, q_max, xdot_max, qdot_max, u_max
print(f"\n--- Constraint Report ---")
print(f"|F|_max = {np.max(np.abs(forces_history)):.2f} / {u_max:.1f}N  "
      f"{'OK' if np.max(np.abs(forces_history)) <= u_max + 0.01 else 'VIOLATED'}")
for i, (lbl, lim) in enumerate(zip(["x","q","xdot","qdot"], [x_max,q_max,xdot_max,qdot_max])):
    pk = np.max(np.abs(states_history[:, i]))
    print(f"|{lbl}|_max = {pk:.4f} / {lim}  {'OK' if pk <= lim + 0.01 else 'VIOLATED'}")

print(f"\n--- Terminal Tolerance ---")
labels = ["x", "q", "x_dot", "q_dot"]
all_pass = True
for i, (lbl, tol) in enumerate(zip(labels, terminal_tol)):
    v = abs(final_state[i] - x_d_history[-1][i])
    ok = v < tol
    if not ok:
        all_pass = False
    print(f"  {lbl:6s} error: {v:.6f} / {tol}  {'PASS' if ok else 'FAIL'}")
print(f"  Result: {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
