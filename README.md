# Cart-Pole MPC Controller

Model Predictive Control (MPC) for an inverted pendulum (cart-pole) system.

## System Model

The cart-pole system is defined by a 4-dimensional state vector and a 1-dimensional control input:

**State Vector `x`**: 
- `x` : Cart position (m)
- `q` : Pole angle (rad), where 0 is perfectly upright
- `x_dot` : Cart velocity (m/s)
- `q_dot` : Pole angular velocity (rad/s)

**Control Input `u`**: 
- Horizontal force applied to the cart (N)

### Linearized State-Space Model
The continuous-time LTI dynamics around the unstable equilibrium (upright, $q=0$) are defined as $\dot{x} = A_c x + B_c u$:

$$
A_c = \begin{bmatrix} 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 \\ 
0 & \frac{mg}{M} & 0 & 0 \\ 
0 & \frac{(M+m)g}{ML} & 0 & 0 
\end{bmatrix}, \quad 
B_c = \begin{bmatrix} 
0 \\ 
0 \\ 
\frac{1}{M} \\ 
\frac{1}{ML} 
\end{bmatrix}
$$

The system is discretized using the matrix exponential ($M_{disc} = e^{M_{cont}dt}$) to yield the discrete matrices $A_d, B_d$.

### MPC Optimization Problem
The controller solves a finite-horizon DP problem at each time step using an **incremental input ($\Delta u$) formulation** to easily penalize input rate of change.

$$
\begin{aligned}
\min_{\Delta U, \varepsilon} \quad & \frac{1}{2} \Delta U^T H \Delta U + x_0^T F^T \Delta U + \rho_{slack} \sum \varepsilon^2 \\
\text{s.t.} \quad & x_{k+1} = A_d x_k + B_d u_k \\
& M_{hard} \Delta U \le \beta_{hard} + b_{hard} x_0 \quad \text{(Actuator Limits: } u_{max}, \Delta u_{max}\text{)}\\
& M_{soft} \Delta U \le \beta_{soft} + b_{soft} x_0 + L_\varepsilon \varepsilon \quad \text{(State Limits: } x_{max}, q_{max}, \dot{x}_{max}, \dot{q}_{max}\text{)} \\
& \varepsilon \ge 0 \quad \text{(Slack Non-negativity)}
\end{aligned}
$$
Where the terminal cost $S$ is either a manual heuristic matrix or solved via the DARE oracle: $P = A_d^T P A_d - A_d^T P B_d (R + B_d^T P B_d)^{-1} B_d^T P A_d + Q$.

## Features

- **LTI-MPC Controller**: Linear Time-Invariant MPC using CVXPY/OSQP, with incremental input (ΔU) formulation and warm-starting for real-time performance (~60Hz)
- **Nonlinear Physics Engine**: Full nonlinear cart-pole dynamics with RK4 integration
- **Reference Tracking**: Three tracking modes — Static (position hold), Cruise (constant velocity), Accel (constant acceleration with differential flatness)
- **Hard & Soft Constraints**: Actuator limits (hard), state bounds with slack variables (soft), and control rate limiting
- **Dynamic Constraint Shifting**: Constraint bounds adapt to current reference for non-zero tracking
- **Oracle Terminal Cost (DARE)**: Infinite-horizon terminal cost S=P auto-solved via Discrete Algebraic Riccati Equation
- **Dataclass Architecture**: Structured typed configurations for strict physics/controller parameter boundaries
- **External Disturbance Testing**: Configurable continuous noise and deterministic wind gusts to test robustness
- **Ghost Trajectory Visualization**: Real-time overlay of MPC's predicted future trajectory
- **GUI Parameter Panel**: tkinter-based panel for adjusting all MPC parameters before simulation
- **Real-time Animation**: Matplotlib animation with 7 live tracking curves (x, q, velocities, forces, disturbances)
- **Logging & Export**: Simulation results saved as XLSX + PDF with tracking curves

## Project Structure

```
├── main.py                          # Entry point
├── run.bat                          # Windows launcher
├── pyproject.toml                   # Dependencies
├── src/
│   ├── config.py                    # System parameters & MPC tuning
│   ├── logger.py                    # XLSX + PDF export
│   ├── controllers/
│   │   ├── base_controller.py       # Controller interface
│   │   ├── lqr_core.py              # LQR placeholder
│   │   └── mpc_core.py              # MPC controller (CVXPY)
│   ├── physics/
│   │   └── plant.py                 # Nonlinear dynamics + RK4
│   ├── reference/
│   │   └── governor.py              # Reference generator + safety clamping
│   └── visualization/
│       ├── gui.py                   # Real-time animation + curves
│       └── mpc_panel.py             # Parameter editing panel
└── tests/
    ├── test_plant_open_loop.py      # Open-loop physics test
    ├── test_phase2_headless.py      # Dynamic tracking test
    ├── test_phase3_headless.py      # Robustness test (noise + gusts)
    └── test_pdf_export.py           # Log export test
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run simulation
uv run main.py
# or double-click run.bat on Windows
```

## Requirements

- Python ≥ 3.14
- Dependencies: numpy, scipy, cvxpy, matplotlib, openpyxl
