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

## Features

- **LTI-MPC Controller**: Linear Time-Invariant MPC using CVXPY/OSQP, with incremental input (ΔU) formulation and warm-starting for real-time performance (~60Hz)
- **Nonlinear Physics Engine**: Full nonlinear cart-pole dynamics with RK4 integration
- **Reference Tracking**: Three tracking modes — Static (position hold), Cruise (constant velocity), Accel (constant acceleration with differential flatness)
- **Hard & Soft Constraints**: Actuator limits (hard), state bounds with slack variables (soft), and control rate limiting
- **Dynamic Constraint Shifting**: Constraint bounds adapt to current reference for non-zero tracking
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
