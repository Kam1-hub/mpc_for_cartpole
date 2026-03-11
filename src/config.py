from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class PhysicalParams:
    """Immutable physical properties of the cart-pole system."""
    M: float = 1.0       # Mass of the cart (kg)
    m: float = 0.1       # Mass of the pole (kg)
    L: float = 0.3       # Half-length of the pole (m)
    g: float = 9.81      # Gravity (m/s^2)
    b: float = 0.0       # Cart friction coefficient
    c: float = 0.0       # Pivot friction coefficient


@dataclass
class MPCTuning:
    """Tuning parameters for the Model Predictive Controller."""
    dt: float = 0.0167   # Time step (s), 60Hz

    Q: np.ndarray = field(default_factory=lambda: np.diag([
        1.0/(0.5**2),     # x
        1.0/(0.24**2),    # q
        1.0/(0.5**2),     # x_dot
        1.0/(1.0**2)      # q_dot
    ]))

    R: np.ndarray = field(default_factory=lambda: np.array([[1.0/(10.0**2)]]))

    S: np.ndarray = field(default_factory=lambda: np.diag([
        1.0/(0.01**2),    # x
        1.0/(0.017**2),   # q
        1.0/(0.05**2),    # x_dot
        1.0/(0.05**2)     # q_dot
    ]))

    Np: int = 30
    auto_dare: bool = True  # If True, S is overridden by DARE solution in mpc_core


@dataclass
class ConstraintLimits:
    """Hard physical limits and soft constraint parameters."""
    u_max: float = 10.0
    x_max: float = 0.5
    q_max: float = 0.24
    xdot_max: float = 10.0     # Relaxed from 0.5
    qdot_max: float = 10.0     # Relaxed from 1.0
    du_max: float = 2.0
    rho_slack: float = 10000.0


@dataclass
class DisturbanceConfig:
    """Phase 3 external disturbance parameters."""
    noise_mean: float = 0.0
    noise_peak: float = 0.1
    random_seed: Optional[int] = None
    disturbances: List[Dict] = field(default_factory=list)


# Backward-compatible flat aliases
_physics = PhysicalParams()
_tuning = MPCTuning()
_constraints = ConstraintLimits()
_disturbance = DisturbanceConfig()

M = _physics.M
m = _physics.m
L = _physics.L
g = _physics.g
b = _physics.b
c = _physics.c

dt = _tuning.dt
Q = _tuning.Q
R = _tuning.R
S = _tuning.S
Np = _tuning.Np
auto_dare = _tuning.auto_dare

u_max = _constraints.u_max
x_max = _constraints.x_max
q_max = _constraints.q_max
xdot_max = _constraints.xdot_max
qdot_max = _constraints.qdot_max
du_max = _constraints.du_max
rho_slack = _constraints.rho_slack

noise_mean = _disturbance.noise_mean
noise_peak = _disturbance.noise_peak
random_seed = _disturbance.random_seed
disturbances = _disturbance.disturbances
