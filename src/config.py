import numpy as np

# Physics Constants
M = 1.0       # Cart mass (kg)
m = 0.1       # Pole mass (kg)
L = 0.3       # Pole half-length (m)
g = 9.81      # Gravity (m/s^2)


b = 0.0       # Cart friction
c = 0.0       # Pivot friction

# Time step for physics integration and controller frequency (60Hz -> ~0.0167s)
dt = 0.0167   

# Cost matrices (Bryson's Rule)
Q = np.diag([1.0/(0.5**2),      # x
             1.0/(0.24**2),     # q
             1.0/(0.5**2),      # x_dot
             1.0/(1.0**2)])     # q_dot

R = np.array([[1.0/(10.0**2)]])

S = np.diag([1.0/(0.01**2),     # x
             1.0/(0.017**2),    # q
             1.0/(0.05**2),     # x_dot
             1.0/(0.05**2)])    # q_dot

# MPC horizon
Np = 30

# Constraint bounds
u_max = 10.0        # Max motor force (N)
x_max = 0.5         # Max track distance (m)
q_max = 0.24        # Max pole tilt angle (rad)
xdot_max = 0.5      # Max cart velocity (m/s)
qdot_max = 1.0      # Max pole angular velocity (rad/s)
du_max = 2.0        # Max control input change per step (N/step)

rho_slack = 10000.0  # Soft constraint penalty weight

# External disturbance defaults
noise_mean = 0.0
noise_peak = 0.1
random_seed = None
disturbances = []

