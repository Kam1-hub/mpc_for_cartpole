import numpy as np
from src.config import PhysicalParams


class CartPolePlant:
    """Nonlinear cart-pole physics simulator using RK4 integration."""

    def __init__(self, state_init=None, physics: PhysicalParams = None):
        self.physics = physics or PhysicalParams()
        if state_init is not None:
            self.state = np.array(state_init, dtype=float)
        else:
            self.state = np.zeros(4)

    def _dynamics(self, state, u):
        x, q, x_dot, q_dot = state
        M = self.physics.M
        m = self.physics.m
        L = self.physics.L
        g = self.physics.g
        b = self.physics.b
        c = self.physics.c

        sin_q = np.sin(q)
        cos_q = np.cos(q)

        denom = M + m - m * cos_q**2

        x_ddot = (u - b * x_dot 
                  - m * L * q_dot**2 * sin_q 
                  + m * g * sin_q * cos_q 
                  - c * q_dot * cos_q / L) / denom

        q_ddot = ((M + m) * g * sin_q 
                  + (u - b * x_dot) * cos_q 
                  - m * L * q_dot**2 * sin_q * cos_q 
                  - c * q_dot * (M + m) / (m * L)) / (L * denom)
        
        return np.array([x_dot, q_dot, x_ddot, q_ddot])

    def step(self, u, dt):
        k1 = self._dynamics(self.state, u)
        k2 = self._dynamics(self.state + 0.5 * dt * k1, u)
        k3 = self._dynamics(self.state + 0.5 * dt * k2, u)
        k4 = self._dynamics(self.state + dt * k3, u)
        
        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
