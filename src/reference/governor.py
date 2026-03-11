import numpy as np
from src.config import PhysicalParams

class ReferenceGovernor:
    """Reference generator and safety clamper for Static/Cruise/Accel tracking modes."""
    def __init__(self, mode="Static", physics: PhysicalParams = None, q_max: float = 0.24):
        self.mode = mode
        self.physics = physics or PhysicalParams()
        self.q_max = q_max
        self.start_pos = 0.0
        self.start_vel = 0.0
        self.target_val = 0.0
        
    def step_reference(self, t):
        x_d = np.zeros(4)
        u_d = 0.0
        M = self.physics.M
        m = self.physics.m
        g = self.physics.g
        q_max = self.q_max
        
        if self.mode == "Static":
            C = self.target_val
            x_d = np.array([C, 0.0, 0.0, 0.0])
            u_d = 0.0
            
        elif self.mode == "Cruise":
            V = self.target_val
            x_d = np.array([self.start_pos + V * t, 0.0, V, 0.0])
            u_d = 0.0
            
        elif self.mode == "Accel":
            A = self.target_val

            q_d = np.arctan(-A / g)

            if abs(q_d) > q_max:
                print(f"[Governor Warning] Requested Accel {A} exceeds limits! Clamping.")
                q_d = np.sign(q_d) * q_max
                A = -g * np.tan(q_d) 
                
            x_d = np.array([self.start_pos + self.start_vel * t + 0.5 * A * t**2, 
                            q_d, 
                            self.start_vel + A * t, 
                            0.0])

            u_d = (M + m) * A
            
        return x_d, u_d
