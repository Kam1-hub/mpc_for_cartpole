"""Open-loop plant simulation test."""

from src.physics.plant import CartPolePlant
from src.config import dt
import numpy as np

def test_freefall():
    print("Testing cart-pole open loop physics (Phase 0)...")
    

    initial_state = np.array([0.0, 0.2, 0.0, 0.0])
    
    plant = CartPolePlant(initial_state)

    print(f"Time 0.00s | Angle: {plant.state[1]:.4f} rad")
    

    for step in range(int(1.0 / dt)):

        u_applied = 0.0
        plant.step(u=u_applied, dt=dt)
        

        if (step + 1) % int(0.1 / dt) == 0:
            time = (step + 1) * dt
            print(f"Time {time:.2f}s | Angle: {plant.state[1]:.4f} rad")

    print("\nTest completed.")
    print("If the angle increases (pendulum falling), the physics engine is working.")

if __name__ == "__main__":
    test_freefall()
