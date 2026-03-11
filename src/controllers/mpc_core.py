import cvxpy as cp
import numpy as np
from src.controllers.base_controller import BaseController


class MPCController(BaseController):
    """LTI-MPC controller using CVXPY with incremental input formulation."""

    def __init__(self, physics=None, tuning=None, constraints=None):
        super().__init__()

        import scipy.linalg
        from scipy.linalg import solve_discrete_are
        from src.config import PhysicalParams, MPCTuning, ConstraintLimits

        physics = physics or PhysicalParams()
        tuning = tuning or MPCTuning()
        constraints = constraints or ConstraintLimits()

        M, m, L, g = physics.M, physics.m, physics.L, physics.g
        dt, Np, Q, R, S = tuning.dt, tuning.Np, tuning.Q, tuning.R, tuning.S
        auto_dare = tuning.auto_dare
        u_max = constraints.u_max
        x_max, q_max = constraints.x_max, constraints.q_max
        xdot_max, qdot_max = constraints.xdot_max, constraints.qdot_max
        du_max, rho_slack = constraints.du_max, constraints.rho_slack

        self.Np = Np
        print("MPC offline matrices construction...")

        A_c = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, m*g/M, 0, 0],
            [0, (M+m)*g / (M*L), 0, 0]
        ])
        B_c = np.array([
            [0],
            [0],
            [1/M],
            [1/(M*L)]
        ])

        n = A_c.shape[0]
        p = B_c.shape[1]

        M_cont = np.vstack([
            np.hstack([A_c, B_c]),
            np.zeros((p, n + p))
        ])
        M_disc = scipy.linalg.expm(M_cont * dt)

        self.A_d = M_disc[:n, :n]
        self.B_d = M_disc[:n, n:]

        self.A_aug = np.block([
            [self.A_d,          self.B_d],
            [np.zeros((p, n)),  np.eye(p)]
        ])

        self.B_aug = np.block([
            [self.B_d],
            [np.eye(p)]
        ])

        n_a = self.A_aug.shape[0]
        p_a = self.B_aug.shape[1]

        Phi_blocks = [np.linalg.matrix_power(self.A_aug, k) for k in range(1, Np + 1)]
        Phi = np.vstack(Phi_blocks)

        Gamma = np.zeros((Np * n_a, Np * p_a))
        for i in range(Np):
            for j in range(i + 1):
                A_pow = Phi_blocks[i - j - 1] if i - j - 1 >= 0 else np.eye(n_a)
                Gamma[i * n_a : (i + 1) * n_a, j * p_a : (j + 1) * p_a] = A_pow @ self.B_aug

        self.Phi = Phi
        self.Gamma = Gamma

        # DARE: compute infinite horizon LQR terminal cost
        if auto_dare:
            P = solve_discrete_are(self.A_d, self.B_d, Q, R)
            S = P
            print("  [DARE] Terminal cost automatically calculated via infinite-horizon LQR.")

        self.S = S

        Q_aug = np.block([[Q, np.zeros((4, 1))], [np.zeros((1, 4)), np.zeros((1, 1))]])
        S_aug = np.block([[S, np.zeros((4, 1))], [np.zeros((1, 4)), np.zeros((1, 1))]])

        Q_bar = scipy.linalg.block_diag(*([Q_aug] * (Np - 1) + [S_aug]))
        R_bar = scipy.linalg.block_diag(*([R] * Np))

        H = Gamma.T @ Q_bar @ Gamma + R_bar
        F = Gamma.T @ Q_bar @ Phi
        self.H = H
        self.F = F

        L1 = np.tril(np.ones((Np, Np)))
        I_Np = np.eye(Np)

        M_hard = np.vstack([I_Np, -I_Np, L1, -L1])

        beta_hard = np.vstack([
            du_max * np.ones((Np, 1)),
            du_max * np.ones((Np, 1)),
            u_max * np.ones((Np, 1)),
            u_max * np.ones((Np, 1))
        ])

        b_hard = np.zeros((4 * Np, n_a))
        b_hard[2 * Np : 3 * Np, 4] = -1.0
        b_hard[3 * Np : 4 * Np, 4] = 1.0

        C_step = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])

        C_soft_bar = scipy.linalg.block_diag(*([C_step] * Np))

        M_soft_half = C_soft_bar @ Gamma
        M_soft = np.vstack([M_soft_half, -M_soft_half])

        b_soft_half = -C_soft_bar @ Phi
        b_soft = np.vstack([b_soft_half, -b_soft_half])

        beta_soft_step = np.array([[x_max], [q_max], [xdot_max], [qdot_max]])
        self.beta_soft_half_mem = np.tile(beta_soft_step, (Np, 1))

        self.dynamic_beta_soft = cp.Parameter(8 * Np)

        L_eps_half = np.eye(4 * Np)
        L_eps = np.vstack([L_eps_half, L_eps_half])

        self.Delta_U = cp.Variable(Np * p_a)
        self.x_init = cp.Parameter(n_a)
        self.eps = cp.Variable(4 * Np, nonneg=True)

        quad_term = 0.5 * cp.quad_form(self.Delta_U, H, assume_PSD=True)
        lin_term = self.x_init.T @ F.T @ self.Delta_U
        slacked_objective = quad_term + lin_term + rho_slack * cp.sum(cp.square(self.eps))
        objective = cp.Minimize(slacked_objective)

        constraints_list = [
            M_hard @ self.Delta_U <= beta_hard.flatten() + b_hard @ self.x_init,
            M_soft @ self.Delta_U <= self.dynamic_beta_soft + b_soft @ self.x_init + L_eps @ self.eps
        ]

        self.prob = cp.Problem(objective, constraints_list)
        print("MPC optimization problem compiled successfully (DPP).")

    def compute_action(self, current_augmented_error, current_reference=None):
        self.x_init.value = current_augmented_error

        if current_reference is None:
            current_reference = np.zeros(4)

        Np = self.Np
        ref_repeated = np.tile(current_reference.reshape(4, 1), (Np, 1))

        beta_upper = self.beta_soft_half_mem.flatten() - ref_repeated.flatten()
        beta_lower = self.beta_soft_half_mem.flatten() + ref_repeated.flatten()

        self.dynamic_beta_soft.value = np.concatenate([beta_upper, beta_lower])

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status != cp.OPTIMAL:
                print(f"[MPC Warning] Solver status: {self.prob.status}")
                if self.Delta_U.value is None:
                    return 0.0, None

            delta_u_star = self.Delta_U.value[0]

            try:
                x_future_aug = self.Phi @ current_augmented_error + self.Gamma @ self.Delta_U.value
                ghost_states = np.zeros((Np, 4))
                for k in range(Np):
                    aug_k = x_future_aug[k * 5 : (k + 1) * 5]
                    ghost_states[k] = aug_k[:4] + current_reference
            except Exception:
                ghost_states = None

            return delta_u_star, ghost_states

        except Exception as e:
            print(f"[MPC Error] Solver crashed: {e}")
            return 0.0, None
