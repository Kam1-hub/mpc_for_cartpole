import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from src.config import PhysicalParams

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class CartPoleGUI:
    """Real-time Matplotlib cart-pole animation with state tracking curves."""

    CART_W = 0.4
    CART_H = 0.2
    MAX_HISTORY = 1000

    def __init__(self, physics: PhysicalParams = None):
        self.physics = physics or PhysicalParams()
        self.POLE_LEN = 2 * self.physics.L
        plt.ion()

        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(
            7, 2, width_ratios=[0.8, 1.2],
            left=0.04, right=0.97, top=0.94, bottom=0.07,
            hspace=0.35, wspace=0.35
        )

        self.ax_anim = self.fig.add_subplot(gs[:, 0])
        self._setup_anim_axes()
        self._create_anim_artists()

        self.curve_configs = [
            ("x (m)",        "steelblue"),
            ("q (deg)",      "firebrick"),
            ("dx/dt (m/s)",  "seagreen"),
            ("dq/dt (r/s)",  "darkorange"),
            ("F (N)",        "mediumpurple"),
            ("F_noise (N)",  "darkcyan"),
            ("F_dist (N)",   "orangered"),
        ]

        self.ax_curves = []
        self.curve_lines = []
        self.target_lines = []
        self.vlines = []

        self.t_history = deque(maxlen=self.MAX_HISTORY)
        self.state_history = [deque(maxlen=self.MAX_HISTORY) for _ in range(7)]

        for i, (label, color) in enumerate(self.curve_configs):
            ax = self.fig.add_subplot(gs[i, 1])
            ax.text(0.02, 0.92, label, transform=ax.transAxes, fontsize=9,
                    fontweight="bold", color=color, verticalalignment="top")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2)
            ax.axhline(y=0, color="k", alpha=0.2, lw=0.5)

            line, = ax.plot([], [], color=color, lw=1.2)
            self.curve_lines.append(line)

            if i < 4:
                tgt_line, = ax.plot([], [], color="mediumseagreen", ls="--", alpha=0.8, lw=1.2)
                self.target_lines.append(tgt_line)

            vl = ax.axvline(x=0, color="red", alpha=0.6, lw=1)
            self.vlines.append(vl)

            if i < 6:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("t (s)", fontsize=9)

            self.ax_curves.append(ax)

        self.fig.tight_layout()

    def _setup_anim_axes(self):
        ax = self.ax_anim
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_title("Cart-Pole MPC Control", fontsize=13, fontweight="bold")
        ax.axhline(y=0, color="k", linewidth=2)

        self.time_text = ax.text(
            0.02, 0.95, "",
            transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
        )

    def _create_anim_artists(self):
        ax = self.ax_anim

        self.cart_patch = patches.Rectangle(
            (0, 0), self.CART_W, self.CART_H,
            fc="steelblue", ec="black", lw=2,
        )
        ax.add_patch(self.cart_patch)

        (self.pole_line,) = ax.plot(
            [], [], "o-",
            color="firebrick", lw=3, markersize=8,
        )

        (self.ghost_line,) = ax.plot(
            [], [], "--",
            color="dodgerblue", alpha=0.4, lw=2,
        )
        self.show_ghost = True

    def render(self, state, time, force=0.0, target_state=None, settle_time=None, ghost_states=None,
               f_noise=0.0, f_dist=0.0):
        x, q, _, _ = state

        self.cart_patch.set_xy((x - self.CART_W / 2, -self.CART_H / 2))

        pole_tip_x = x - self.POLE_LEN * np.sin(q)
        pole_tip_y = self.POLE_LEN * np.cos(q)
        self.pole_line.set_data([x, pole_tip_x], [0, pole_tip_y])

        if self.show_ghost and ghost_states is not None:
            ghost_cart_x = ghost_states[:, 0]
            ghost_q = ghost_states[:, 1]
            ghost_tip_x = ghost_cart_x - self.POLE_LEN * np.sin(ghost_q)
            ghost_tip_y = self.POLE_LEN * np.cos(ghost_q)
            self.ghost_line.set_data(ghost_tip_x, ghost_tip_y)
        else:
            self.ghost_line.set_data([], [])

        if settle_time is not None:
            self.time_text.set_text(f"t = {time:.2f} s  |  Settled at {settle_time:.3f}s")
        else:
            self.time_text.set_text(f"t = {time:.2f} s")

        self.t_history.append(time)
        curve_data = [state[0], np.degrees(state[1]), state[2], state[3], force, f_noise, f_dist]
        for i, val in enumerate(curve_data):
            self.state_history[i].append(val)

        t_arr = np.array(self.t_history)

        for i in range(7):
            data_arr = np.array(self.state_history[i])
            self.curve_lines[i].set_data(t_arr, data_arr)

            if i < 4 and target_state is not None:
                tgt_val = target_state[i]
                if i == 1:
                    tgt_val = np.degrees(tgt_val)
                self.target_lines[i].set_data(t_arr, np.full_like(t_arr, tgt_val))

            ax = self.ax_curves[i]
            if len(data_arr) > 1:
                d_min, d_max = np.min(data_arr), np.max(data_arr)
                d_range = d_max - d_min if d_max != d_min else 1.0
                ax.set_ylim(d_min - 0.1 * d_range, d_max + 0.1 * d_range)

            if len(t_arr) > 1:
                ax.set_xlim(t_arr[0], max(t_arr[-1], 0.5))

            if settle_time is not None and not hasattr(self, '_settle_drawn'):
                ax.axvline(x=settle_time, color="gray", ls="--", alpha=0.5, lw=0.8)

            self.vlines[i].set_xdata([time])

        if settle_time is not None:
            self._settle_drawn = True

        self.fig.canvas.flush_events()
