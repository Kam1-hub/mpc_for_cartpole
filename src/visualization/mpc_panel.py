import tkinter as tk
from tkinter import ttk
import numpy as np
from src import config


def show_mpc_panel():
    """Show MPC parameter panel, returns confirmed parameter dict or None."""
    result = {"confirmed": False}

    root = tk.Tk()
    root.title("MPC Cart-Pole — Parameter Panel")
    root.resizable(False, False)

    vars_ = {}
    row = 0

    # Module A: Simulation Settings
    ttk.Label(root, text="Module A: Simulation Settings", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5))
    row += 1

    ttk.Label(root, text="T_FINAL (s):").grid(row=row, column=0, sticky="e", padx=(10, 5))
    vars_["t_final"] = tk.StringVar(value="10.0")
    ttk.Entry(root, textvariable=vars_["t_final"], width=10).grid(row=row, column=1, sticky="w")

    ttk.Label(root, text="DT (s):").grid(row=row, column=2, sticky="e", padx=(10, 5))
    vars_["dt"] = tk.StringVar(value=str(config.dt))
    ttk.Entry(root, textvariable=vars_["dt"], width=10).grid(row=row, column=3, sticky="w", padx=(0, 10))
    row += 1

    # Module A.5: Initial State
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="Module A.5: Initial State (X0)", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 5))
    row += 1

    x0_defaults = [0.0, 0.1, 0.0, 0.0]
    x0_labels = ["x (m):", "q (rad):", "dx/dt (m/s):", "dq/dt (rad/s):"]
    x0_keys = ["x0_0", "x0_1", "x0_2", "x0_3"]
    for i, (label, key) in enumerate(zip(x0_labels, x0_keys)):
        col = (i % 2) * 2
        if i % 2 == 0 and i > 0:
            row += 1
        ttk.Label(root, text=label).grid(row=row, column=col, sticky="e", padx=(10, 5))
        vars_[key] = tk.StringVar(value=str(x0_defaults[i]))
        ttk.Entry(root, textvariable=vars_[key], width=12).grid(
            row=row, column=col + 1, sticky="w", padx=(0, 10))
    row += 1

    # Module B: Reference Tracking
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="Module B: Reference Tracking", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 5))
    row += 1

    ttk.Label(root, text="Tracking Mode:").grid(row=row, column=0, sticky="e", padx=(10, 5))
    vars_["tracking_mode"] = tk.StringVar(value="Static")
    combo_tracking = ttk.Combobox(root, textvariable=vars_["tracking_mode"],
                                  values=["Static", "Cruise", "Accel"], width=12, state="readonly")
    combo_tracking.grid(row=row, column=1, sticky="w")

    lbl_tracking_val = ttk.Label(root, text="Target Value:")
    lbl_tracking_val.grid(row=row, column=2, sticky="e", padx=(10, 5))
    vars_["tracking_val"] = tk.StringVar(value="0.0")
    tracking_entry = ttk.Entry(root, textvariable=vars_["tracking_val"], width=12)
    tracking_entry.grid(row=row, column=3, sticky="w", padx=(0, 10))
    row += 1

    def on_tracking_mode_change(event=None):
        mode = vars_["tracking_mode"].get()
        if mode == "Static":
            lbl_tracking_val.config(text="Offset C (m):")
        elif mode == "Cruise":
            lbl_tracking_val.config(text="Velocity V (m/s):")
        elif mode == "Accel":
            lbl_tracking_val.config(text="Accel A (m/s²):")

    combo_tracking.bind("<<ComboboxSelected>>", on_tracking_mode_change)
    on_tracking_mode_change()

    # Module C: Cost Weight Matrices
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="Module C: Q Matrix (diagonal)", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 5))
    row += 1

    q_defaults = np.diag(config.Q).tolist()
    q_labels = ["Q11 (x):", "Q22 (q):", "Q33 (dx):", "Q44 (dq):"]
    q_keys = ["q0", "q1", "q2", "q3"]
    for i, (label, key) in enumerate(zip(q_labels, q_keys)):
        col = (i % 2) * 2
        if i % 2 == 0 and i > 0:
            row += 1
        ttk.Label(root, text=label).grid(row=row, column=col, sticky="e", padx=(10, 5))
        vars_[key] = tk.StringVar(value=str(round(q_defaults[i], 4)))
        ttk.Entry(root, textvariable=vars_[key], width=12).grid(
            row=row, column=col + 1, sticky="w", padx=(0, 10))
    row += 1

    # R matrix
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="R Matrix", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 5))
    row += 1

    ttk.Label(root, text="R11:").grid(row=row, column=0, sticky="e", padx=(10, 5))
    vars_["r0"] = tk.StringVar(value=str(config.R[0, 0]))
    ttk.Entry(root, textvariable=vars_["r0"], width=12).grid(row=row, column=1, sticky="w")
    row += 1

    # S matrix
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="S Matrix (diagonal, terminal)", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 5))
        
    vars_["auto_dare"] = tk.BooleanVar(value=True)
    dare_cb = ttk.Checkbutton(root, text="Auto-calculate via DARE", variable=vars_["auto_dare"])
    dare_cb.grid(row=row, column=2, columnspan=2, sticky="e", padx=10)
    row += 1

    s_defaults = np.diag(config.S).tolist()
    s_labels = ["S11 (x):", "S22 (q):", "S33 (dx):", "S44 (dq):"]
    s_keys = ["s0", "s1", "s2", "s3"]
    s_entries = []
    for i, (label, key) in enumerate(zip(s_labels, s_keys)):
        col = (i % 2) * 2
        if i % 2 == 0 and i > 0:
            row += 1
        ttk.Label(root, text=label).grid(row=row, column=col, sticky="e", padx=(10, 5))
        vars_[key] = tk.StringVar(value=str(round(s_defaults[i], 4)))
        s_entry = ttk.Entry(root, textvariable=vars_[key], width=12)
        s_entry.grid(row=row, column=col + 1, sticky="w", padx=(0, 10))
        s_entries.append(s_entry)
        
    def _toggle_dare(*args):
        state = "disabled" if vars_["auto_dare"].get() else "normal"
        for entry in s_entries:
            entry.config(state=state)
            
    vars_["auto_dare"].trace_add("write", _toggle_dare)
    _toggle_dare()
    row += 1

    # Module D: MPC Horizon & Bounds
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="Module D: MPC Horizon & Bounds", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 5))
    row += 1

    d_params = [
        ("Np:", "np_val", str(config.Np)),
        ("u_max (N):", "u_max", str(config.u_max)),
        ("x_max (m):", "x_max", str(config.x_max)),
        ("q_max (rad):", "q_max", str(config.q_max)),
        ("xdot_max (m/s):", "xdot_max", str(config.xdot_max)),
        ("qdot_max (rad/s):", "qdot_max", str(config.qdot_max)),
        ("du_max (N/step):", "du_max", str(config.du_max)),
        ("rho_slack:", "rho_slack", str(config.rho_slack)),
    ]
    for i, (label, key, default) in enumerate(d_params):
        col = (i % 2) * 2
        if i % 2 == 0 and i > 0:
            row += 1
        ttk.Label(root, text=label).grid(row=row, column=col, sticky="e", padx=(10, 5))
        vars_[key] = tk.StringVar(value=default)
        ttk.Entry(root, textvariable=vars_[key], width=12).grid(
            row=row, column=col + 1, sticky="w", padx=(0, 10))
    row += 1

    # Module E: Disturbances
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1
    ttk.Label(root, text="Module E: Phase 3 Disturbances", font=("", 11, "bold")).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 5))
    row += 1


    ttk.Label(root, text="Noise Mean (N):").grid(row=row, column=0, sticky="e", padx=(10, 5))
    vars_["noise_mean"] = tk.StringVar(value=str(config.noise_mean))
    ttk.Entry(root, textvariable=vars_["noise_mean"], width=12).grid(row=row, column=1, sticky="w")

    ttk.Label(root, text="Noise Peak (N):").grid(row=row, column=2, sticky="e", padx=(10, 5))
    vars_["noise_peak"] = tk.StringVar(value=str(config.noise_peak))
    ttk.Entry(root, textvariable=vars_["noise_peak"], width=12).grid(row=row, column=3, sticky="w", padx=(0, 10))
    row += 1

    ttk.Label(root, text="Random Seed:").grid(row=row, column=0, sticky="e", padx=(10, 5))
    vars_["random_seed"] = tk.StringVar(value="")
    ttk.Entry(root, textvariable=vars_["random_seed"], width=12).grid(row=row, column=1, sticky="w")
    ttk.Label(root, text="(empty = random)").grid(row=row, column=2, sticky="w", padx=(0, 5))
    row += 1


    ttk.Label(root, text="Deterministic Wind Gusts (Overlappable):").grid(
        row=row, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 2))
    row += 1

    dist_frame = ttk.Frame(root)
    dist_frame.grid(row=row, column=0, columnspan=4, padx=10, sticky="ew")
    row += 1

    dist_list = tk.Listbox(dist_frame, height=4, selectmode=tk.SINGLE)
    dist_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
    dist_scroll = ttk.Scrollbar(dist_frame, orient="vertical", command=dist_list.yview)
    dist_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    dist_list.config(yscrollcommand=dist_scroll.set)


    disturbance_data = []

    input_frame = ttk.Frame(root)
    input_frame.grid(row=row, column=0, columnspan=4, padx=10, pady=(2, 5), sticky="ew")
    row += 1

    ttk.Label(input_frame, text="start(s)").grid(row=0, column=0, padx=(0, 2))
    vars_["d_start"] = tk.StringVar(value="1.0")
    ttk.Entry(input_frame, textvariable=vars_["d_start"], width=6).grid(row=0, column=1)

    ttk.Label(input_frame, text="end(s)").grid(row=0, column=2, padx=(5, 2))
    vars_["d_end"] = tk.StringVar(value="2.0")
    ttk.Entry(input_frame, textvariable=vars_["d_end"], width=6).grid(row=0, column=3)

    ttk.Label(input_frame, text="F(N)").grid(row=0, column=4, padx=(5, 2))
    vars_["d_f"] = tk.StringVar(value="5.0")
    ttk.Entry(input_frame, textvariable=vars_["d_f"], width=6).grid(row=0, column=5)

    def on_add_dist():
        try:
            ts = float(vars_["d_start"].get())
            te = float(vars_["d_end"].get())
            f = float(vars_["d_f"].get())
            if te <= ts:
                return
            disturbance_data.append({"t_start": ts, "t_end": te, "force": f})
            dist_list.insert(tk.END, f"{ts:.2f}s – {te:.2f}s : {f:+.2f} N")
        except ValueError:
            pass

    def on_rem_dist():
        sel = dist_list.curselection()
        if sel:
            idx = sel[0]
            dist_list.delete(idx)
            disturbance_data.pop(idx)

    ttk.Button(input_frame, text="+ Add", width=6, command=on_add_dist).grid(row=0, column=6, padx=(8, 2))
    ttk.Button(input_frame, text="– Rem", width=6, command=on_rem_dist).grid(row=0, column=7, padx=(2, 0))
    row += 1

    # Module F: Visualization
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1

    vars_["ghost_trajectory"] = tk.BooleanVar(value=True)
    ttk.Checkbutton(root, text="Show Ghost Trajectory (MPC Predicted Future)", variable=vars_["ghost_trajectory"]).grid(
        row=row, column=0, columnspan=4, padx=10, pady=(2, 0), sticky="w")
    row += 1

    # Log & Start
    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
    row += 1

    vars_["save_log"] = tk.BooleanVar(value=True)
    ttk.Checkbutton(root, text="Save log (.xlsx)", variable=vars_["save_log"]).grid(
        row=row, column=0, columnspan=4, padx=10, pady=(5, 0), sticky="w")
    row += 1

    def on_start():
        result["confirmed"] = True
        result["params"] = {

            "t_final": float(vars_["t_final"].get()),
            "dt": float(vars_["dt"].get()),

            "x0": [float(vars_[f"x0_{i}"].get()) for i in range(4)],

            "tracking_mode": vars_["tracking_mode"].get(),
            "tracking_val": float(vars_["tracking_val"].get()),

            "q_diag": [float(vars_[f"q{i}"].get()) for i in range(4)],
            "r_val": float(vars_["r0"].get()),
            "s_diag": [float(vars_[f"s{i}"].get()) for i in range(4)],
            "auto_dare": vars_["auto_dare"].get(),

            "Np": int(vars_["np_val"].get()),
            "u_max": float(vars_["u_max"].get()),
            "x_max": float(vars_["x_max"].get()),
            "q_max": float(vars_["q_max"].get()),
            "xdot_max": float(vars_["xdot_max"].get()),
            "qdot_max": float(vars_["qdot_max"].get()),
            "du_max": float(vars_["du_max"].get()),
            "rho_slack": float(vars_["rho_slack"].get()),

            "noise_mean": float(vars_["noise_mean"].get()),
            "noise_peak": float(vars_["noise_peak"].get()),
            "random_seed": int(vars_["random_seed"].get()) if vars_["random_seed"].get().strip() else None,
            "disturbances": list(disturbance_data),

            "ghost_trajectory": vars_["ghost_trajectory"].get(),

            "save_log": vars_["save_log"].get(),
        }
        root.destroy()

    btn = ttk.Button(root, text="▶  Start Simulation", command=on_start)
    btn.grid(row=row, column=0, columnspan=4, pady=(10, 15))


    root.bind("<Return>", lambda e: on_start())


    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()

    if result["confirmed"]:
        return result["params"]
    return None
