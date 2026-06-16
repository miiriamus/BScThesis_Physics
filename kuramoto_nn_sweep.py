# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 02:02:00 2026

@author: Miriam_Ucendo
"""
"""
kuramoto_nn_sweep.py
====================
Kuramoto model with Lorentzian frequency distribution.

Fixed parameters:
    K     = 20 * Kc
    Delta = 1
    N     = 500

Sweep:
    nn in [1, nn_max=250]   ->  f = 2*nn / (N-1)

For each nn, compute the time-averaged order parameter r via RK4 and
plot f vs. r alongside a horizontal line with the analytical value
r = sqrt(1 - Kc/K) for the fixed K.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
Delta          = 0.1
omega0         = 0.0
N              = 1000
nn_max         = 500
nn_step        = 1
T              = 250
dt             = 0.1
transient_frac = 0.8
seed           = 130402
export_png     = True
png_filename   = f"kuramoto_f_vs_r_N{N}_D{Delta}_T{T}.png"

Kc = 2.0 * Delta
K  = 20.0 * Kc
print(f"Parameters: Delta={Delta}, N={N}, T={T}, Kc={Kc:.4f}, K={K:.4f} (= 20*Kc)")
# ──────────────────────────────────────────────


# ── Adjacency matrix  ─────────────────
def create_coupling_matrix(N, nn):
    row_indices, col_indices = np.ogrid[:N, :N]
    diff = np.abs(row_indices - col_indices)
    distance_matrix = np.minimum(diff, N - diff)
    mask = (distance_matrix > 0) & (distance_matrix <= nn)
    A = mask.astype(int)
    return A


# ── Analytical r  ─────────────────────
def analytical_r(K_array, Kc):
    r = np.zeros_like(K_array)
    mask = K_array >= Kc
    r[mask] = np.sqrt(1.0 - Kc / K_array[mask])
    return r


# ── Lorentzian frequency sampling  ────────────────────────────────
def sample_lorentzian(N, Delta, omega0, rng):
    u = rng.uniform(0.0, 1.0, N)
    omega = omega0 + Delta * np.tan(np.pi * (u - 0.5))
    clip = 50.0 * Delta
    return np.clip(omega, omega0 - clip, omega0 + clip)


# ── RK4 with coupling matrix  ─────────────────────────────────────
def kuramoto_rhs_matrix(theta, omega, K, nn, N, row_idx, col_idx):
    sin_values = np.sin(theta[col_idx] - theta[row_idx])
    coupling_sum = K / (nn * 2) * np.bincount(row_idx, weights=sin_values, minlength=N)
    return omega + coupling_sum

def rk4_step_matrix(theta, omega, K, nn, N, dt, row_idx, col_idx):
    k1 = kuramoto_rhs_matrix(theta,              omega, K, nn, N, row_idx, col_idx)
    k2 = kuramoto_rhs_matrix(theta + 0.5*dt*k1, omega, K, nn, N, row_idx, col_idx)
    k3 = kuramoto_rhs_matrix(theta + 0.5*dt*k2, omega, K, nn, N, row_idx, col_idx)
    k4 = kuramoto_rhs_matrix(theta +     dt*k3, omega, K, nn, N, row_idx, col_idx)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter_matrix(K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx):
    N           = len(theta0)
    n_steps     = int(T / dt)
    n_transient = int(n_steps * transient_frac)
    theta       = theta0.copy()
    r_values = []
    for step in range(n_steps):
        theta = rk4_step_matrix(theta, omega, K, nn, N, dt, row_idx, col_idx)
        if step >= n_transient:
            rho = np.mean(np.exp(1j * theta))
            r_values.append(np.abs(rho))
    return np.mean(r_values)


# ── Main sweep ────────────────────────────────────────────────────────────────
def run_f_sweep():
    rng    = np.random.default_rng(seed)
    omega  = sample_lorentzian(N, Delta, omega0, rng)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, N)

    nn_values = np.arange(1, nn_max + 1, nn_step)
    f_values  = 2 * nn_values / (N - 1)
    r_values  = np.zeros(len(nn_values))

    t_total = time.time()
    for idx, nn in enumerate(nn_values):
        t0 = time.time()
        A = create_coupling_matrix(N, nn)
        row_idx, col_idx = A.nonzero()
        row_idx = row_idx.astype(np.int32)
        col_idx = col_idx.astype(np.int32)
        r_values[idx] = simulate_order_parameter_matrix(
            K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx
        )
        elapsed = time.time() - t0
        print(f"  nn={nn:4d}  f={f_values[idx]:.4f}  r={r_values[idx]:.4f}  ({elapsed:.2f} s)")

    print(f"\nTotal sweep time: {time.time()-t_total:.1f} s")
    return f_values, r_values


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_f_vs_r(f_values, r_values, K, Kc, Delta, N, export_png, png_filename):
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.linewidth":    1.2,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "figure.dpi":        150,
    })

    # Analytical r for the fixed K — same call as in kuramoto_lorentzian.py
    r_theory = analytical_r(np.array([K]), Kc)[0]
    print(f"Analytical r(K={K:.2f}, Kc={Kc:.2f}) = {r_theory:.6f}")

    fig_w = 7.0
    fig_h = fig_w * 0.65
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.plot(f_values, r_values, color="#e94560", lw=1.5, marker="o",
            markersize=3.0, alpha=0.85, label=fr"RK4 simulation ($N={N}$)")

    ax.axhline(r_theory, color="#1a1a2e", lw=2.0, ls="-",
               label=fr"Analytical $r = \sqrt{{1-K_c/K}} = {r_theory:.4f}$  $(N\to\infty)$")

    ax.set_xlabel(r"Connectivity fraction $f = 2nn/(N-1)$", labelpad=6)
    ax.set_ylabel(r"Order parameter $r$", labelpad=6)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        fr"Kuramoto–Lorentzian: $f$ vs $r$  "
        fr"($K=20K_c={K:.1f}$, $\Delta={Delta}$, $N={N}$)",
        pad=8, fontsize=11
    )

    fig.tight_layout()

    if export_png:
        fig.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
        print(f"Figure saved as '{png_filename}'")

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    f_values, r_values = run_f_sweep()

    dat_filename = f"kuramoto_f_vs_r_N{N}_D{Delta}.dat"
    np.savetxt(dat_filename, np.column_stack([f_values, r_values]),
               header="f   r_rk4", fmt="%.6f")
    print(f"Data saved to '{dat_filename}'")

    plot_f_vs_r(f_values, r_values, K, Kc, Delta, N, export_png, png_filename)