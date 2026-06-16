# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:34:57 2026

@author: Miriam_Ucendo
"""
"""
kuramoto_lorentzian_multiN_nn.py
===================
Kuramoto model with Lorentzian frequency distribution.
Plots r(K) for multiple system sizes N on a single figure,
comparing each against the analytical mean-field solution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
Delta          = 1
omega0         = 0.0
nn             = 1          # k-nearest neighbours (1 = local; N//2 = mean-field)
K_points       = 100
T              = 250
dt             = 0.1
transient_frac = 0.8
seed           = 130402
export_png     = True

# ── Introduce aquí los valores de N que quieras simular ──────────────────
N_LIST = [10, 50, 100, 200, 300, 500, 1000]
f_list = []
# ─────────────────────────────────────────────────────────────────────────

# Colores generados automáticamente según el número de valores en N_LIST
import matplotlib.cm as _cm
COLORS = [_cm.plasma(i / max(len(N_LIST) - 1, 1)) for i in range(len(N_LIST))]

png_filename = f"NN{nn}_multiN_d{Delta}_T{T}_distrib3.png"
# ──────────────────────────────────────────────


# ── Adjacency matrix ──────────────────────────
def create_coupling_matrix(N, nn):
    row_indices, col_indices = np.ogrid[:N, :N]
    diff = np.abs(row_indices - col_indices)
    distance_matrix = np.minimum(diff, N - diff)
    mask = (distance_matrix > 0) & (distance_matrix <= nn)
    return mask.astype(int)


# ── Analytical solution ───────────────────────
def analytical_r(K_array, Kc):
    r = np.zeros_like(K_array)
    mask = K_array >= Kc
    r[mask] = np.sqrt(1.0 - Kc / K_array[mask])
    return r

def critical_coupling(Delta):
    return 2.0 * Delta


# ── Lorentzian sampling ───────────────────────
def sample_lorentzian(N, Delta, omega0, rng):
    u = rng.uniform(0.0, 1.0, N)
    omega = omega0 + Delta * np.tan(np.pi * (u - 0.5))
    clip = 50.0 * Delta
    return np.clip(omega, omega0 - clip, omega0 + clip)


# ── RK4: all-to-all O(N) ─────────────────────
def kuramoto_rhs(theta, omega, K, N):
    sum_sin  = np.sum(np.sin(theta))
    sum_cos  = np.sum(np.cos(theta))
    coupling = (K / N) * (sum_sin * np.cos(theta) - sum_cos * np.sin(theta))
    return omega + coupling

def rk4_step(theta, omega, K, N, dt):
    k1 = kuramoto_rhs(theta,             omega, K, N)
    k2 = kuramoto_rhs(theta + 0.5*dt*k1, omega, K, N)
    k3 = kuramoto_rhs(theta + 0.5*dt*k2, omega, K, N)
    k4 = kuramoto_rhs(theta +     dt*k3, omega, K, N)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter(K, omega, theta0, T, dt, transient_frac):
    N           = len(theta0)
    n_steps     = int(T / dt)
    n_transient = int(n_steps * transient_frac)
    theta       = theta0.copy()
    r_values    = []
    for step in range(n_steps):
        theta = rk4_step(theta, omega, K, N, dt)
        if step >= n_transient:
            r_values.append(np.abs(np.mean(np.exp(1j * theta))))
    return np.mean(r_values)


# ── RK4: sparse topology O(|E|) ───────────────
def kuramoto_rhs_matrix(theta, omega, K, nn, N, row_idx, col_idx):
    sin_values   = np.sin(theta[col_idx] - theta[row_idx])
    coupling_sum = K / (nn * 2) * np.bincount(row_idx, weights=sin_values, minlength=N)
    return omega + coupling_sum

def rk4_step_matrix(theta, omega, K, nn, N, dt, row_idx, col_idx):
    k1 = kuramoto_rhs_matrix(theta,             omega, K, nn, N, row_idx, col_idx)
    k2 = kuramoto_rhs_matrix(theta + 0.5*dt*k1, omega, K, nn, N, row_idx, col_idx)
    k3 = kuramoto_rhs_matrix(theta + 0.5*dt*k2, omega, K, nn, N, row_idx, col_idx)
    k4 = kuramoto_rhs_matrix(theta +     dt*k3, omega, K, nn, N, row_idx, col_idx)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter_matrix(K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx):
    N           = len(theta0)
    n_steps     = int(T / dt)
    n_transient = int(n_steps * transient_frac)
    theta       = theta0.copy()
    r_values    = []
    for step in range(n_steps):
        theta = rk4_step_matrix(theta, omega, K, nn, N, dt, row_idx, col_idx)
        if step >= n_transient:
            r_values.append(np.abs(np.mean(np.exp(1j * theta))))
    return np.mean(r_values)


# ── Numerical Kc detection ────────────────────
def detect_kc_numerical(K_array, r_array, threshold=0.05):
    idx = np.argmax(r_array > threshold)
    return K_array[idx] if r_array[idx] > threshold else None


# ── Single-N sweep ────────────────────────────
def run_sweep_for_N(N, Delta, omega0, nn, K_points, T, dt, transient_frac, seed, K_array):
    rng    = np.random.default_rng(seed)
    omega  = sample_lorentzian(N, Delta, omega0, rng)
    theta0 = rng.uniform(0.0, 0.01 * np.pi, N)

    A = create_coupling_matrix(N, nn)
    all_to_all_ref = np.ones((N, N)) - np.eye(N)
    is_all_to_all  = np.array_equal(A, all_to_all_ref) or np.all(A == 1)

    if is_all_to_all:
        print("-> All-to-all coupling detected. Using fast O(N) algorithm...")
        row_idx = col_idx = None
    else:
        print("-> Custom network topology detected. Using structural sparse-coordinate algorithm...")
        row_idx, col_idx = A.nonzero()
        row_idx = row_idx.astype(np.int32)
        col_idx = col_idx.astype(np.int32)

    r_rk4 = np.zeros(K_points)
    for i, K in enumerate(K_array):
        if is_all_to_all:
            r_rk4[i] = simulate_order_parameter(K, omega, theta0, T, dt, transient_frac)
        else:
            r_rk4[i] = simulate_order_parameter_matrix(K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx)

    return r_rk4


# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    Kc      = critical_coupling(Delta)
    K_array = np.linspace(0.0, 4.0 * Kc, K_points)
    r_anal  = analytical_r(K_array, Kc)

    results = {}
    for N in N_LIST:
        t0 = time.time()
        print(f"\nRunning N={N} ...")
        r = run_sweep_for_N(N, Delta, omega0, nn,
                            K_points, T, dt, transient_frac, seed, K_array)
        results[N] = r
        Kc_n = detect_kc_numerical(K_array, r)
        f_list.append(100 * 2*nn / (N - 1))
        print(f"  N={N:>4d} | Kc_numerical={Kc_n:.3f} | elapsed={time.time()-t0:.1f}s")

    print("The connectivity for each value of N considering NN-interaction is \n")
    print("f:", [f"{x:.4f}" for x in f_list])

    # ── Plot ──────────────────────────────────
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.linewidth":    1.2,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "figure.dpi":        150,
    })

    fig_w = 7.5
    fig_h = fig_w * (1 + np.sqrt(5)) / 2 * 0.55
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.08)
    ax1   = fig.add_subplot(gs[0])
    ax2   = fig.add_subplot(gs[1], sharex=ax1)

    # Analytical curve (reference)
    ax1.plot(K_array, r_anal, color="black", lw=2.2, ls="-", zorder=10,
             label=r"Analytical $(N\to\infty)$")
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)

    # One curve per N
    for N, color in zip(N_LIST, COLORS):
        r = results[N]
        ax1.plot(K_array, r, color=color, lw=1.4, ls="--", alpha=0.85,
                 label=fr"$N={N}$")
        error = np.abs(r_anal - r)
        ax2.semilogy(K_array, error + 1e-10, color=color, lw=1.2, alpha=0.85)

    # Kc marker
    ax1.axvline(Kc, color="gray", lw=1.0, ls=":", label=fr"$K_c = 2\Delta = {Kc:.2f}$")
    ax2.axvline(Kc, color="gray", lw=1.0, ls=":")

    ax1.set_ylabel(r"Order parameter $r$", labelpad=6)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax1.set_title(
        fr"Kuramoto model — Lorentzian ($\Delta={Delta}$, $\omega_0={omega0}$, "
        fr"$k={nn}$-nearest neighbours)",
        pad=8, fontsize=10.5
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_xlabel(r"Coupling strength $K$", labelpad=6)
    ax2.set_ylabel(r"$|r_{\rm an} - r_{\rm RK4}|$", labelpad=6)
    ax2.set_ylim(bottom=1e-4)

    fig.align_ylabels([ax1, ax2])

    if export_png:
        fig.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
        print(f"\nFigure saved as '{png_filename}'")

    plt.show()