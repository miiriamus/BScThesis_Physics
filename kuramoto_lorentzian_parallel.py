# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 01:29:11 2026

@author: Miriam_Ucendo
"""
"""
kuramoto_lorentzian_parallel.py
================================
Kuramoto model in the thermodynamic limit with a Lorentzian frequency distribution.
Parallelised K-sweep using multiprocessing.Pool.

Compares:
  - Analytical solution (exact, N -> inf):
        r = 0              if K < Kc
        r = sqrt(1 - Kc/K) if K >= Kc,   Kc = 2*Delta
  - Direct RK4 simulation (finite N):
        r(K) = time-average of |<exp(i*theta)>| over steady state

Reference: Kuramoto (1984); Strogatz (2000) Physica D 143, 1-20.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import multiprocessing
from functools import partial

# ──────────────────────────────────────────────
# CONFIG  —  edit here
# ──────────────────────────────────────────────
Delta              = 1.0      # Lorentzian half-width
omega0             = 0.0      # Central frequency
N                  = 5000      # Number of oscillators
neighborhood_range = 1      # interaction with k-nearest neighbors
K_points           = 30       # Resolution of K sweep
T                  = 50.0     # Total integration time
dt                 = 0.1      # RK4 step
transient_frac     = 0.8      # Fraction discarded as transient
seed               = 130402       # Random seed
export_png         = True     # Save figure as PNG
n_workers          = None     # Parallel workers (None = use all CPU cores)
png_filename = f"kuramoto_lorentzian_k{neighborhood_range}_N{N}.png"
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Adjacent matrix
# ──────────────────────────────────────────────
def create_coupling_matrix(N, neighborhood_range):
    """
    Creates an NxN adjacency matrix with a specific neighborhood interaction range.
    The main diagonal is always 0.
    """
    row_indices, col_indices = np.ogrid[:N, :N]
    distance_matrix = np.abs(row_indices - col_indices)
    mask = (distance_matrix > 0) & (distance_matrix <= neighborhood_range)
    A = mask.astype(int)
    return A


# ── 1. Analytical r solution ────────────────────
def analytical_r(K_array, Kc):
    r = np.zeros_like(K_array)
    mask = K_array >= Kc
    r[mask] = np.sqrt(1.0 - Kc / K_array[mask])
    return r

def critical_coupling(Delta):
    return 2.0 * Delta


# ── 2. Lorentzian frequency sampling ─────────
def sample_lorentzian(N, Delta, omega0, rng):
    u = rng.uniform(0.0, 1.0, N)
    omega = omega0 + Delta * np.tan(np.pi * (u - 0.5))
    clip = 50.0 * Delta
    return np.clip(omega, omega0 - clip, omega0 + clip)


# ── 3. RK4 integrator (vectorised) ───────────
def kuramoto_rhs(theta, omega, K, N):
    sum_sin = np.sum(np.sin(theta))
    sum_cos = np.sum(np.cos(theta))
    coupling_sum = (K / N) * (sum_sin * np.cos(theta) - sum_cos * np.sin(theta))
    return omega + coupling_sum

def rk4_step(theta, omega, K, N, dt):
    k1 = kuramoto_rhs(theta,             omega, K, N)
    k2 = kuramoto_rhs(theta + 0.5*dt*k1, omega, K, N)
    k3 = kuramoto_rhs(theta + 0.5*dt*k2, omega, K, N)
    k4 = kuramoto_rhs(theta +     dt*k3, omega, K, N)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter(K, omega, theta0, T, dt, transient_frac):
    N          = len(theta0)
    n_steps    = int(T / dt)
    n_transient= int(n_steps * transient_frac)
    theta      = theta0.copy()
    r_values   = []
    for step in range(n_steps):
        theta = rk4_step(theta, omega, K, N, dt)
        if step >= n_transient:
            rho = np.mean(np.exp(1j * theta))
            r_values.append(np.abs(rho))
    return np.mean(r_values)


# ── 3.2 RK4 integrator with Coupling Matrix ───
def kuramoto_rhs_matrix(theta, omega, K, N, row_idx, col_idx):
    sin_values   = np.sin(theta[col_idx] - theta[row_idx])
    coupling_sum = K/N * np.bincount(row_idx, weights=sin_values, minlength=N)
    return omega + coupling_sum

def rk4_step_matrix(theta, omega, K, N, dt, row_idx, col_idx):
    k1 = kuramoto_rhs_matrix(theta,             omega, K, N, row_idx, col_idx)
    k2 = kuramoto_rhs_matrix(theta + 0.5*dt*k1, omega, K, N, row_idx, col_idx)
    k3 = kuramoto_rhs_matrix(theta + 0.5*dt*k2, omega, K, N, row_idx, col_idx)
    k4 = kuramoto_rhs_matrix(theta +     dt*k3, omega, K, N, row_idx, col_idx)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter_matrix(K, omega, theta0, T, dt, transient_frac,
                                    row_idx, col_idx):
    N          = len(theta0)
    n_steps    = int(T / dt)
    n_transient= int(n_steps * transient_frac)
    theta      = theta0.copy()
    r_values   = []
    for step in range(n_steps):
        theta = rk4_step_matrix(theta, omega, K, N, dt, row_idx, col_idx)
        if step >= n_transient:
            rho = np.mean(np.exp(1j * theta))
            r_values.append(np.abs(rho))
    return np.mean(r_values)


# ── 4. Worker functions (module-level, picklable) ─────────────────────────────
# multiprocessing requires top-level functions; lambdas and nested defs won't work.

def _worker_all_to_all(args):
    """Worker for all-to-all coupling. Returns (index, r_value)."""
    i, K, omega, theta0, T, dt, transient_frac = args
    r = simulate_order_parameter(K, omega, theta0, T, dt, transient_frac)
    return i, r

def _worker_matrix(args):
    """Worker for sparse matrix coupling. Returns (index, r_value)."""
    i, K, omega, theta0, T, dt, transient_frac, row_idx, col_idx = args
    r = simulate_order_parameter_matrix(K, omega, theta0, T, dt, transient_frac,
                                        row_idx, col_idx)
    return i, r


# ── 5. Numerical Kc detection ─────────────────
def detect_kc_numerical(K_array, r_array, threshold=0.05):
    idx = np.argmax(r_array > threshold)
    if r_array[idx] > threshold:
        return K_array[idx]
    return None


# ── 6. Main sweep (parallelised) ──────────────
def run_sweep(Delta, omega0, N, neighborhood_range, K_points, T, dt,
              transient_frac, seed, A, n_workers=None):
    """
    Perform the full K sweep.
    Each K value is dispatched to a separate process via multiprocessing.Pool.
    Results are collected in order regardless of completion order.
    """
    rng   = np.random.default_rng(seed)
    omega = sample_lorentzian(N, Delta, omega0, rng)

    Kc      = critical_coupling(Delta)
    K_array = np.linspace(0.0, 2.0 * Kc, K_points)
    theta0  = rng.uniform(0.0, 2.0 * np.pi, N)

    all_to_all_reference = np.ones((N, N)) - np.eye(N)
    is_all_to_all = np.array_equal(A, all_to_all_reference) or np.all(A == 1)

    if is_all_to_all:
        print("-> All-to-all coupling detected. Using fast O(N) algorithm...")
        row_idx, col_idx = None, None
    else:
        print("-> Custom network topology detected. Using structural sparse-coordinate algorithm...")
        row_idx, col_idx = A.nonzero()
        row_idx = row_idx.astype(np.int32)
        col_idx = col_idx.astype(np.int32)

    n_cpu = multiprocessing.cpu_count() if n_workers is None else n_workers
    print(f"{neighborhood_range}-nearest neighbor interaction")
    print(f"Kc (theoretical) = {Kc:.4f}")
    print(f"Sweeping {K_points} values of K in [0, {2*Kc:.2f}] "
          f"using {n_cpu} parallel workers ...")

    # Build argument list — one tuple per K value
    if is_all_to_all:
        task_args = [
            (i, K, omega, theta0, T, dt, transient_frac)
            for i, K in enumerate(K_array)
        ]
        worker_fn = _worker_all_to_all
    else:
        task_args = [
            (i, K, omega, theta0, T, dt, transient_frac, row_idx, col_idx)
            for i, K in enumerate(K_array)
        ]
        worker_fn = _worker_matrix

    r_rk4    = np.zeros(K_points)
    t_start  = time.time()
    completed = 0

    # imap_unordered returns results as soon as each worker finishes,
    # so we track progress in real time while still storing results correctly.
    with multiprocessing.Pool(processes=n_cpu) as pool:
        for i, r_val in pool.imap_unordered(worker_fn, task_args):
            r_rk4[i] = r_val
            completed += 1
            elapsed  = time.time() - t_start
            eta      = elapsed / completed * (K_points - completed)
            print(f"  [{completed:>{len(str(K_points))}}/{K_points}]  "
                  f"K = {K_array[i]:.3f}  r = {r_val:.4f}  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s")

    r_analytical = analytical_r(K_array, Kc)
    Kc_numerical = detect_kc_numerical(K_array, r_rk4)

    return {
        "K":            K_array,
        "r_analytical": r_analytical,
        "r_rk4":        r_rk4,
        "Kc_theory":    Kc,
        "Kc_numerical": Kc_numerical,
    }


# ── 7. Plot ───────────────────────────────────
def plot_results(results, Delta, N, T, dt, export_png, png_filename):
    K            = results["K"]
    r_analytical = results["r_analytical"]
    r_rk4        = results["r_rk4"]
    Kc_t         = results["Kc_theory"]
    Kc_n         = results["Kc_numerical"]
    error        = np.abs(r_analytical - r_rk4)

    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.linewidth":   1.2,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.top":        True,
        "ytick.right":      True,
        "legend.frameon":   True,
        "legend.framealpha":0.9,
        "figure.dpi":       150,
    })

    fig_w = 7.0
    fig_h = fig_w * (1 + np.sqrt(5)) / 2 * 0.55
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.08)
    ax1   = fig.add_subplot(gs[0])
    ax2   = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(K, r_analytical, color="#1a1a2e", lw=2.0, ls="-",
             label=r"Analytical: $r = \sqrt{1 - K_c/K}$  $(N\to\infty)$")
    ax1.plot(K, r_rk4, color="#e94560", lw=1.5, ls="--", alpha=0.85,
             label=fr"RK4 simulation ($N={N}$, $T={T}$, $\Delta t={dt}$)")
    ax1.axvline(Kc_t, color="#1a1a2e", lw=1.0, ls=":",
                label=fr"$K_c^{{\rm theory}} = 2\Delta = {Kc_t:.3f}$")
    if Kc_n is not None:
        ax1.axvline(Kc_n, color="#e94560", lw=1.0, ls=":",
                    label=fr"$K_c^{{\rm numerical}} \approx {Kc_n:.3f}$")

    ax1.set_ylabel(r"Order parameter $r$", labelpad=6)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(
        fr"Kuramoto model — Lorentzian distribution "
        fr"($\Delta={Delta}$, $\omega_0={0}$)",
        pad=8, fontsize=11
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.semilogy(K, error + 1e-10, color="#0f3460", lw=1.4)
    ax2.axvline(Kc_t, color="#1a1a2e", lw=1.0, ls=":")
    ax2.set_xlabel(r"Coupling strength $K$", labelpad=6)
    ax2.set_ylabel(r"$|r_{\rm an} - r_{\rm RK4}|$", labelpad=6)
    ax2.set_ylim(bottom=1e-4)

    fig.align_ylabels([ax1, ax2])

    if export_png:
        fig.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
        print(f"Figure saved as '{png_filename}'")

    plt.show()


# ── 8. Entry point ────────────────────────────
# IMPORTANT: the if __name__ == "__main__" guard is MANDATORY when using
# multiprocessing on Windows (spawn start method). Without it, each worker
# process would re-execute the top-level code, spawning infinite children.

if __name__ == "__main__":
    A = create_coupling_matrix(N, neighborhood_range)
    results = run_sweep(
        Delta              = Delta,
        omega0             = omega0,
        N                  = N,
        neighborhood_range = neighborhood_range,
        K_points           = K_points,
        T                  = T,
        dt                 = dt,
        transient_frac     = transient_frac,
        seed               = seed,
        A                  = A,
        n_workers          = n_workers,
    )

    print(f"\nKc theoretical : {results['Kc_theory']:.4f}")
    print(f"Kc numerical   : {results['Kc_numerical']:.4f}"
          if results["Kc_numerical"] else "Kc numerical   : not detected")

    plot_results(
        results      = results,
        Delta        = Delta,
        N            = N,
        T            = T,
        dt           = dt,
        export_png   = export_png,
        png_filename = png_filename,
    )