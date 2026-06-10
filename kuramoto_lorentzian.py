# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:30:07 2026

@author: Miriam_Ucendo
"""

"""
kuramoto_lorentzian.py
======================
Kuramoto model in the thermodynamic limit with a Lorentzian frequency distribution.

Compares:
  - Analytical solution (exact, N -> inf):
        r = 0              if K < Kc
        r = sqrt(1 - Kc/K) if K >= Kc,   Kc = 2*Delta
  - Direct RK4 simulation (finite N):
        r(K) = time-average of |<exp(i*theta)>| over steady state

Reference: Kuramoto (1984); Strogatz (2000) Physica D 143, 1-20.

Parameters (edit the CONFIG block below):
    Delta   : Lorentzian half-width at half-maximum
    omega0  : Central frequency (use 0 for the standard canonical case)
    N       : Number of oscillators
    nn      : neighborhood range
    K_points: Number of coupling values in [0, 2*Kc]
    T       : Total integration time per K value
    dt      : RK4 time step
    transient_frac : Fraction of T discarded as transient
    seed    : Random seed for reproducibility
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ──────────────────────────────────────────────
# CONFIG  —  edit here
# ──────────────────────────────────────────────
Delta          = 1        # Lorentzian half-width
omega0         = 0.0      # Central frequency
N              = 10000    # Number of oscillators
nn             = 5000     # interaction with k-nearest neighbors
K_points       = 100      # Resolution of K sweep
T              = 250      # Total integration time
dt             = 0.1      # RK4 step
transient_frac = 0.8      # Fraction discarded as transient
seed           = 130402   # Random seed
export_png     = True     # Save figure as PNG
png_filename = f"MF_N{N}_d{Delta}_T{T}_long.png"
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Adjacent matrix
# ──────────────────────────────────────────────
def create_coupling_matrix(N, nn):
    """
    Creates an NxN adjacency matrix with a specific neighborhood interaction range.
    The main diagonal is always 0.
    
    Parameters:
    ----------
    N : int
        Size of the square matrix (number of oscillators).
    nn : int
        1: Nearest neighbors only (1 step away)
        2: Up to second-nearest neighbors (2 steps away)
        3: Up to third-nearest neighbors (3 steps away)
        ...
        N: All-to-all connection (minus the main diagonal)
        
    Returns:
    -------
    A : ndarray
        The resulting NxN boundary/banded adjacency matrix.
    """
    # 1. Create a coordinate grid of indices from 0 to N-1
    # row_indices is shape (N, 1), col_indices is shape (1, N)
    row_indices, col_indices = np.ogrid[:N, :N]
    
    # 2. Compute the absolute distance between indices for every cell
    diff = np.abs(row_indices - col_indices)
    distance_matrix = np.minimum(diff, N - diff)
    
    # 3. Create a boolean mask: true if distance is > 0 (excludes diagonal) 
    # and <= nn (includes up to the requested neighbor)
    mask = (distance_matrix > 0) & (distance_matrix <= nn)
    
    # 4. Convert the boolean matrix to integers (1s and 0s)
    A = mask.astype(int)
    
    return A

# ── 1. Analytical r solution ────────────────────

def analytical_r(K_array, Kc):
    """
    Exact order parameter for the Lorentzian Kuramoto model (N -> inf).
    Uses the residue trick: the self-consistency integral reduces to
    r = K/2 * r * (1/Delta) for the Lorentzian, giving r=sqrt(1-Kc/K).
    """
    r = np.zeros_like(K_array)
    mask = K_array >= Kc
    r[mask] = np.sqrt(1.0 - Kc / K_array[mask])
    return r

def critical_coupling(Delta):
    """Theoretical critical coupling: Kc = 2*Delta."""
    return 2.0 * Delta

# ── 2. Lorentzian frequency sampling ─────────
def sample_lorentzian(N, Delta, omega0, rng):
    """
    Draw N frequencies from Lorentz(omega0, Delta) via inverse CDF:
        omega = omega0 + Delta * tan(pi*(u - 0.5)),  u ~ Uniform(0,1)
    Truncation is applied at ±50*Delta to avoid extreme outliers
    that would require an impractically small dt.
    """
    u = rng.uniform(0.0, 1.0, N)
    omega = omega0 + Delta * np.tan(np.pi * (u - 0.5))
    clip = 50.0 * Delta
    return np.clip(omega, omega0 - clip, omega0 + clip)

# ── 3. RK4 integrator (vectorised) ───────────
def kuramoto_rhs(theta, omega, K, N):
    """
    Right-hand side of the Kuramoto ODE (vectorised over all N oscillators):
        d theta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)
        sin(A - B) = sin(A)cos(B) - cos(A)sin(B)
    """
    sum_sin = np.sum(np.sin(theta))
    sum_cos = np.sum(np.cos(theta))
    coupling_sum = (K / N) * (sum_sin * np.cos(theta) - sum_cos * np.sin(theta))
    return omega + coupling_sum

def rk4_step(theta, omega, K, N, dt):
    """Single RK4 step."""
    k1 = kuramoto_rhs(theta,             omega, K, N)
    k2 = kuramoto_rhs(theta + 0.5*dt*k1, omega, K, N)
    k3 = kuramoto_rhs(theta + 0.5*dt*k2, omega, K, N)
    k4 = kuramoto_rhs(theta +     dt*k3, omega, K, N)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter(K, omega, theta0, T, dt, transient_frac):
    """
    Integrate the Kuramoto model with RK4 for a single K value.
    Returns the time-averaged |<exp(i*theta)>| over the steady-state window.
    """
    N           = len(theta0)
    n_steps     = int(T / dt)
    n_transient = int(n_steps * transient_frac)
    theta       = theta0.copy()

    r_values = []
    for step in range(n_steps):
        theta = rk4_step(theta, omega, K, N, dt)
        if step >= n_transient:
            rho = np.mean(np.exp(1j * theta))
            r_values.append(np.abs(rho))

    return np.mean(r_values)

# ── 3.2 RK4 integrator with Coupling Matrix ───
def kuramoto_rhs_matrix(theta, omega, K, nn, N, row_idx, col_idx):
    """Right-hand side processing ONLY the active network connections."""
    # Compute phase differences exclusively for connected nodes
    sin_values = np.sin(theta[col_idx] - theta[row_idx])
    
    # Fast sum aggregation mapping back to each specific row node 'i'
    coupling_sum = K / (nn * 2) * np.bincount(row_idx, weights=sin_values, minlength=N)
    
    return omega + coupling_sum

def rk4_step_matrix(theta, omega, K, nn, N, dt, row_idx, col_idx):
    """Single RK4 step forwarding the coupling matrix A."""
    k1 = kuramoto_rhs_matrix(theta,             omega, K, nn, N, row_idx, col_idx)
    k2 = kuramoto_rhs_matrix(theta + 0.5*dt*k1, omega, K, nn, N, row_idx, col_idx)
    k3 = kuramoto_rhs_matrix(theta + 0.5*dt*k2, omega, K, nn, N, row_idx, col_idx)
    k4 = kuramoto_rhs_matrix(theta +     dt*k3, omega, K, nn, N, row_idx, col_idx)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_order_parameter_matrix(K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx):
    """Integrate the Kuramoto model for a single K value over network A."""
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

# ── 4. Numerical Kc detection ─────────────────
def detect_kc_numerical(K_array, r_array, threshold=0.05):
    """
    Return the first K where r exceeds threshold (numerical Kc estimate).
    Returns None if synchronisation is never reached.
    """
    idx = np.argmax(r_array > threshold)
    if r_array[idx] > threshold:
        return K_array[idx]
    return None


# ── 5. Main sweep ─────────────────────────────
def run_sweep(Delta, omega0, N, nn, K_points, T, dt, transient_frac, seed, A):
    """Perform the full K sweep dynamically selecting the fastest algorithm."""
    rng   = np.random.default_rng(seed)
    omega = sample_lorentzian(N, Delta, omega0, rng)

    Kc      = critical_coupling(Delta)
    K_array = np.linspace(0.0, 20.0 * Kc, K_points)
#    theta0  = rng.uniform(0.0, 0.0001 *np.pi, N)
    theta0  = rng.uniform(0.0, 2.0 * np.pi, N)

    # Dynamic check: Is it an all-to-all global coupling network?
    all_to_all_reference = np.ones((N, N)) - np.eye(N)
    is_all_to_all = np.array_equal(A, all_to_all_reference) or np.all(A == 1)

    if is_all_to_all:
        print("-> All-to-all coupling detected. Using fast O(N) algorithm...")
        row_idx, col_idx = None, None
    else:
        print("-> Custom network topology detected. Using structural sparse-coordinate algorithm...")
        # Precompute active connection indices ONCE to prevent CPU stalling inside the loops
        row_idx, col_idx = A.nonzero()
        # since all of them are integers, use as less space as possible
        row_idx = row_idx.astype(np.int32)
        col_idx = col_idx.astype(np.int32)

    print(f"{nn}-nearest neighbor interaction")
    print(f"Kc (theoretical) = {Kc:.4f}")
    print(f"Sweeping {K_points} values of K in [0, {2*Kc:.2f}] ...")

    r_rk4 = np.zeros(K_points)
    for i, K in enumerate(K_array):
        step_start = time.time()
        
        # Select the integration function based on the matrix configuration
        if is_all_to_all:
            r_rk4[i] = simulate_order_parameter(K, omega, theta0, T, dt, transient_frac)
        else:
            r_rk4[i] = simulate_order_parameter_matrix(K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx)
        
        step_time = time.time() - step_start
        print(f"  Step {i+1}/{K_points} (K = {K:.2f}) completed in {step_time:.3f} seconds.")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{K_points} done")

    r_analytical = analytical_r(K_array, Kc)
    Kc_numerical = detect_kc_numerical(K_array, r_rk4)
         
    return {
        "K":            K_array,
        "r_analytical": r_analytical,
        "r_rk4":        r_rk4,
        "Kc_theory":    Kc,
        "Kc_numerical": Kc_numerical,
    }

# ── 6. Plot ───────────────────────────────────
def plot_results(results, Delta, N, T, dt, export_png, png_filename):
    """
    Two-panel figure:
      Top   : r(K) — analytical vs. RK4 simulation + Kc markers
      Bottom: absolute error |r_analytical - r_rk4|
    """
    K            = results["K"]
    r_analytical = results["r_analytical"]
    r_rk4        = results["r_rk4"]
    Kc_t         = results["Kc_theory"]
    Kc_n         = results["Kc_numerical"]
    error        = np.abs(r_analytical - r_rk4)

    # Publication-style settings
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

    # Golden-ratio figure
    fig_w = 7.0
    fig_h = fig_w * (1 + np.sqrt(5)) / 2 * 0.55
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.08)
    ax1   = fig.add_subplot(gs[0])
    ax2   = fig.add_subplot(gs[1], sharex=ax1)

    # ── Top panel ──
    ax1.plot(K, r_analytical, color="#1a1a2e", lw=2.0, ls="-",
             label=r"Analytical: $r = \sqrt{1 - K_c/K}$  $(N\to\infty)$")
    ax1.plot(K, r_rk4, color="#e94560", lw=1.5, ls="--", alpha=0.85,
             label=fr"RK4 simulation ($N={N}$, $T={T}$, $\Delta t={dt}$)")

    # Kc markers
    ax1.axvline(Kc_t, color="#1a1a2e", lw=1.0, ls=":",
                label=fr"$K_c^{{\rm theory}} = 2\Delta = {Kc_t:.3f}$")
    if Kc_n is not None:
        ax1.axvline(Kc_n, color="#e94560", lw=1.0, ls=":",
                    label=fr"$K_c^{{\rm numerical}} \approx {Kc_n:.3f}$")
    # K = 20*Kc marker
    ax1.axvline(20.0 * Kc_t, color="#2ca02c", lw=1.0, ls=":",
                label=fr"$K = 20K_c = {20.0*Kc_t:.3f}$")

    ax1.set_ylabel(r"Order parameter $r$", labelpad=6)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(
        fr"Kuramoto model — Lorentzian distribution "
        fr"($\Delta={Delta}$, $\omega_0={0}$)",
        pad=8, fontsize=11
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Bottom panel: error ──
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


# ── 7. Entry point ────────────────────────────

if __name__ == "__main__":
    A = create_coupling_matrix(N, nn)
    results = run_sweep(
        Delta          = Delta,
        omega0         = omega0,
        N              = N,
        nn             = nn,
        K_points       = K_points,
        T              = T,
        dt             = dt,
        transient_frac = transient_frac,
        seed           = seed,
        A              = A,
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