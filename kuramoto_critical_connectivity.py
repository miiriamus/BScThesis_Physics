# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 02:24:19 2026

@author: Miriam_Ucendo
"""
"""
kuramoto_critical_connectivity.py
==================================
For each N in N_LIST, sweeps nn from 1 upward (f = 2*nn/(N-1)) at fixed
K = 20*Kc and stops as soon as D(f) = |r(f) - r_an| < epsilon.
Saves the critical pair (N, f*) and plots N vs f* with a linear regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
Delta          = 0.1
omega0         = 0.0
#LIST           = np.arange(5, 101, 5)
#LIST           = np.arange(10, 51, 10)
#LIST           = np.arange(10, 101, 20)
N_LIST = [605,615]
#N_LIST         = [150, 250, 350, 450, 550]
epsilon        = 0.05      # threshold for D(f) = |r(f) - r_an|
T              = 50
dt             = 0.1
transient_frac = 0.8
seed           = 130402
export_png     = True
png_filename   = f"kuramoto_N_vs_fstar_e{epsilon}.png"
dat_filename   = f"kuramoto_N_vs_fstar_e{epsilon}.dat"

Kc = 2.0 * Delta
K  = 20.0 * Kc
print(f"Delta={Delta}, Kc={Kc:.4f}, K={K:.4f}, epsilon={epsilon}")
# ──────────────────────────────────────────────


# ── Functions (unchanged from kuramoto_lorentzian.py) ────────────────────────
def create_coupling_matrix(N, nn):
    row_indices, col_indices = np.ogrid[:N, :N]
    diff = np.abs(row_indices - col_indices)
    distance_matrix = np.minimum(diff, N - diff)
    mask = (distance_matrix > 0) & (distance_matrix <= nn)
    return mask.astype(int)

def analytical_r(K_array, Kc):
    r = np.zeros_like(K_array)
    mask = K_array >= Kc
    r[mask] = np.sqrt(1.0 - Kc / K_array[mask])
    return r

def sample_lorentzian(N, Delta, omega0, rng):
    u = rng.uniform(0.0, 1.0, N)
    omega = omega0 + Delta * np.tan(np.pi * (u - 0.5))
    clip = 50.0 * Delta
    return np.clip(omega, omega0 - clip, omega0 + clip)

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


# ── Sweep for a single N, stopping at f* ─────────────────────────────────────
def find_critical_connectivity(N, K, Kc, Delta, omega0, T, dt, transient_frac, seed, epsilon):
    rng    = np.random.default_rng(seed)
    omega  = sample_lorentzian(N, Delta, omega0, rng)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, N)

    r_an = analytical_r(np.array([K]), Kc)[0]

    nn_max = (N) // 2   # maximum meaningful nn for a ring of N nodes
                        # since is defined using the ceiling function, here +1 is added 
                        # to use the flour function of C

    f_star = None
    for nn in range(1, nn_max + 1):
        A = create_coupling_matrix(N, nn)
        row_idx, col_idx = A.nonzero()
        row_idx = row_idx.astype(np.int32)
        col_idx = col_idx.astype(np.int32)

        r_sim = simulate_order_parameter_matrix(
            K, nn, omega, theta0, T, dt, transient_frac, row_idx, col_idx
        )
        f = 2 * nn / (N - 1)
        D = np.abs(r_sim - r_an)

        print(f"    nn={nn:4d}  f={f:.4f}  r={r_sim:.4f}  D={D:.4f}")

        if D < epsilon:
            f_star = f
            print(f"  -> f* = {f_star:.4f} reached at nn={nn} for N={N}")
            break

    if f_star is None:
        print(f"  WARNING: f* not reached for N={N} within nn_max={nn_max}")

    return f_star


# ── Main loop over N_LIST ─────────────────────────────────────────────────────
results = []   # list of (N, f*) pairs

for N in N_LIST:
    print(f"\n{'='*50}")
    print(f"N = {N}")
    t0 = time.time()
    f_star = find_critical_connectivity(
        N, K, Kc, Delta, omega0, T, dt, transient_frac, seed, epsilon
    )
    print(f"  Elapsed: {time.time()-t0:.1f} s")
    if f_star is not None:
        results.append((N, f_star))

results = np.array(results)
N_arr   = results[:, 0]
f_arr   = results[:, 1]

# Save .dat
np.savetxt(dat_filename, results, header="N   f_star", fmt=["%.0f", "%.6f"])
print(f"\nData saved to '{dat_filename}'")
print("Pairs (N, f*):")
for N, f in results:
    print(f"  N={int(N):4d}  f*={f:.4f}")


# ── Linear regression ─────────────────────────────────────────────────────────
slope, intercept, r_value, p_value, std_err = stats.linregress(N_arr, f_arr)
N_fit  = np.linspace(N_arr.min(), N_arr.max(), 300)
f_fit  = slope * N_fit + intercept
print(f"\nLinear regression: f* = {slope:.6f}·N + {intercept:.6f}  (R²={r_value**2:.4f})")


# ── Plot ──────────────────────────────────────────────────────────────────────
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

fig_w = 7.0
fig_h = fig_w * 0.65
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.scatter(N_arr, f_arr, color="#e94560", s=50, zorder=5,
           label=r"Simulated $(N,\, f^*)$")

ax.set_xlabel(r"Number of oscillators $N$", labelpad=6)
ax.set_ylabel(r"Critical connectivity $f^*$", labelpad=6)
ax.legend(loc="best", fontsize=9)
ax.set_title(
    fr"Critical connectivity vs $N$  "
    fr"($K=20K_c={K:.1f}$, $\Delta={Delta}$, $\varepsilon={epsilon}$)",
    pad=8, fontsize=11
)

fig.tight_layout()

if export_png:
    fig.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{png_filename}'")

plt.show()