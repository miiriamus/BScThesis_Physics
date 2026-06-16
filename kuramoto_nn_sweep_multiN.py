# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 02:49:45 2026

@author: Miriam_Ucendo
"""
"""
kuramoto_nn_sweep_multiN.py
=======================
Loads the .dat files produced by kuramoto_nn_sweep.py for each N in N_LIST
and plots all f vs. r curves on the same figure, with one colour per N.
The analytical horizontal line r = sqrt(1 - Kc/K) is drawn for reference.

Expected .dat filename format: kuramoto_f_vs_r_N{N}_D{Delta}.dat
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ──────────────────────────────────────────────
# CONFIG  —  must match the nn_sweep runs
# ──────────────────────────────────────────────
Delta   = 0.1
Kc      = 2.0 * Delta
K       = 20.0 * Kc

N_LIST  = [10, 50, 100, 200, 300, 500]   # edit to match the files you have

dat_dir      = "data_LC_to_MF"          # folder where the .dat files live
export_png   = True
png_filename = f"NN_sweep_multiN_d{Delta}_T250.pdf"
# ──────────────────────────────────────────────

# Analytical r for the fixed K
r_theory = np.sqrt(1.0 - Kc / K)
print(f"Analytical r(K={K:.2f}) = {r_theory:.6f}")

# Colour palette — one colour per N
colors = cm.plasma(np.linspace(0.1, 0.85, len(N_LIST)))

# ── Plot ──────────────────────────────────────
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

for N, color in zip(N_LIST, colors):
    fname = os.path.join(dat_dir, f"kuramoto_f_vs_r_N{N}_D{Delta}.dat")
    if not os.path.exists(fname):
        print(f"  WARNING: file not found — {fname}")
        continue
    data = np.loadtxt(fname, comments="#")
    f_values = data[:, 0]
    r_values = data[:, 1]
    ax.plot(f_values, r_values, color=color, lw=1.5, marker="o",
            markersize=3.0, alpha=0.85, label=fr"$N={N}$")
    print(f"  Loaded N={N}: {len(f_values)} points")

# Horizontal analytical line
ax.axhline(r_theory, color="black", lw=2.0, ls="--",
           label=fr"Analytical $r = \sqrt{{1-K_c/K}} = {r_theory:.4f}$  $(N\to\infty)$")

ax.set_xlabel(r"Connectivity fraction $f = 2nn/(N-1)$", labelpad=6)
ax.set_ylabel(r"Order parameter $r$", labelpad=6)
ax.set_xlim(0.0, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=9)
ax.set_title(
    fr"Kuramoto–Lorentzian: $f$ vs $r$ for varying $N$  "
    fr"($K=20K_c={K:.1f}$, $\Delta={Delta}$)",
    pad=8, fontsize=11
)

fig.tight_layout()

if export_png:
    fig.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{png_filename}'")

plt.show()