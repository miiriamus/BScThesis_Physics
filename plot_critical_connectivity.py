# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:55:25 2026

@author: Miriam_Ucendo
"""
"""
plot_critical_connectivity.py
─────────────────────────────
Reads kuramoto_critical_connectivity.xlsx and produces two separate figures:
  - fig_delta1.png   (Δ = 1.0)
  - fig_delta01.png  (Δ = 0.1)

Each figure shows a scatter of N vs f* together with the linear regression.

Requirements: openpyxl, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from openpyxl import load_workbook
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

def read_sheet(wb, sheet_name):
    ws = wb[sheet_name]
    N_vals, f_vals = [], []
    for row in ws.iter_rows(min_row=3, values_only=True):
        n, f = row[0], row[1]
        if isinstance(n, (int, float)) and isinstance(f, (int, float)):
            N_vals.append(float(n))
            f_vals.append(float(f))
    return np.array(N_vals), np.array(f_vals)


def linear_regression(xs, ys):
    m, b = np.polyfit(xs, ys, 1)
    r2   = np.corrcoef(xs, ys)[0, 1] ** 2
    return m, b, r2


def make_plot(N, f_star, delta_str, params, outfile):
    m, b, r2 = linear_regression(N, f_star)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    sort_idx = np.argsort(N)
    ax.plot(N[sort_idx], f_star[sort_idx],
        color="#2C6FAC", linewidth=0.8, alpha=0.4, zorder=2)
    
    # ── scatter ──
    ax.scatter(N, f_star,
               s=30, color="#2C6FAC", alpha=0.75,
               edgecolors="white", linewidths=0.4,
               zorder=3, label="Simulation data")


        # ── labels / formatting ──
    ax.set_xlabel(r"$N$  (number of oscillators)", fontsize=12)
    ax.set_ylabel(r"$f^*$  (critical connectivity)", fontsize=12)
    ax.set_title(
        rf"$N$ vs $f^*$  —  Kuramoto model,  $\Delta = {delta_str}$"
        f"\n{params}",
        fontsize=12, pad=10
    )
    
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved: {outfile}")


# ── Main ───────────────────────────────────────────────────────────────────────

XLSX = Path(__file__).parent / "kuramoto_critical_connectivity.xlsx"
PARAMS = r"$K_c = 2.0$,  $K = 40.0$,  $\varepsilon = 0.05$"

wb = load_workbook(XLSX, data_only=True)

N1, f1 = read_sheet(wb, "Delta=1.0")
make_plot(N1, f1,
          delta_str="1.0", params=PARAMS,
          outfile=Path(__file__).parent / "N_vs_f_d1.pdf")

N2, f2 = read_sheet(wb, "Delta=0.1")
make_plot(N2, f2,
          delta_str="0.1", params=PARAMS,
          outfile=Path(__file__).parent / "N_vs_f_d0.1.pdf")