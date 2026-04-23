"""
run_rule_based.py
=================
Full-year rule-based EMS simulation runner.

Thin wrapper around rule_based.run_rule_based_year() with the same
interface as run_milp_mpc.py so compare.py can treat both identically.

Interface:
    results = run_rule_based(sizing, profiles, Batt_template, Hs_template,
                              dt=1.0, verbose=True)

Returns
-------
results : dict of np.arrays (Nh,) — identical keys to run_milp_mpc output
"""

import numpy as np
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rule_based import run_rule_based_year


def run_rule_based(sizing, profiles, Batt_template, Hs_template,
                   dt=1.0, verbose=True):
    """
    Run full-year rule-based EMS simulation.

    Parameters
    ----------
    sizing        : array-like [C_pv, C_wt, C_batt, C_el, C_hs]
    profiles      : dict {'PV','WT','H2','ED'}
    Batt_template : dict  battery parameters
    Hs_template   : dict  H₂ storage parameters
    dt            : float timestep length  (default 1 h)
    verbose       : bool

    Returns
    -------
    results : dict of np.arrays (Nh,)
    """
    C_pv, C_wt, C_batt, C_el, C_hs = [float(v) for v in sizing]

    # ── Build parameter dicts ─────────────────────────────────────────────
    Batt = dict(Batt_template)
    Batt["Qnom"]   = C_batt
    Batt["Pb_max"] = C_batt * Batt.get("c_rate", 0.5)

    Capacities = {
        "PV": C_pv, "WT": C_wt, "Batt": C_batt,
        "Pnom_El": C_el, "Hs": C_hs,
    }
    Hs = dict(Hs_template)

    # ── Scale profiles ────────────────────────────────────────────────────
    P_pv_yr   = profiles["PV"].flatten() * C_pv
    P_wind_yr = profiles["WT"].flatten() * C_wt
    P_load_yr = profiles["ED"].flatten()
    H_dem_yr  = profiles["H2"].flatten()

    if verbose:
        print("="*65)
        print("  Rule-Based EMS Full-Year Simulation")
        print(f"  C_pv={C_pv:.0f}kW  C_wt={C_wt:.0f}kW  "
              f"C_batt={C_batt:.0f}kWh")
        print(f"  C_el={C_el:.0f}kW  C_hs={C_hs:.0f}kWh")
        print("="*65)

    t_start = time.perf_counter()
    results = run_rule_based_year(
        P_pv_yr, P_wind_yr, P_load_yr, H_dem_yr,
        Batt, Capacities, Hs, dt
    )
    elapsed = time.perf_counter() - t_start

    # ── Add status/solve_time arrays (for interface consistency) ──────────
    Nh = len(P_load_yr)
    results["status"]     = np.zeros(Nh)      # rule-based never fails
    results["solve_time"] = np.full(Nh, elapsed / Nh)

    if verbose:
        LPSP = results["P_unmet"].sum() / (P_load_yr.sum() + 1e-9) * 100
        LHSP = results["H_unmet"].sum() / (H_dem_yr.sum()  + 1e-9) * 100
        print(f"\n  Elapsed time   : {elapsed:.3f} s")
        print(f"  LPSP           : {LPSP:.3f} %")
        print(f"  LHSP           : {LHSP:.3f} %")
        print(f"  Curtailment    : {results['P_curtail'].sum():.1f} kWh")
        print(f"  H₂ stored      : {results['H_El2Hs'].sum():.1f} kWh_th")
        print(f"  Final SOC_batt : {results['SOC'][-1]:.1f} %")
        print(f"  Final SOC_H₂   : {results['SOC_Hs'][-1]:.1f} %")

    # Attach metadata
    results["_meta"] = {
        "method":    "Rule-Based",
        "elapsed":   elapsed,
        "sizing":    list(sizing),
        "P_load":    P_load_yr,
        "H_demand":  H_dem_yr,
        "P_pv":      P_pv_yr,
        "P_wind":    P_wind_yr,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import scipy.io as sio

    print("run_rule_based.py — self-test (full year)")

    data     = sio.loadmat("Data/Longyearbyen_profiles.mat")
    profiles = {k: data[k].flatten() for k in ["PV", "WT", "H2", "ED"]}

    Batt_template = {
        "Pb_min": 0.0, "SOC_max": 90.0, "SOC_min": 10.0,
        "eta_c": 0.95, "eta_d": 0.95, "SOC_init": 50.0, "c_rate": 0.5,
    }
    Hs_template = {"SOC_min": 0.0, "SOC_max": 100.0, "SOC_init": 20.0}
    sizing = [200.0, 400.0, 400.0, 150.0, 4000.0]

    results = run_rule_based(sizing, profiles, Batt_template, Hs_template,
                              verbose=True)

    assert results["P_unmet"].sum() == 0.0, "Unexpected unmet load"
    print("\n  Self-test passed.")
