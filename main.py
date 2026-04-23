"""
main.py
=======
Master script — runs the full project pipeline end-to-end.

Pipeline stages
-------------
  Stage 1  DATA       Load synthetic Longyearbyen profiles
  Stage 2  SIZE-PSO   Run PSO to find optimal component sizing
  Stage 3  SIZE-GA    Run GA  to find optimal component sizing
  Stage 4  SIM-MPC    Full-year MILP-MPC simulation with optimal sizing
  Stage 5  SIM-RB     Full-year rule-based simulation with same sizing


Configuration
-------------
Edit the CONFIG dict below to control which stages run and key parameters.
All results are saved under Results/ and Figures/.

Quick run (< 5 min):
    python main.py  with  RUN_FULL_SIZING=False  (uses pre-set sizing)

Full research run:
    python main.py  with  RUN_FULL_SIZING=True, N_WORKERS=4
"""

import numpy as np
import scipy.io as sio
import os
import time
import warnings

# ==========================================================================
# USER CONFIGURATION
# ==========================================================================
CONFIG = {
    # ── Stage control -----------------------------------------------------
    "RUN_FULL_SIZING":   False,   # True = run PSO+GA (slow); False = use PRESET_SIZING
    "RUN_MILP_MPC":      True,    # True = run full-year MILP-MPC (slow)
    "RUN_RULE_BASED":    True,    # True = run full-year rule-based (fast)
    "RUN_COMPARISON":    True,    # True = compute KPIs and generate plots

    # ── Preset sizing (used when RUN_FULL_SIZING=False) ------------------
    # [C_pv kW, C_wt kW, C_batt kWh, C_el kW, C_hs kWh]
    "PRESET_SIZING": [150.0, 300.0, 300.0, 120.0, 3000.0],

    # ── MPC settings -----------------------------------------------------
    "NP":             24,         # MPC prediction horizon (hours)
    "DT":             1.0,        # timestep (hours)

    # ── Paths ------------------------------------------------------------
    "PROFILES_PATH":  "Data/Longyearbyen_profiles.mat",
    "RESULTS_DIR":    "Results",
    "FIGURES_DIR":    "Figures",
}

# ==========================================================================
# SYSTEM PARAMETERS  (shared across all modules)
# ==========================================================================
BATT_TEMPLATE = {
    "Pb_min":   0.0,
    "SOC_max":  90.0,
    "SOC_min":  10.0,
    "eta_c":    0.95,
    "eta_d":    0.95,
    "SOC_init": 50.0,
    "c_rate":   0.5,          # Pb_max = c_rate × Qnom
}

HS_TEMPLATE = {
    "SOC_min":  0.0,
    "SOC_max":  100.0,
    "SOC_init": 20.0,
}


# ==========================================================================
# Helpers
# ==========================================================================
def _banner(text, width=65):
    print("\n" + "="*width)
    print(f"  {text}")
    print("="*width)


def _save_results(obj, name):
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    path = os.path.join(CONFIG["RESULTS_DIR"], name)
    np.save(path, obj, allow_pickle=True)
    print(f"  Saved → {path}.npy")


def _load_results(name):
    path = os.path.join(CONFIG["RESULTS_DIR"], name + ".npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


# ==========================================================================
# STAGE 1 — DATA
# ==========================================================================
def stage_data():
    _banner("STAGE 1 — Data Generation")
    path = CONFIG["PROFILES_PATH"]
    data     = sio.loadmat(path)
    profiles = {k: data[k].flatten() for k in ["PV", "WT", "H2", "ED"]}
    print(f"\n  Loaded profiles: {len(profiles['PV'])} hourly steps")
    return profiles


# ==========================================================================
# STAGE 2+3 — SIZING  (PSO then GA)
# ==========================================================================
def stage_sizing(profiles):
    _banner("STAGE 2+3 — Component Sizing (PSO + GA)")

    if not CONFIG["RUN_FULL_SIZING"]:
        sizing = np.array(CONFIG["PRESET_SIZING"])
        print(f"  Using preset sizing (RUN_FULL_SIZING=False):")
        labels = ["C_pv", "C_wt", "C_batt", "C_el", "C_hs"]
        for name, val in zip(labels, sizing):
            print(f"    {name} = {val:.0f}")
        return sizing, None, None, None, None
    else:
        None
        # PSO & GA


# ==========================================================================
# STAGE 4 — MILP-MPC SIMULATION
# ==========================================================================
def stage_milp_mpc(sizing, profiles):
    _banner("STAGE 5 — MILP-MPC Full-Year Simulation")

    if not CONFIG["RUN_MILP_MPC"]:
        print("  Skipped (RUN_MILP_MPC=False).")
        return None

    # Check for cached results
    cached = _load_results("milp_results")
    if cached is not None:
        cached_sizing = np.array(cached.get("_meta", {}).get("sizing", []))
        if np.allclose(cached_sizing, sizing, rtol=1e-3):
            print("  Loading cached MILP-MPC results.")
            return cached

    from run_milp_mpc import run_milp_mpc
    results = run_milp_mpc(
        sizing, profiles, BATT_TEMPLATE, HS_TEMPLATE,
        Np=CONFIG["NP"], dt=CONFIG["DT"], verbose=True
    )
    _save_results(results, "milp_results")
    return results


# ==========================================================================
# STAGE 5 — RULE-BASED SIMULATION
# ==========================================================================
def stage_rule_based(sizing, profiles):
    _banner("STAGE 6 — Rule-Based Full-Year Simulation")

    if not CONFIG["RUN_RULE_BASED"]:
        print("  Skipped (RUN_RULE_BASED=False).")
        return None

    from run_rule_based import run_rule_based
    results = run_rule_based(
        sizing, profiles, BATT_TEMPLATE, HS_TEMPLATE,
        dt=CONFIG["DT"], verbose=True
    )
    _save_results(results, "rb_results")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def stage_compare(milp_results, rb_results, sizing, profiles):
    _banner("STAGE 7 — Controller Comparison")

    if not CONFIG["RUN_COMPARISON"]:
        print("  Skipped (RUN_COMPARISON=False).")
        return

    if milp_results is None or rb_results is None:
        print("  Cannot compare — one or both simulation results missing.")
        return

    from compare import compare_controllers, plot_comparison

    table = compare_controllers(
        milp_results, rb_results,
        sizing, profiles,
        BATT_TEMPLATE, HS_TEMPLATE,
        save_path=os.path.join(CONFIG["RESULTS_DIR"], "comparison_table.txt")
    )

    plot_comparison(
        milp_results, rb_results, profiles,
        save_dir=CONFIG["FIGURES_DIR"]
    )

    return table


# ==========================================================================
# ENTRY POINT
# ==========================================================================
def main():
    t0 = time.perf_counter()
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["FIGURES_DIR"], exist_ok=True)

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    _banner("LONGYEARBYEN ISOLATED MICROGRID — FULL PIPELINE", width=65)
    print("  PV + Wind + Battery + Electrolyzer + H₂ Storage")
    print("  MILP-MPC  |  Rule-Based  |  PSO + GA Sizing")
    print("  Site: Svalbard, Norway (78.22°N, 15.63°E)")

    # ── Stage 1: Data -----------------------------------------------------
    profiles = stage_data()

    # ── Stages 2+3: Sizing ------------------------------------------------
    sizing_out = stage_sizing(profiles)
    sizing = sizing_out[0]

    # ── Stage 4: MILP-MPC -------------------------------------------------
    milp_results = stage_milp_mpc(sizing, profiles)

    # ── Stage 5: Rule-based -----------------------------------------------
    rb_results = stage_rule_based(sizing, profiles)

    # ── Stage 6: Compare ─────────────────────────────────────────────────
    stage_compare(milp_results, rb_results, sizing, profiles)

if __name__ == "__main__":
    main()
