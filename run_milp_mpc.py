"""
run_milp_mpc.py
===============
Full-year Model Predictive Control (MPC) simulation using the MILP solver.

MPC principle:
  At each timestep t, solve a 24-hour lookahead MILP (horizon Np=24).
  Apply only the FIRST step of the optimal solution to the system.
  Advance t by 1, update initial conditions, repeat.

This is computationally expensive (~seconds per step × 8760 steps).
Progress is printed every 24 steps (once per day).

Interface (identical to run_rule_based.py for compare.py):
    results = run_milp_mpc(sizing, profiles, Batt_template, Hs_template,
                           Np=24, dt=1.0, verbose=True)

Returns
-------
results : dict of np.arrays, one value per hour:
    Pb_c, Pb_d, SOC, lambda_pv, lambda_wt,
    SOC_Hs, H_El2Hs, H_El2Hd, H_Hs2Hd, Pel_in,
    P_RE_B, P_RE_Ele, P_RE_L, P_B_Ele, P_B_L,
    P_curtail, delta_b,
    P_unmet, H_unmet,        ← reliability tracking
    status, solve_time       ← diagnostics
"""

import numpy as np
import time
import os

from solver_islanded import MILP_islanded, extract_first_step


# ─────────────────────────────────────────────────────────────────────────────
# Fallback rule-based step (used when MILP fails at a timestep)
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_step(P_pv, P_wind, P_load, H_demand,
                   SOC_batt, SOC_hs, Batt, Capacities, Hs, dt):
    """
    Single-step rule-based fallback when MILP returns infeasible.
    Mirrors the priority order in rule_based.py.
    """
    eps = 1e-9

    Pb_max   = Batt["Pb_max"]
    eta_c    = Batt["eta_c"]
    eta_d    = Batt["eta_d"]
    Qnom     = Batt["Qnom"]
    SOC_min  = Batt["SOC_min"]
    SOC_max  = Batt["SOC_max"]
    Pnom_El  = Capacities["Pnom_El"]
    eta_El   = 0.6
    Qnom_Hs  = Capacities["Hs"]
    SOC_Hs_min = Hs["SOC_min"]
    SOC_Hs_max = Hs["SOC_max"]

    P_RE = P_pv + P_wind

    # RE → load
    P_RE_L    = min(P_RE, P_load)
    P_deficit = P_load - P_RE_L
    P_RE_left = P_RE - P_RE_L

    # Battery discharge → remaining deficit
    P_b_disch = min(Pb_max,
                    max((SOC_batt - SOC_min) / 100 * Qnom * eta_d / dt, 0))
    P_B_L     = min(P_deficit, P_b_disch)
    P_deficit -= P_B_L

    # Battery charge from surplus
    P_b_charg = min(Pb_max,
                    max((SOC_max - SOC_batt) / 100 * Qnom * eta_c / dt, 0))
    P_RE_B    = min(P_RE_left, P_b_charg)
    P_RE_left -= P_RE_B

    # Electrolyzer from surplus
    P_RE_Ele  = min(P_RE_left, Pnom_El)
    P_RE_left -= P_RE_Ele
    P_curtail = max(P_RE_left, 0.0)

    # Battery SOC update
    delta_soc = (100 / Qnom) * dt * (eta_c * P_RE_B - (1 / eta_d) * P_B_L)
    new_SOC   = np.clip(SOC_batt + delta_soc, SOC_min, SOC_max)

    # Electrolyzer heat
    Pel_in  = P_RE_Ele
    H_total = Pel_in * eta_El

    # H₂ storage headroom
    H_hs_room = max((SOC_Hs_max - SOC_hs) / 100 * Qnom_Hs / dt, 0)
    H_hs_avail = max((SOC_hs - SOC_Hs_min) / 100 * Qnom_Hs / dt, 0)

    H_El2Hd = min(H_total, H_demand)
    H_El2Hs = min(H_total - H_El2Hd, H_hs_room)
    H_remaining = H_demand - H_El2Hd
    H_Hs2Hd    = min(H_remaining, H_hs_avail)
    H_unmet     = max(H_remaining - H_Hs2Hd, 0.0)

    delta_hs = (100 / Qnom_Hs) * dt * (H_El2Hs - H_Hs2Hd)
    new_SOC_Hs = np.clip(SOC_hs + delta_hs, SOC_Hs_min, SOC_Hs_max)

    return {
        "Pb_c": P_RE_B, "Pb_d": P_B_L,
        "SOC": new_SOC, "SOC_Hs": new_SOC_Hs,
        "lambda_pv": P_pv / (P_RE + eps),
        "lambda_wt": P_wind / (P_RE + eps),
        "H_El2Hs": H_El2Hs, "H_El2Hd": H_El2Hd, "H_Hs2Hd": H_Hs2Hd,
        "Pel_in": Pel_in, "P_RE_B": P_RE_B,
        "P_RE_Ele": P_RE_Ele, "P_RE_L": P_RE_L,
        "P_B_Ele": 0.0, "P_B_L": P_B_L,
        "P_curtail": P_curtail, "delta_b": 1.0 if P_RE_B > eps else 0.0,
        "P_unmet": P_deficit, "H_unmet": H_unmet,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation function
# ─────────────────────────────────────────────────────────────────────────────
def run_milp_mpc(sizing, profiles, Batt_template, Hs_template,
                 Np=24, dt=1.0, verbose=True):
    """
    Run full-year MPC simulation with MILP solver.

    Parameters
    ----------
    sizing        : array-like [C_pv, C_wt, C_batt, C_el, C_hs]
    profiles      : dict {'PV','WT','H2','ED'} — capacity factors / reference kW
    Batt_template : dict  battery parameters
    Hs_template   : dict  H₂ storage parameters
    Np            : int   MPC prediction horizon  (default 24 h)
    dt            : float timestep length         (default 1 h)
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

    Cost = {"w_curtail": 1.0, "w_batt": 0.01, "w_H2": 0.5}

    # ── Scale profiles ────────────────────────────────────────────────────
    PV_cf  = profiles["PV"].flatten()
    WT_cf  = profiles["WT"].flatten()
    H2_ref = profiles["H2"].flatten()
    ED_ref = profiles["ED"].flatten()

    Nh = len(ED_ref)    # number of simulation steps (8760)

    P_pv_yr   = PV_cf  * C_pv
    P_wind_yr = WT_cf  * C_wt
    P_load_yr = ED_ref
    H_dem_yr  = H2_ref

    # ── Pre-allocate result arrays ────────────────────────────────────────
    keys = ["Pb_c", "Pb_d", "SOC", "lambda_pv", "lambda_wt",
            "SOC_Hs", "H_El2Hs", "H_El2Hd", "H_Hs2Hd", "Pel_in",
            "P_RE_B", "P_RE_Ele", "P_RE_L", "P_B_Ele", "P_B_L",
            "P_curtail", "delta_b", "P_unmet", "H_unmet",
            "status", "solve_time"]
    results = {k: np.zeros(Nh) for k in keys}

    # ── Initial conditions ────────────────────────────────────────────────
    Batt["SOC_init"] = Batt.get("SOC_init", 50.0)
    Hs["SOC_init"]   = Hs.get("SOC_init",   20.0)

    n_fallback = 0
    t_total    = time.perf_counter()

    if verbose:
        print("="*65)
        print("  MILP-MPC Full-Year Simulation")
        print(f"  Horizon Np={Np}h  |  Steps={Nh}  |  "
              f"C_pv={C_pv:.0f}kW  C_wt={C_wt:.0f}kW")
        print(f"  C_batt={C_batt:.0f}kWh  C_el={C_el:.0f}kW  "
              f"C_hs={C_hs:.0f}kWh")
        print("="*65)
        print(f"  {'Day':>4}  {'Hour':>5}  {'Status':>7}  "
              f"{'SOC_b%':>7}  {'SOC_h%':>7}  {'Time(s)':>8}")
        print("  " + "-"*50)

    # ── MPC loop ──────────────────────────────────────────────────────────
    for t in range(Nh):
        t_step = time.perf_counter()

        # Extract horizon window (wrap at end of year)
        idx  = np.arange(t, t + Np) % Nh
        P_pv_w   = P_pv_yr[idx]
        P_wind_w = P_wind_yr[idx]
        P_load_w = P_load_yr[idx]
        H_dem_w  = H_dem_yr[idx]

        # Solve MILP
        try:
            x, status, fval, mip_gap, _ = MILP_islanded(
                P_pv_w, P_wind_w, P_load_w, H_dem_w,
                Batt, dt, Np, Capacities, Hs, Cost
            )

            if status == 0 and x is not None:
                step = extract_first_step(x, Np)
                # Actual unmet: load balance check
                step["P_unmet"] = max(
                    P_load_w[0] - step["P_RE_L"] - step["P_B_L"], 0.0
                )
                step["H_unmet"] = max(
                    H_dem_w[0] - step["H_El2Hd"] - step["H_Hs2Hd"], 0.0
                )
            else:
                raise ValueError(f"MILP status={status}")

        except Exception:
            # Fallback to single-step rule-based
            step = _fallback_step(
                P_pv_yr[t], P_wind_yr[t], P_load_yr[t], H_dem_yr[t],
                Batt["SOC_init"], Hs["SOC_init"],
                Batt, Capacities, Hs, dt
            )
            status = -1
            n_fallback += 1

        solve_time = time.perf_counter() - t_step

        # ── Store results ─────────────────────────────────────────────────
        for k in keys[:-2]:    # all except status, solve_time
            results[k][t] = step.get(k, 0.0)
        results["status"][t]     = status
        results["solve_time"][t] = solve_time

        # ── Warm-start: update initial SOC for next step ──────────────────
        Batt["SOC_init"] = float(step["SOC"])
        Hs["SOC_init"]   = float(step["SOC_Hs"])

        # ── Progress print (every 24 steps = 1 day) ───────────────────────
        if verbose and (t % 24 == 0 or t == Nh - 1):
            day = t // 24 + 1
            print(f"  {day:>4}  {t:>5}  {int(status):>7}  "
                  f"{step['SOC']:>7.1f}  {step['SOC_Hs']:>7.1f}  "
                  f"{solve_time:>8.3f}")

    elapsed = time.perf_counter() - t_total

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print("  " + "-"*50)
        total_unmet_P = results["P_unmet"].sum()
        total_unmet_H = results["H_unmet"].sum()
        LPSP = total_unmet_P / (P_load_yr.sum() + 1e-9) * 100
        LHSP = total_unmet_H / (H_dem_yr.sum()  + 1e-9) * 100
        print(f"\n  Total time     : {elapsed:.1f} s  "
              f"({elapsed/3600:.2f} h)")
        print(f"  Avg per step   : {elapsed/Nh*1000:.1f} ms")
        print(f"  Fallback steps : {n_fallback} / {Nh}")
        print(f"  LPSP           : {LPSP:.3f} %")
        print(f"  LHSP           : {LHSP:.3f} %")
        print(f"  Final SOC_batt : {results['SOC'][-1]:.1f} %")
        print(f"  Final SOC_H₂   : {results['SOC_Hs'][-1]:.1f} %")

    # Attach metadata
    results["_meta"] = {
        "method":      "MILP-MPC",
        "Np":          Np,
        "n_fallback":  n_fallback,
        "elapsed":     elapsed,
        "sizing":      list(sizing),
        "P_load":      P_load_yr,
        "H_demand":    H_dem_yr,
        "P_pv":        P_pv_yr,
        "P_wind":      P_wind_yr,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (1 week = 168 hours)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import scipy.io as sio

    print("run_milp_mpc.py — quick test (168 steps, first week of year)")

    data     = sio.loadmat("Data/Longyearbyen_profiles.mat")
    profiles = {k: data[k].flatten()[:192]    # 192 = 168 + 24 buffer for horizon
                for k in ["PV", "WT", "H2", "ED"]}
    # Trim profiles to 168 for simulation
    profiles_sim = {k: v[:168] for k, v in profiles.items()}

    Batt_template = {
        "Pb_min": 0.0, "SOC_max": 90.0, "SOC_min": 10.0,
        "eta_c": 0.95, "eta_d": 0.95, "SOC_init": 50.0, "c_rate": 0.5,
    }
    Hs_template = {"SOC_min": 0.0, "SOC_max": 100.0, "SOC_init": 20.0}

    sizing = [200.0, 400.0, 400.0, 150.0, 4000.0]

    results = run_milp_mpc(
        sizing, profiles_sim, Batt_template, Hs_template,
        Np=24, dt=1.0, verbose=True
    )

    print(f"\n  P_unmet total : {results['P_unmet'].sum():.2f} kWh")
    print(f"  Curtailment   : {results['P_curtail'].sum():.2f} kWh")
    print("  Test complete.")
