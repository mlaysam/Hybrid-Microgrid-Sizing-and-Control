"""
rule_based.py
=============
Rule-based Energy Management System (EMS) for an isolated microgrid.

System topology  (identical to solver_islanded.py):
    PV + Wind → Power Bus → Electrical Load
                          → Battery (charge / discharge)
                          → Electrolyzer → H₂ Storage → Heating Demand
                                         → Direct Heating

Priority order (agreed design):
    1. RE serves electrical load directly
    2. RE surplus charges battery
    3. Remaining RE surplus powers electrolyzer
    4. Remaining RE surplus is curtailed
    5. If load not fully met by RE → battery discharges to cover deficit
    6. Electrolyzer heat → direct heating first, then H₂ storage
    7. If heating demand not met by electrolyzer → draw from H₂ storage

All physical constraints (SOC limits, power limits, efficiencies) are
enforced identically to the MILP solver for a fair comparison.

Usage
-----
Instantiate RuleBasedEMS once, then call step() for each timestep.
The controller is stateful (tracks SOC between calls).

    ems = RuleBasedEMS(Batt, Capacities, Hs, dt)
    for t in range(Nh):
        result = ems.step(P_pv[t], P_wind[t], P_load[t], H_demand[t])
"""

import numpy as np


class RuleBasedEMS:
    """
    Rule-based single-step EMS controller.

    Parameters
    ----------
    Batt       : dict  Battery parameters (same keys as solver_islanded.py)
    Capacities : dict  Installed capacities
    Hs         : dict  H₂ storage parameters
    dt         : float Timestep length [h]
    """

    def __init__(self, Batt, Capacities, Hs, dt=1.0):
        # ── Battery ───────────────────────────────────────────────────────────
        self.Pb_max   = Batt["Pb_max"]       # max charge/discharge power  [kW]
        self.SOC_max  = Batt["SOC_max"]       # %
        self.SOC_min  = Batt["SOC_min"]       # %
        self.eta_c    = Batt["eta_c"]         # charging efficiency
        self.eta_d    = Batt["eta_d"]         # discharging efficiency
        self.Qnom     = Batt["Qnom"]          # nominal capacity            [kWh]
        self.SOC      = Batt["SOC_init"]      # current state               %

        # ── Electrolyzer ──────────────────────────────────────────────────────
        self.Pnom_El  = Capacities["Pnom_El"] # rated power                 [kW]
        self.eta_El   = 0.6                    # electrical → thermal

        # ── H₂ storage ────────────────────────────────────────────────────────
        self.Qnom_Hs  = Capacities["Hs"]      # nominal capacity            [kWh]
        self.SOC_Hs_max = Hs["SOC_max"]        # %
        self.SOC_Hs_min = Hs["SOC_min"]        # %
        self.SOC_Hs   = Hs["SOC_init"]         # current state               %

        self.dt = dt

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _batt_charge_headroom(self):
        """Maximum power the battery can absorb this step [kW]."""
        energy_room = (self.SOC_max - self.SOC) / 100.0 * self.Qnom
        # How much power fills that room in dt hours (accounting for efficiency)
        P_max_energy = energy_room * self.eta_c / self.dt
        return min(self.Pb_max, max(P_max_energy, 0.0))

    def _batt_discharge_headroom(self):
        """Maximum power the battery can supply this step [kW]."""
        energy_avail = (self.SOC - self.SOC_min) / 100.0 * self.Qnom
        P_max_energy = energy_avail * self.eta_d / self.dt
        return min(self.Pb_max, max(P_max_energy, 0.0))

    def _hs_store_headroom(self):
        """Maximum thermal energy the H₂ tank can absorb this step [kW_th]."""
        energy_room  = (self.SOC_Hs_max - self.SOC_Hs) / 100.0 * self.Qnom_Hs
        return max(energy_room / self.dt, 0.0)

    def _hs_discharge_headroom(self):
        """Maximum thermal energy the H₂ tank can supply this step [kW_th]."""
        energy_avail = (self.SOC_Hs - self.SOC_Hs_min) / 100.0 * self.Qnom_Hs
        return max(energy_avail / self.dt, 0.0)

    def _update_battery_soc(self, Pb_c, Pb_d):
        """Apply charge/discharge to battery SOC."""
        delta = (100.0 / self.Qnom) * self.dt * (
            self.eta_c * Pb_c - (1.0 / self.eta_d) * Pb_d
        )
        self.SOC = np.clip(self.SOC + delta, self.SOC_min, self.SOC_max)

    def _update_hs_soc(self, H_in, H_out):
        """Apply charge/discharge to H₂ storage SOC."""
        delta = (100.0 / self.Qnom_Hs) * self.dt * (H_in - H_out)
        self.SOC_Hs = np.clip(self.SOC_Hs + delta,
                               self.SOC_Hs_min, self.SOC_Hs_max)

    # ─────────────────────────────────────────────────────────────────────────
    # Main control step
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, P_pv, P_wind, P_load, H_demand):
        """
        Execute one rule-based control step.

        Parameters
        ----------
        P_pv    : float  Available PV power     [kW]
        P_wind  : float  Available wind power   [kW]
        P_load  : float  Electrical load        [kW]
        H_demand: float  Heating demand         [kW_th]

        Returns
        -------
        dict with all power flows and state variables (same keys as
        extract_first_step in solver_islanded.py for direct comparison)
        """

        eps = 1e-9    # numerical tolerance

        # ── Total available RE ─────────────────────────────────────────────
        P_RE_total = P_pv + P_wind    # [kW]

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 1: RE → electrical load
        # ═══════════════════════════════════════════════════════════════════
        P_RE_L     = min(P_RE_total, P_load)
        P_deficit  = P_load - P_RE_L             # still unmet after RE
        P_RE_left  = P_RE_total - P_RE_L         # RE still unallocated

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 2 (deficit path): battery discharges to cover deficit
        # ═══════════════════════════════════════════════════════════════════
        P_B_L = 0.0
        if P_deficit > eps:
            P_B_L     = min(P_deficit, self._batt_discharge_headroom())
            P_deficit -= P_B_L

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 3 (surplus path): RE surplus → charge battery
        # ═══════════════════════════════════════════════════════════════════
        P_RE_B = 0.0
        if P_RE_left > eps:
            P_RE_B    = min(P_RE_left, self._batt_charge_headroom())
            P_RE_left -= P_RE_B

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 4 (surplus path): RE surplus → electrolyzer
        # ═══════════════════════════════════════════════════════════════════
        P_RE_Ele = 0.0
        if P_RE_left > eps:
            P_RE_Ele  = min(P_RE_left, self.Pnom_El)
            P_RE_left -= P_RE_Ele

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 5 (surplus path): curtail whatever remains
        # ═══════════════════════════════════════════════════════════════════
        P_curtail = max(P_RE_left, 0.0)

        # ═══════════════════════════════════════════════════════════════════
        # Battery state update
        # ═══════════════════════════════════════════════════════════════════
        Pb_c = P_RE_B     # battery charges from RE only (islanded)
        Pb_d = P_B_L      # battery discharges to load
        self._update_battery_soc(Pb_c, Pb_d)
        SOC_batt_new = self.SOC

        # ═══════════════════════════════════════════════════════════════════
        # Electrolyzer: total input = RE only (islanded, no battery → Ele)
        # ═══════════════════════════════════════════════════════════════════
        Pel_in   = P_RE_Ele
        H_total  = Pel_in * self.eta_El         # total thermal output [kW_th]

        # ═══════════════════════════════════════════════════════════════════
        # HEATING PRIORITY 6: electrolyzer heat → direct demand first
        # ═══════════════════════════════════════════════════════════════════
        H_El2Hd  = min(H_total, H_demand)
        H_El2Hs  = min(H_total - H_El2Hd, self._hs_store_headroom())
        H_El2Hs  = max(H_El2Hs, 0.0)

        # ═══════════════════════════════════════════════════════════════════
        # HEATING PRIORITY 7: remaining demand → H₂ storage discharge
        # ═══════════════════════════════════════════════════════════════════
        H_remaining = H_demand - H_El2Hd
        H_Hs2Hd    = min(H_remaining, self._hs_discharge_headroom())
        H_Hs2Hd    = max(H_Hs2Hd, 0.0)
        H_unmet     = max(H_remaining - H_Hs2Hd, 0.0)   # track reliability

        # H₂ storage state update
        self._update_hs_soc(H_El2Hs, H_Hs2Hd)
        SOC_Hs_new = self.SOC_Hs

        # ═══════════════════════════════════════════════════════════════════
        # Dispatch fractions (for consistency with MILP results structure)
        # ═══════════════════════════════════════════════════════════════════
        lambda_pv = (P_pv / (P_pv + P_wind)) if (P_pv + P_wind) > eps else 0.0
        lambda_wt = (P_wind / (P_pv + P_wind)) if (P_pv + P_wind) > eps else 0.0

        # ═══════════════════════════════════════════════════════════════════
        # Return (same keys as extract_first_step in solver_islanded.py)
        # ═══════════════════════════════════════════════════════════════════
        return {
            # Power flows
            "Pb_c":         Pb_c,
            "Pb_d":         Pb_d,
            "SOC":          SOC_batt_new,
            "lambda_pv":    lambda_pv,
            "lambda_wt":    lambda_wt,
            "SOC_Hs":       SOC_Hs_new,
            "H_El2Hs":      H_El2Hs,
            "H_El2Hd":      H_El2Hd,
            "H_Hs2Hd":      H_Hs2Hd,
            "Pel_in":       Pel_in,
            "P_RE_B":       P_RE_B,
            "P_RE_Ele":     P_RE_Ele,
            "P_RE_L":       P_RE_L,
            "P_B_Ele":      0.0,   # rule-based: battery does not feed Ele
            "P_B_L":        P_B_L,
            "P_curtail":    P_curtail,
            "delta_b":      1.0 if Pb_c > eps else 0.0,
            # Reliability tracking
            "P_unmet":      P_deficit,    # unmet electrical demand  [kW]
            "H_unmet":      H_unmet,      # unmet heating demand     [kW_th]
        }

    def reset(self, SOC_init, SOC_Hs_init):
        """Reset state (used between simulation runs)."""
        self.SOC    = SOC_init
        self.SOC_Hs = SOC_Hs_init


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run a full-year simulation
# ─────────────────────────────────────────────────────────────────────────────
def run_rule_based_year(P_pv, P_wind, P_load, H_demand,
                        Batt, Capacities, Hs, dt=1.0):
    """
    Run rule-based EMS over a full year and return result arrays.

    Parameters
    ----------
    P_pv, P_wind  : array (Nh,)  available generation profiles   [kW]
    P_load        : array (Nh,)  electrical demand               [kW]
    H_demand      : array (Nh,)  heating demand                  [kW_th]
    Batt, Capacities, Hs, dt : same as RuleBasedEMS constructor

    Returns
    -------
    results : dict of arrays (Nh,) — one value per timestep
    """
    Nh  = len(P_pv)
    ems = RuleBasedEMS(Batt, Capacities, Hs, dt)

    # Pre-allocate
    keys = ["Pb_c", "Pb_d", "SOC", "lambda_pv", "lambda_wt",
            "SOC_Hs", "H_El2Hs", "H_El2Hd", "H_Hs2Hd", "Pel_in",
            "P_RE_B", "P_RE_Ele", "P_RE_L", "P_B_Ele", "P_B_L",
            "P_curtail", "delta_b", "P_unmet", "H_unmet"]
    results = {k: np.zeros(Nh) for k in keys}

    for t in range(Nh):
        r = ems.step(P_pv[t], P_wind[t], P_load[t], H_demand[t])
        for k in keys:
            results[k][t] = r[k]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import scipy.io as sio, os, time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("="*60)
    print("  Rule-based EMS — full-year simulation test")
    print("="*60)

    mat_path = os.path.join("Data", "Longyearbyen_profiles.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError("Run data_generator.py first.")

    data  = sio.loadmat(mat_path)
    PV_cf = data["PV"].flatten()
    WT_cf = data["WT"].flatten()
    H2    = data["H2"].flatten()
    ED    = data["ED"].flatten()

    # Capacities (same placeholder values used in solver_islanded test)
    C_pv, C_wt, C_batt, C_el, C_hs = 300.0, 500.0, 500.0, 200.0, 5000.0

    Batt = {
        "Pb_max":   C_batt * 0.5,
        "Pb_min":   0.0,
        "SOC_max":  90.0,
        "SOC_min":  10.0,
        "eta_c":    0.95,
        "eta_d":    0.95,
        "SOC_init": 50.0,
        "Qnom":     C_batt,
    }
    Capacities = {"PV": C_pv, "WT": C_wt, "Batt": C_batt,
                  "Pnom_El": C_el, "Hs": C_hs}
    Hs  = {"SOC_min": 0.0, "SOC_max": 100.0, "SOC_init": 20.0}
    dt  = 1.0

    P_pv_yr   = PV_cf * C_pv
    P_wind_yr = WT_cf * C_wt
    P_load_yr = ED
    H_dem_yr  = H2

    t0 = time.perf_counter()
    results = run_rule_based_year(P_pv_yr, P_wind_yr, P_load_yr, H_dem_yr,
                                  Batt, Capacities, Hs, dt)
    elapsed = time.perf_counter() - t0

    Nh = len(P_pv_yr)

    # ── KPI summary ───────────────────────────────────────────────────────────
    total_load      = P_load_yr.sum()
    total_heat      = H_dem_yr.sum()
    total_unmet_P   = results["P_unmet"].sum()
    total_unmet_H   = results["H_unmet"].sum()
    total_curtail   = results["P_curtail"].sum()
    total_H2_stored = results["H_El2Hs"].sum()
    LPSP            = total_unmet_P / (total_load + 1e-9) * 100

    print(f"\n── KPI Summary ──────────────────────────────────────────────")
    print(f"  Simulation time       : {elapsed:.3f} s  ({Nh} steps)")
    print(f"  Total elec. demand    : {total_load:>10.1f} kWh")
    print(f"  Unmet elec. demand    : {total_unmet_P:>10.1f} kWh  (LPSP = {LPSP:.2f}%)")
    print(f"  Total heat demand     : {total_heat:>10.1f} kWh_th")
    print(f"  Unmet heat demand     : {total_unmet_H:>10.1f} kWh_th")
    print(f"  Total RE curtailment  : {total_curtail:>10.1f} kWh")
    print(f"  Total H₂ stored       : {total_H2_stored:>10.1f} kWh_th")
    print(f"  Final battery SOC     : {results['SOC'][-1]:.1f} %")
    print(f"  Final H₂ storage SOC  : {results['SOC_Hs'][-1]:.1f} %")
    print(f"────────────────────────────────────────────────────────────────\n")

    # Power balance check
    for t in range(Nh):
        re_used = (results["P_RE_L"][t] + results["P_RE_B"][t]
                   + results["P_RE_Ele"][t] + results["P_curtail"][t])
        re_avail = P_pv_yr[t] + P_wind_yr[t]
        assert abs(re_used - re_avail) < 1e-6, f"RE balance error at t={t}"
    print("  RE power balance check passed for all 8760 steps.")

    # Load balance check
    for t in range(Nh):
        supplied = results["P_RE_L"][t] + results["P_B_L"][t]
        required = P_load_yr[t]
        unmet    = results["P_unmet"][t]
        assert abs(supplied + unmet - required) < 1e-6, f"Load balance error at t={t}"
    print("  Load balance check    passed for all 8760 steps.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    days = np.arange(Nh) / 24.0
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Rule-Based EMS — Full-Year Simulation (Longyearbyen)",
                 fontsize=13)

    axes[0].plot(days, P_pv_yr,                  lw=0.5, color='orange',   label="PV")
    axes[0].plot(days, P_wind_yr,                lw=0.5, color='teal',     label="Wind")
    axes[0].plot(days, P_load_yr,                lw=0.8, color='crimson',  label="Load")
    axes[0].set_ylabel("kW")
    axes[0].set_title("Generation & Load")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].stackplot(days,
                      results["P_RE_L"], results["P_B_L"],
                      results["P_curtail"],
                      labels=["RE→Load", "Batt→Load", "Curtail"],
                      colors=["steelblue", "purple", "lightcoral"],
                      alpha=0.8)
    axes[1].set_ylabel("kW")
    axes[1].set_title("Electrical Power Dispatch")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(days, results["SOC"],    lw=0.8, color='navy',    label="Battery SOC")
    axes[2].plot(days, results["SOC_Hs"], lw=0.8, color='darkgreen', label="H₂ SOC")
    axes[2].set_ylabel("%")
    axes[2].set_title("State of Charge — Battery & H₂ Storage")
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(days, H_dem_yr,              lw=0.8, color='crimson',   label="Heating demand")
    axes[3].plot(days, results["H_El2Hd"],    lw=0.8, color='orange',    label="Elec→Direct heat")
    axes[3].plot(days, results["H_Hs2Hd"],    lw=0.8, color='purple',    label="H₂ storage→Heat")
    axes[3].plot(days, results["H_unmet"],    lw=1.0, color='black',  ls='--', label="Unmet heat")
    axes[3].set_ylabel("kW_th")
    axes[3].set_xlabel("Day of year")
    axes[3].set_title("Heating System Dispatch")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, 365)

    plt.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig_path = "Figures/rule_based_year.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {fig_path}")
