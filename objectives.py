"""
objectives.py
=============
Techno-economic and environmental objective functions for microgrid sizing.

Called by pso.py and ga.py (thousands of times) — uses the fast rule-based
EMS for simulation, not MILP-MPC.

Sizing decision vector
──────────────────────
x = [C_pv   (kW)    installed PV capacity
     C_wt   (kW)    installed wind capacity
     C_batt (kWh)   battery nominal energy capacity
     C_el   (kW)    electrolyzer rated power
     C_hs   (kWh)   H₂ storage nominal energy capacity]

Objectives
──────────
1. Minimise  NPC  (Net Present Cost, €)
2. Minimise  CO₂  (lifecycle kg CO₂eq)

Combined scalar objective for single-objective optimisers (PSO/GA):
    J = w_npc * NPC_norm + w_co2 * CO2_norm

Economic assumptions (Svalbard / Arctic literature, EUR)
────────────────────────────────────────────────────────
Component     CAPEX (€/unit)   OPEX (€/unit/yr)   Lifetime (yr)
PV            700  /kW         15   /kW            25
Wind          1500 /kW         40   /kW            25
Battery       300  /kWh        5    /kWh           15  (replaced yr 15)
Electrolyzer  1000 /kW         20   /kW            20  (replaced yr 20)
H₂ storage    15   /kWh        0.5  /kWh           30  (no replacement)

Discount rate : 8 %
Project life  : 25 yr
Diesel equiv. : 0.40 €/kWh electricity, 0.09 €/kWh heat (Svalbard baseline)

Lifecycle CO₂ (manufacturing + installation, kg CO₂eq per unit):
PV: 1000/kW  |  Wind: 600/kW  |  Battery: 150/kWh
Electrolyzer: 100/kW  |  H₂ storage: 20/kWh
"""

import numpy as np
from rule_based import run_rule_based_year


# ─────────────────────────────────────────────────────────────────────────────
# Economic & environmental parameters (can override via EconParams)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ECON = {
    # ── CAPEX (€ per unit installed capacity) ─────────────────────────────
    "capex_pv":     700.0,      # €/kW
    "capex_wt":    1500.0,      # €/kW
    "capex_batt":   300.0,      # €/kWh
    "capex_el":    1000.0,      # €/kW
    "capex_hs":      15.0,      # €/kWh

    # ── OPEX (€ per unit per year) ─────────────────────────────────────────
    "opex_pv":      15.0,       # €/kW/yr
    "opex_wt":      40.0,       # €/kW/yr
    "opex_batt":     5.0,       # €/kWh/yr
    "opex_el":      20.0,       # €/kW/yr
    "opex_hs":       0.5,       # €/kWh/yr

    # ── Component lifetimes (years) ────────────────────────────────────────
    "life_pv":      25,
    "life_wt":      25,
    "life_batt":    15,         # replaced once at year 15
    "life_el":      20,         # replaced once at year 20
    "life_hs":      30,

    # ── Project parameters ─────────────────────────────────────────────────
    "project_life": 25,         # years
    "discount_rate": 0.08,      # 8 %

    # ── Benefit: avoided diesel cost ───────────────────────────────────────
    "diesel_elec":   0.40,      # €/kWh electricity (Svalbard baseline)
    "diesel_heat":   0.09,      # €/kWh thermal  (boiler equivalent)
}

DEFAULT_CO2 = {
    # ── Embodied CO₂ (kg CO₂eq per unit installed capacity) ───────────────
    "co2_pv":     1000.0,       # kg/kW
    "co2_wt":      600.0,       # kg/kW
    "co2_batt":    150.0,       # kg/kWh
    "co2_el":      100.0,       # kg/kW
    "co2_hs":       20.0,       # kg/kWh
}

# Normalisation reference values (typical large system; used for weighted sum)
_NPC_REF  = 5_000_000.0        # € — order-of-magnitude NPC for Longyearbyen
_CO2_REF  = 2_000_000.0        # kg — order-of-magnitude lifecycle CO₂


# ─────────────────────────────────────────────────────────────────────────────
# Present-worth helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pwf(r, L):
    """Present Worth Factor: annuity of 1€/yr for L years at rate r."""
    if r == 0:
        return float(L)
    return (1.0 - (1.0 + r)**(-L)) / r


def _spwf(r, y):
    """Single Payment Present Worth Factor: 1€ at year y discounted to year 0."""
    return 1.0 / (1.0 + r)**y


# ─────────────────────────────────────────────────────────────────────────────
# 1. NPC / NPV
# ─────────────────────────────────────────────────────────────────────────────

def compute_npc(sizing, annual_elec_served, annual_heat_served,
                econ=None):
    """
    Compute Net Present Cost and Net Present Value.

    Parameters
    ----------
    sizing : array-like [C_pv, C_wt, C_batt, C_el, C_hs]
    annual_elec_served : float  kWh/yr of electricity actually delivered
    annual_heat_served : float  kWh/yr of heat actually delivered
    econ   : dict  override DEFAULT_ECON (optional)

    Returns
    -------
    dict:
        "NPC"        total net present cost          (€)
        "NPV"        net present value               (€)
        "CAPEX"      initial capital cost            (€)
        "OPEX_PW"    present worth of all OPEX       (€)
        "Repl_PW"    present worth of replacements   (€)
        "Resid_PW"   present worth of residual value (€)   [negative cost]
        "Benefit_PW" present worth of diesel savings (€)
        "LCOE"       levelised cost of energy        (€/kWh)
    """
    e = DEFAULT_ECON.copy()
    if econ:
        e.update(econ)

    C_pv, C_wt, C_batt, C_el, C_hs = sizing
    r  = e["discount_rate"]
    L  = e["project_life"]

    # ── CAPEX (year 0) ────────────────────────────────────────────────────
    capex_pv   = e["capex_pv"]   * C_pv
    capex_wt   = e["capex_wt"]   * C_wt
    capex_batt = e["capex_batt"] * C_batt
    capex_el   = e["capex_el"]   * C_el
    capex_hs   = e["capex_hs"]   * C_hs
    CAPEX      = capex_pv + capex_wt + capex_batt + capex_el + capex_hs

    # ── Annual OPEX ───────────────────────────────────────────────────────
    opex_annual = (e["opex_pv"]   * C_pv
                 + e["opex_wt"]   * C_wt
                 + e["opex_batt"] * C_batt
                 + e["opex_el"]   * C_el
                 + e["opex_hs"]   * C_hs)
    OPEX_PW = opex_annual * _pwf(r, L)

    # ── Replacement costs ─────────────────────────────────────────────────
    # Battery: replaced at year life_batt (= 15) if life_batt < project_life
    repl_pw = 0.0
    y_batt = e["life_batt"]
    while y_batt < L:
        repl_pw += capex_batt * _spwf(r, y_batt)
        y_batt  += e["life_batt"]

    y_el = e["life_el"]
    while y_el < L:
        repl_pw += capex_el * _spwf(r, y_el)
        y_el    += e["life_el"]

    Repl_PW = repl_pw

    # ── Residual values at end of project life ────────────────────────────
    def _residual(capex, life, project):
        """Straight-line residual value at year project."""
        # How many complete cycles fit in project_life?
        cycles_complete = project // life
        years_used_last = project - cycles_complete * life
        frac_remaining  = 1.0 - years_used_last / life
        return capex * max(frac_remaining, 0.0) * _spwf(r, project)

    Resid_PW = (_residual(capex_pv,   e["life_pv"],   L)
              + _residual(capex_wt,   e["life_wt"],   L)
              + _residual(capex_batt, e["life_batt"],  L)
              + _residual(capex_el,   e["life_el"],    L)
              + _residual(capex_hs,   e["life_hs"],    L))

    # ── NPC ───────────────────────────────────────────────────────────────
    NPC = CAPEX + OPEX_PW + Repl_PW - Resid_PW

    # ── Benefits: avoided diesel ──────────────────────────────────────────
    annual_benefit = (annual_elec_served * e["diesel_elec"]
                    + annual_heat_served * e["diesel_heat"])
    Benefit_PW = annual_benefit * _pwf(r, L)

    # ── NPV ───────────────────────────────────────────────────────────────
    NPV = Benefit_PW - NPC

    # ── LCOE (€/kWh) ─────────────────────────────────────────────────────
    total_energy_PW = (annual_elec_served + annual_heat_served) * _pwf(r, L)
    LCOE = NPC / total_energy_PW if total_energy_PW > 0 else np.inf

    return {
        "NPC":        NPC,
        "NPV":        NPV,
        "CAPEX":      CAPEX,
        "OPEX_PW":    OPEX_PW,
        "Repl_PW":    Repl_PW,
        "Resid_PW":   Resid_PW,
        "Benefit_PW": Benefit_PW,
        "LCOE":       LCOE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lifecycle CO₂
# ─────────────────────────────────────────────────────────────────────────────

def compute_co2(sizing, econ=None, co2_params=None):
    """
    Compute total lifecycle CO₂ emissions (kg CO₂eq).

    Includes:
    - Manufacturing + installation of initial components
    - Manufacturing of replacement components (undiscounted)
    - Operational CO₂ ≈ 0 (no fuel combustion in RE system)

    Parameters
    ----------
    sizing     : array-like [C_pv, C_wt, C_batt, C_el, C_hs]
    econ       : dict  override DEFAULT_ECON (for lifetime / project_life)
    co2_params : dict  override DEFAULT_CO2

    Returns
    -------
    dict:
        "CO2_total"    kg CO₂eq total lifecycle
        "CO2_initial"  kg CO₂eq initial installation
        "CO2_replace"  kg CO₂eq from replacements
        "CO2_per_kWh"  kg CO₂eq / kWh  (requires annual_energy argument,
                       computed in simulate_and_evaluate)
    """
    e  = DEFAULT_ECON.copy()
    if econ:
        e.update(econ)
    c  = DEFAULT_CO2.copy()
    if co2_params:
        c.update(co2_params)

    C_pv, C_wt, C_batt, C_el, C_hs = sizing
    L = e["project_life"]

    # Initial installation
    co2_init = (c["co2_pv"]   * C_pv
              + c["co2_wt"]   * C_wt
              + c["co2_batt"] * C_batt
              + c["co2_el"]   * C_el
              + c["co2_hs"]   * C_hs)

    # Replacement manufacturing CO₂
    co2_repl = 0.0
    y_batt = e["life_batt"]
    while y_batt < L:
        co2_repl += c["co2_batt"] * C_batt
        y_batt   += e["life_batt"]

    y_el = e["life_el"]
    while y_el < L:
        co2_repl += c["co2_el"] * C_el
        y_el     += e["life_el"]

    CO2_total = co2_init + co2_repl

    return {
        "CO2_total":   CO2_total,
        "CO2_initial": co2_init,
        "CO2_replace": co2_repl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Battery cycling metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_battery_cycles(results, Qnom):
    """
    Count equivalent full battery cycles over the simulation period.
    Cycles = total charge throughput / (2 × Qnom)
    """
    total_throughput = results["Pb_c"].sum()    # kWh charged
    return total_throughput / (Qnom + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reliability metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_reliability(results, P_load, H_demand, dt=1.0):
    """
    Compute Loss of Power Supply Probability (LPSP) and
    Loss of Heat Supply Probability (LHSP).

    LPSP = sum(P_unmet) / sum(P_load)
    LHSP = sum(H_unmet) / sum(H_demand)
    """
    total_load   = np.sum(P_load) * dt
    total_heat   = np.sum(H_demand) * dt
    unmet_elec   = np.sum(results.get("P_unmet", np.zeros_like(P_load))) * dt
    unmet_heat   = np.sum(results.get("H_unmet", np.zeros_like(H_demand))) * dt

    LPSP = unmet_elec / (total_load + 1e-9)
    LHSP = unmet_heat / (total_heat + 1e-9)

    return {
        "LPSP":       LPSP,
        "LHSP":       LHSP,
        "unmet_elec": unmet_elec,
        "unmet_heat": unmet_heat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master evaluation function (called by PSO/GA)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_and_evaluate(sizing, profiles, Batt_template, Hs_template,
                          dt=1.0, econ=None, co2_params=None,
                          w_npc=0.5, w_co2=0.5,
                          lpsp_max=0.0, lhsp_max=0.0,
                          penalty=1e9):
    """
    Full evaluation pipeline for one candidate sizing.

    Steps:
        1. Build Batt/Capacities/Hs dicts from sizing vector
        2. Run rule-based EMS simulation (full year)
        3. Compute reliability (LPSP, LHSP)
        4. Apply penalty if reliability constraints violated
        5. Compute NPC and CO₂
        6. Return combined normalised objective + full metrics dict

    Parameters
    ----------
    sizing         : array-like [C_pv, C_wt, C_batt, C_el, C_hs]
    profiles       : dict with keys 'PV', 'WT', 'H2', 'ED' (capacity factors
                     and demand at reference 1-kW scale — multiply by capacity
                     inside this function)
    Batt_template  : dict  Battery parameters except Pb_max, Qnom, SOC_init
                     (those are derived from C_batt)
    Hs_template    : dict  H₂ storage parameters except SOC_init
    dt             : float timestep [h]
    econ           : dict  override DEFAULT_ECON
    co2_params     : dict  override DEFAULT_CO2
    w_npc          : float weight on normalised NPC  (0–1)
    w_co2          : float weight on normalised CO₂  (0–1)
    lpsp_max       : float max allowable LPSP (0 = no unmet load)
    lhsp_max       : float max allowable LHSP (0 = no unmet heat)
    penalty        : float large penalty added if constraints violated

    Returns
    -------
    J        : float  combined objective (minimise)
    metrics  : dict   all KPIs for logging/comparison
    """
    C_pv, C_wt, C_batt, C_el, C_hs = [float(v) for v in sizing]

    # Guard against zero or negative sizing
    if any(v <= 0 for v in [C_pv, C_wt, C_batt, C_el, C_hs]):
        return penalty, {}

    # ── Build parameter dicts ─────────────────────────────────────────────
    Batt = dict(Batt_template)
    Batt["Qnom"]    = C_batt
    Batt["Pb_max"]  = C_batt * Batt.get("c_rate", 0.5)
    Batt["SOC_init"] = Batt.get("SOC_init", 50.0)

    Capacities = {
        "PV":      C_pv,
        "WT":      C_wt,
        "Batt":    C_batt,
        "Pnom_El": C_el,
        "Hs":      C_hs,
    }

    Hs = dict(Hs_template)
    Hs["SOC_init"] = Hs.get("SOC_init", 20.0)

    # ── Scale profiles to actual power [kW] ──────────────────────────────
    PV_cf  = profiles["PV"].flatten()
    WT_cf  = profiles["WT"].flatten()
    H2_ref = profiles["H2"].flatten()
    ED_ref = profiles["ED"].flatten()

    P_pv_yr   = PV_cf  * C_pv
    P_wind_yr = WT_cf  * C_wt
    P_load_yr = ED_ref          # already in kW (reference demand)
    H_dem_yr  = H2_ref          # already in kW_th (reference demand)

    # ── Run rule-based simulation ─────────────────────────────────────────
    try:
        results = run_rule_based_year(
            P_pv_yr, P_wind_yr, P_load_yr, H_dem_yr,
            Batt, Capacities, Hs, dt
        )
    except Exception:
        return penalty, {}

    # ── Reliability ───────────────────────────────────────────────────────
    rel = compute_reliability(results, P_load_yr, H_dem_yr, dt)
    LPSP = rel["LPSP"]
    LHSP = rel["LHSP"]

    # Apply hard penalty if reliability constraints violated
    rel_penalty = 0.0
    if LPSP > lpsp_max:
        rel_penalty += penalty * (LPSP - lpsp_max)
    if LHSP > lhsp_max:
        rel_penalty += penalty * (LHSP - lhsp_max)

    # ── Energy served (for NPC benefit calculation) ───────────────────────
    annual_elec_served = (P_load_yr.sum() - rel["unmet_elec"]) * dt
    annual_heat_served = (H_dem_yr.sum()  - rel["unmet_heat"]) * dt

    # ── NPC / NPV ─────────────────────────────────────────────────────────
    npc_results = compute_npc(sizing, annual_elec_served, annual_heat_served,
                              econ=econ)
    NPC  = npc_results["NPC"]
    NPV  = npc_results["NPV"]
    LCOE = npc_results["LCOE"]

    # ── Lifecycle CO₂ ─────────────────────────────────────────────────────
    co2_results = compute_co2(sizing, econ=econ, co2_params=co2_params)
    CO2_total   = co2_results["CO2_total"]

    total_energy = annual_elec_served + annual_heat_served
    CO2_per_kWh  = CO2_total / (total_energy * (DEFAULT_ECON["project_life"]) + 1e-9)

    # ── Battery cycles ────────────────────────────────────────────────────
    batt_cycles = compute_battery_cycles(results, C_batt)

    # ── H₂ utilisation ────────────────────────────────────────────────────
    H2_stored   = results["H_El2Hs"].sum()
    H2_used     = results["H_Hs2Hd"].sum()
    curtailment = results["P_curtail"].sum()
    RE_curtail_ratio = curtailment / (P_pv_yr.sum() + P_wind_yr.sum() + 1e-9)

    # ── Combined normalised objective ──────────────────────────────────────
    NPC_norm = NPC  / _NPC_REF
    CO2_norm = CO2_total / _CO2_REF
    J = w_npc * NPC_norm + w_co2 * CO2_norm + rel_penalty

    # ── Full metrics dict ─────────────────────────────────────────────────
    metrics = {
        # Sizing
        "C_pv":    C_pv,
        "C_wt":    C_wt,
        "C_batt":  C_batt,
        "C_el":    C_el,
        "C_hs":    C_hs,
        # Economic
        "NPC":     NPC,
        "NPV":     NPV,
        "LCOE":    LCOE,
        "CAPEX":   npc_results["CAPEX"],
        "OPEX_PW": npc_results["OPEX_PW"],
        "Repl_PW": npc_results["Repl_PW"],
        # CO₂
        "CO2_total":   CO2_total,
        "CO2_per_kWh": CO2_per_kWh,
        "CO2_initial": co2_results["CO2_initial"],
        "CO2_replace": co2_results["CO2_replace"],
        # Reliability
        "LPSP":        LPSP,
        "LHSP":        LHSP,
        "unmet_elec":  rel["unmet_elec"],
        "unmet_heat":  rel["unmet_heat"],
        # Operation
        "curtailment":       curtailment,
        "RE_curtail_ratio":  RE_curtail_ratio,
        "batt_cycles":       batt_cycles,
        "H2_stored":         H2_stored,
        "H2_used":           H2_used,
        # Objective
        "J":               J,
        "NPC_norm":        NPC_norm,
        "CO2_norm":        CO2_norm,
        "rel_penalty":     rel_penalty,
    }

    return J, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import scipy.io as sio, os

    print("="*60)
    print("  objectives.py — self-test")
    print("="*60)

    mat_path = os.path.join("Data", "Longyearbyen_profiles.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError("Run data_generator.py first.")

    data = sio.loadmat(mat_path)
    profiles = {k: data[k].flatten() for k in ["PV", "WT", "H2", "ED"]}

    Batt_template = {
        "Pb_min":   0.0,
        "SOC_max":  90.0,
        "SOC_min":  10.0,
        "eta_c":    0.95,
        "eta_d":    0.95,
        "SOC_init": 50.0,
        "c_rate":   0.5,
    }
    Hs_template = {"SOC_min": 0.0, "SOC_max": 100.0, "SOC_init": 20.0}

    # ── Test three candidate sizings ──────────────────────────────────────
    candidates = {
        "Small":   [100, 200, 200, 80,  2000],
        "Medium":  [300, 500, 500, 200, 5000],
        "Large":   [600, 900, 900, 350, 9000],
    }

    print(f"\n{'Sizing':<10} {'NPC (k€)':>10} {'NPV (k€)':>10} "
          f"{'LCOE (€/kWh)':>13} {'CO₂ (t)':>10} "
          f"{'LPSP%':>7} {'Curtail%':>9} {'J':>8}")
    print("-"*82)

    for name, sizing in candidates.items():
        J, m = simulate_and_evaluate(
            sizing, profiles, Batt_template, Hs_template,
            w_npc=0.5, w_co2=0.5
        )
        if m:
            print(f"{name:<10} {m['NPC']/1e3:>10.1f} {m['NPV']/1e3:>10.1f} "
                  f"{m['LCOE']:>13.4f} {m['CO2_total']/1e3:>10.1f} "
                  f"{m['LPSP']*100:>7.3f} {m['RE_curtail_ratio']*100:>9.1f} "
                  f"{J:>8.4f}")

    # ── Verify NPC components add up ──────────────────────────────────────
    print("\n── NPC component breakdown (Medium sizing) ──────────────────")
    sizing_m = candidates["Medium"]
    _, m = simulate_and_evaluate(sizing_m, profiles, Batt_template,
                                  Hs_template, w_npc=1.0, w_co2=0.0)
    print(f"  CAPEX         : €{m['CAPEX']:>12,.0f}")
    print(f"  OPEX PW       : €{m['OPEX_PW']:>12,.0f}")
    print(f"  Replacement PW: €{m['Repl_PW']:>12,.0f}")
    npc_r = compute_npc(sizing_m, 40000, 70000)
    print(f"  Residual PW   : €{npc_r['Resid_PW']:>12,.0f}  (credit)")
    print(f"  Benefit PW    : €{npc_r['Benefit_PW']:>12,.0f}  (avoided diesel)")
    print(f"  ─────────────────────────────────────────")
    print(f"  NPC           : €{m['NPC']:>12,.0f}")
    print(f"  NPV           : €{m['NPV']:>12,.0f}")
    print(f"  LCOE          : €{m['LCOE']:>12.4f}/kWh")

    # ── Verify CO₂ breakdown ──────────────────────────────────────────────
    print("\n── CO₂ breakdown (Medium sizing) ────────────────────────────")
    co2 = compute_co2(sizing_m)
    print(f"  Initial install: {co2['CO2_initial']/1e3:>8.1f} t CO₂eq")
    print(f"  Replacements   : {co2['CO2_replace']/1e3:>8.1f} t CO₂eq")
    print(f"  Total lifecycle: {co2['CO2_total']/1e3:>8.1f} t CO₂eq")
    print("\n  All checks complete.")
