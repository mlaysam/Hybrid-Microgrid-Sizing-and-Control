"""
compare.py
==========
Quantitative comparison between MILP-MPC and rule-based EMS.

Computes 6 KPI categories:
  1. Reliability     LPSP, LHSP, unmet energy
  2. Economics       NPC, NPV, LCOE (for the operating year)
  3. Environment     CO₂ lifecycle, CO₂/kWh
  4. Battery health  equivalent full cycles, depth-of-discharge distribution
  5. H₂ system       H₂ stored, H₂ used, storage utilisation
  6. Operations      curtailment, RE utilisation, electrolyzer utilisation

Usage
-----
    from compare import compare_controllers, plot_comparison
    table = compare_controllers(milp_results, rb_results,
                                sizing, profiles,
                                Batt_template, Hs_template)
    plot_comparison(milp_results, rb_results, profiles)
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from objectives import (compute_npc, compute_co2,
                        compute_battery_cycles, compute_reliability)


# -----------------------------------------------------------------------------
# 1.  KPI computation
# -----------------------------------------------------------------------------

def _battery_dod_stats(SOC, SOC_min, SOC_max):
    """
    Compute depth-of-discharge (DoD) statistics from SOC array.
    DoD(t) = (SOC_max - SOC(t)) / (SOC_max - SOC_min)
    """
    span = SOC_max - SOC_min + 1e-9
    DoD  = (SOC_max - SOC) / span
    DoD  = np.clip(DoD, 0, 1)
    return {
        "DoD_mean":   float(DoD.mean()),
        "DoD_max":    float(DoD.max()),
        "DoD_p95":    float(np.percentile(DoD, 95)),
    }


def compute_kpis(results, sizing, profiles,
                 Batt_template, Hs_template,
                 econ=None, co2_params=None, dt=1.0):
    """
    Compute all KPIs for one controller's results.

    Parameters
    ----------
    results       : dict of arrays from run_milp_mpc or run_rule_based
    sizing        : array [C_pv, C_wt, C_batt, C_el, C_hs]
    profiles      : dict {'PV','WT','H2','ED'}
    Batt_template : dict
    Hs_template   : dict
    econ          : dict  override economic parameters
    co2_params    : dict  override CO₂ parameters
    dt            : float

    Returns
    -------
    kpis : dict   flat dict of scalar KPIs
    """
    C_pv, C_wt, C_batt, C_el, C_hs = [float(v) for v in sizing]

    P_load  = results["_meta"]["P_load"]
    H_dem   = results["_meta"]["H_demand"]
    P_pv    = results["_meta"]["P_pv"]
    P_wind  = results["_meta"]["P_wind"]
    Nh      = len(P_load)

    # -- 1. Reliability ----------------------------------------------------
    rel = compute_reliability(results, P_load, H_dem, dt)

    # -- 2. Economics ------------------------------------------------------
    annual_elec = (P_load.sum() - rel["unmet_elec"]) * dt
    annual_heat = (H_dem.sum()  - rel["unmet_heat"]) * dt
    npc = compute_npc(sizing, annual_elec, annual_heat, econ=econ)

    # -- 3. CO₂ ------------------------------------------------------------
    co2 = compute_co2(sizing, econ=econ, co2_params=co2_params)
    total_energy_lifetime = (annual_elec + annual_heat) * 25   # 25-yr project
    co2_per_kWh = co2["CO2_total"] / (total_energy_lifetime + 1e-9)

    # -- 4. Battery health -------------------------------------------------
    cycles  = compute_battery_cycles(results, C_batt)
    SOC_min = Batt_template.get("SOC_min", 10.0)
    SOC_max = Batt_template.get("SOC_max", 90.0)
    dod     = _battery_dod_stats(results["SOC"], SOC_min, SOC_max)
    charge_throughput = results["Pb_c"].sum() * dt    # kWh

    # -- 5. H₂ system ------------------------------------------------------
    H2_stored    = results["H_El2Hs"].sum() * dt
    H2_used      = results["H_Hs2Hd"].sum() * dt
    H2_direct    = results["H_El2Hd"].sum() * dt
    H2_util      = H2_used / (H2_stored + 1e-9)
    Hs_soc_mean  = results["SOC_Hs"].mean()
    electrolyzer_util = results["Pel_in"].mean() / (C_el + 1e-9)

    # -- 6. Operations -----------------------------------------------------
    RE_total     = (P_pv + P_wind).sum() * dt
    curtailment  = results["P_curtail"].sum() * dt
    RE_used      = RE_total - curtailment
    RE_util      = RE_used / (RE_total + 1e-9)
    RE_curtail_r = curtailment / (RE_total + 1e-9)

    # Simulation metadata
    method    = results["_meta"].get("method", "Unknown")
    elapsed   = results["_meta"].get("elapsed", 0.0)
    n_fall    = results["_meta"].get("n_fallback", 0)

    return {
        # Identity
        "method":              method,
        # Reliability
        "LPSP_%":              rel["LPSP"] * 100,
        "LHSP_%":              rel["LHSP"] * 100,
        "unmet_elec_kWh":      rel["unmet_elec"],
        "unmet_heat_kWh":      rel["unmet_heat"],
        # Economics
        "NPC_EUR":             npc["NPC"],
        "NPV_EUR":             npc["NPV"],
        "LCOE_EUR_kWh":        npc["LCOE"],
        "CAPEX_EUR":           npc["CAPEX"],
        # CO₂
        "CO2_total_tonne":     co2["CO2_total"] / 1e3,
        "CO2_per_kWh_gCO2":    co2_per_kWh * 1e3,   # g CO₂eq/kWh
        # Battery
        "batt_cycles":         cycles,
        "DoD_mean_%":          dod["DoD_mean"] * 100,
        "DoD_max_%":           dod["DoD_max"]  * 100,
        "DoD_p95_%":           dod["DoD_p95"]  * 100,
        "charge_throughput_kWh": charge_throughput,
        # H₂
        "H2_stored_kWh":       H2_stored,
        "H2_used_kWh":         H2_used,
        "H2_direct_kWh":       H2_direct,
        "H2_roundtrip_%":      H2_util * 100,
        "Hs_SOC_mean_%":       Hs_soc_mean,
        "electrolyzer_util_%": electrolyzer_util * 100,
        # Operations
        "RE_available_kWh":    RE_total,
        "RE_utilised_%":       RE_util * 100,
        "curtailment_kWh":     curtailment,
        "curtailment_%":       RE_curtail_r * 100,
        # Simulation
        "sim_time_s":          elapsed,
        "fallback_steps":      n_fall,
    }


# =============================================================================
# 2.  Comparison table
# =============================================================================

def compare_controllers(milp_results, rb_results,
                        sizing, profiles,
                        Batt_template, Hs_template,
                        econ=None, co2_params=None, dt=1.0,
                        save_path="Results/comparison_table.txt"):
    """
    Compute KPIs for both controllers and print a side-by-side table.

    Returns
    -------
    table : dict  {'MILP-MPC': kpis_dict, 'Rule-Based': kpis_dict}
    """
    print("\n" + "="*70)
    print("  Computing KPIs for MILP-MPC ...")
    milp_kpis = compute_kpis(milp_results, sizing, profiles,
                              Batt_template, Hs_template, econ, co2_params, dt)

    print("  Computing KPIs for Rule-Based ...")
    rb_kpis   = compute_kpis(rb_results,   sizing, profiles,
                              Batt_template, Hs_template, econ, co2_params, dt)

    # ── Print table -----------------------------------------------------------
    rows = [
        # (label, key, unit, format)
        ("-- RELIABILITY ----------------------------------------", None, "", ""),
        ("LPSP",              "LPSP_%",             "%",         ".3f"),
        ("LHSP",              "LHSP_%",             "%",         ".3f"),
        ("Unmet electricity", "unmet_elec_kWh",     "kWh",       ".1f"),
        ("Unmet heat",        "unmet_heat_kWh",     "kWh_th",    ".1f"),
        ("-- ECONOMICS ----------------------------------------", None, "", ""),
        ("NPC",               "NPC_EUR",            "€",         ",.0f"),
        ("NPV",               "NPV_EUR",            "€",         ",.0f"),
        ("LCOE",              "LCOE_EUR_kWh",       "€/kWh",     ".4f"),
        ("CAPEX",             "CAPEX_EUR",          "€",         ",.0f"),
        ("-- CO₂  ---------------------------------------------", None, "", ""),
        ("Lifecycle CO₂",     "CO2_total_tonne",    "t CO₂eq",   ".1f"),
        ("CO₂ intensity",     "CO2_per_kWh_gCO2",  "g/kWh",     ".2f"),
        ("-- BATTERY ------------------------------------------", None, "", ""),
        ("Equiv. full cycles","batt_cycles",        "",          ".1f"),
        ("Mean DoD",          "DoD_mean_%",         "%",         ".1f"),
        ("P95 DoD",           "DoD_p95_%",          "%",         ".1f"),
        ("Charge throughput", "charge_throughput_kWh", "kWh",   ",.0f"),
        ("-- H₂ SYSTEM ----------------------------------------", None, "", ""),
        ("H₂ stored",         "H2_stored_kWh",      "kWh_th",   ",.0f"),
        ("H₂ used (heating)", "H2_used_kWh",        "kWh_th",   ",.0f"),
        ("H₂ direct heat",    "H2_direct_kWh",      "kWh_th",   ",.0f"),
        ("H₂ round-trip util","H2_roundtrip_%",     "%",         ".1f"),
        ("H₂ store SOC mean", "Hs_SOC_mean_%",      "%",         ".1f"),
        ("Electrolyzer util", "electrolyzer_util_%","%",         ".1f"),
        ("-- OPERATIONS ----------------------------------------", None, "", ""),
        ("RE available",      "RE_available_kWh",   "kWh",      ",.0f"),
        ("RE utilised",       "RE_utilised_%",      "%",         ".1f"),
        ("Curtailment",       "curtailment_kWh",    "kWh",      ",.0f"),
        ("Curtailment ratio", "curtailment_%",      "%",         ".1f"),
        ("-- SIMULATION ----------------------------------------", None, "", ""),
        ("Simulation time",   "sim_time_s",         "s",         ".2f"),
        ("Fallback steps",    "fallback_steps",     "",          ".0f"),
    ]

    w = 26   # label width
    hdr = (f"\n  {'KPI':<{w}}  {'MILP-MPC':>15}  {'Rule-Based':>15}  "
           f"{'Unit':<12}  {'Δ (MILP−RB)':>14}")
    sep = "  " + "-"*80

    lines = [hdr, sep]
    print(hdr)
    print(sep)

    for label, key, unit, fmt in rows:
        if key is None:
            line = f"\n  {label}"
            print(line)
            lines.append(line)
            continue

        m_val = milp_kpis.get(key, float("nan"))
        r_val = rb_kpis.get(key, float("nan"))
        try:
            delta = m_val - r_val
            m_str = format(m_val, fmt)
            r_str = format(r_val, fmt)
            d_str = format(delta, fmt)
            better = ""
            # Mark which is better (lower = better for most KPIs)
            better_lower = ["LPSP_%","LHSP_%","unmet_elec_kWh","unmet_heat_kWh",
                            "NPC_EUR","LCOE_EUR_kWh","CO2_total_tonne",
                            "CO2_per_kWh_gCO2","batt_cycles","DoD_mean_%",
                            "DoD_p95_%","curtailment_kWh","curtailment_%",
                            "sim_time_s","fallback_steps"]
            better_higher = ["RE_utilised_%","H2_stored_kWh","H2_used_kWh",
                             "electrolyzer_util_%","H2_roundtrip_%"]
            if key in better_lower:
                better = "  ← MILP" if delta < -1e-9 else ("  ← RB" if delta > 1e-9 else "")
            elif key in better_higher:
                better = "  ← MILP" if delta > 1e-9 else ("  ← RB" if delta < -1e-9 else "")
        except Exception:
            m_str = str(m_val)
            r_str = str(r_val)
            d_str = "-"
            better = ""

        line = (f"  {label:<{w}}  {m_str:>15}  {r_str:>15}  "
                f"{unit:<12}  {d_str:>14}{better}")
        print(line)
        lines.append(line)

    print(sep)

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Table saved → {save_path}")

    return {"MILP-MPC": milp_kpis, "Rule-Based": rb_kpis}


# =============================================================================
# 3.  Comparison plots
# =============================================================================
def plot_comparison(milp_results, rb_results, profiles,
                    save_dir="Figures"):
    """
    Generate 4 comparison figure panels:
      Fig 1 — SOC evolution (battery + H₂)
      Fig 2 — Electrical power dispatch stacked
      Fig 3 — Heating system dispatch
      Fig 4 — KPI radar chart
    """
    os.makedirs(save_dir, exist_ok=True)
    days  = np.arange(8760) / 24.0
    P_load = milp_results["_meta"]["P_load"]
    H_dem  = milp_results["_meta"]["H_demand"]

    # -- Fig 1: SOC evolution ------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("SOC Evolution: MILP-MPC vs Rule-Based", fontsize=13)

    axes[0].plot(days, milp_results["SOC"], lw=0.8, color="navy",
                 label="MILP-MPC")
    axes[0].plot(days, rb_results["SOC"],   lw=0.8, color="crimson",
                 ls="--", label="Rule-Based")
    axes[0].set_ylabel("Battery SOC (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(days, milp_results["SOC_Hs"], lw=0.8, color="darkgreen",
                 label="MILP-MPC")
    axes[1].plot(days, rb_results["SOC_Hs"],   lw=0.8, color="orange",
                 ls="--", label="Rule-Based")
    axes[1].set_ylabel("H₂ Storage SOC (%)")
    axes[1].set_xlabel("Day of year")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlim(0, 365)

    plt.tight_layout()
    p = os.path.join(save_dir, "compare_SOC.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")

    # -- Fig 2: Electrical dispatch (one winter week for clarity) ---------
    slice_h = slice(0, 168)     # first week of January
    d_week  = days[slice_h]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Electrical Dispatch — First Week (Jan)\n"
                 "MILP-MPC (top) vs Rule-Based (bottom)", fontsize=12)

    for ax, res, title in zip(axes,
                               [milp_results, rb_results],
                               ["MILP-MPC", "Rule-Based"]):
        ax.stackplot(d_week,
                     res["P_RE_L"][slice_h],
                     res["P_B_L"][slice_h],
                     res["P_curtail"][slice_h],
                     labels=["RE→Load", "Batt→Load", "Curtailed"],
                     colors=["steelblue", "purple", "lightcoral"],
                     alpha=0.85)
        ax.plot(d_week, P_load[slice_h], color="black",
                lw=1.5, label="Load demand")
        ax.set_ylabel("kW")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Day of year")
    plt.tight_layout()
    p = os.path.join(save_dir, "compare_elec_dispatch.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")

    # -- Fig 3: Heating dispatch (winter vs summer) ------------------------
    slices   = [slice(0, 24*30), slice(24*180, 24*210)]
    s_labels = ["January (polar night)", "June–July (midnight sun)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Heating System Dispatch", fontsize=13)

    for col, (sl, slabel) in enumerate(zip(slices, s_labels)):
        d_sl = days[sl]
        for row, (res, rlabel) in enumerate(
                zip([milp_results, rb_results], ["MILP-MPC", "Rule-Based"])):
            ax = axes[row][col]
            ax.fill_between(d_sl, H_dem[sl],
                            color="lightcoral", alpha=0.4, label="Demand")
            ax.plot(d_sl, res["H_El2Hd"][sl],
                    color="orange", lw=1.0, label="Elec→Direct")
            ax.plot(d_sl, res["H_Hs2Hd"][sl],
                    color="purple", lw=1.0, label="H₂→Heat")
            ax.plot(d_sl, res["H_unmet"][sl],
                    color="black", lw=1.0, ls="--", label="Unmet")
            ax.set_title(f"{rlabel} — {slabel}", fontsize=9)
            ax.set_ylabel("kW_th")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

    axes[-1][0].set_xlabel("Day of year")
    axes[-1][1].set_xlabel("Day of year")
    plt.tight_layout()
    p = os.path.join(save_dir, "compare_heat_dispatch.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")

    # -- Fig 4: KPI bar chart ------------------------------------------
    kpi_labels = [
        "RE utilised\n(%)",
        "H₂ stored\n(MWh_th)",
        "Curtailment\n(%)",
        "Batt cycles\n(norm)",
        "Elec util\n(%)",
        "H₂ SOC mean\n(%)",
    ]

    def _norm_kpi(milp_k, rb_k):
        """Normalise both to [0,1] relative to max."""
        mx = max(abs(milp_k), abs(rb_k), 1e-9)
        return milp_k / mx, rb_k / mx

    milp_kpis_raw = [
        milp_results["P_curtail"].sum() / (milp_results["_meta"]["P_pv"].sum()
                                           + milp_results["_meta"]["P_wind"].sum() + 1e-9) * 100,
        milp_results["H_El2Hs"].sum() / 1000,
        milp_results["P_curtail"].sum() / (milp_results["_meta"]["P_pv"].sum()
                                            + milp_results["_meta"]["P_wind"].sum() + 1e-9) * 100,
        float(np.sum(milp_results["Pb_c"])) / 1000,
        milp_results["Pel_in"].mean(),
        milp_results["SOC_Hs"].mean(),
    ]
    rb_kpis_raw = [
        (1 - rb_results["P_curtail"].sum() / (rb_results["_meta"]["P_pv"].sum()
                                              + rb_results["_meta"]["P_wind"].sum() + 1e-9)) * 100,
        rb_results["H_El2Hs"].sum() / 1000,
        rb_results["P_curtail"].sum() / (rb_results["_meta"]["P_pv"].sum()
                                          + rb_results["_meta"]["P_wind"].sum() + 1e-9) * 100,
        float(np.sum(rb_results["Pb_c"])) / 1000,
        rb_results["Pel_in"].mean(),
        rb_results["SOC_Hs"].mean(),
    ]

    x     = np.arange(len(kpi_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("KPI Comparison: MILP-MPC vs Rule-Based", fontsize=13)
    bars1 = ax.bar(x - width/2, milp_kpis_raw, width,
                   label="MILP-MPC",  color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width/2, rb_kpis_raw,   width,
                   label="Rule-Based", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(kpi_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Key Operational Metrics (absolute values)", fontsize=11)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    p = os.path.join(save_dir, "compare_kpi_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")

    print(f"\n  All comparison plots saved to {save_dir}/")


# =============================================================================
# Self-test (uses rule-based for both sides since MILP-MPC is slow)
# =============================================================================
if __name__ == "__main__":
    import scipy.io as sio
    from run_rule_based import run_rule_based

    print("compare.py — self-test (both sides = rule-based)")

    data     = sio.loadmat("Data/Longyearbyen_profiles.mat")
    profiles = {k: data[k].flatten() for k in ["PV", "WT", "H2", "ED"]}

    Batt_template = {
        "Pb_min": 0.0, "SOC_max": 90.0, "SOC_min": 10.0,
        "eta_c": 0.95, "eta_d": 0.95, "SOC_init": 50.0, "c_rate": 0.5,
    }
    Hs_template = {"SOC_min": 0.0, "SOC_max": 100.0, "SOC_init": 20.0}
    sizing      = [200.0, 400.0, 400.0, 150.0, 4000.0]

    res1 = run_rule_based(sizing, profiles, Batt_template, Hs_template,
                           verbose=False)
    # Simulate slightly different sizing as "MILP" proxy for test
    res1["_meta"]["method"] = "MILP-MPC"

    res2 = run_rule_based(sizing, profiles, Batt_template, Hs_template,
                           verbose=False)

    table = compare_controllers(res1, res2, sizing, profiles,
                                Batt_template, Hs_template)
    plot_comparison(res1, res2, profiles)
    print("\n  Self-test complete.")
