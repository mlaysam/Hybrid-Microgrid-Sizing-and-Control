"""
solver_islanded.py
==================
MILP-based MPC energy management for an isolated (off-grid) microgrid.

System: PV + Wind → Power Bus → Electrical Load
                              → Battery (charge / discharge)
                              → Electrolyzer → H2 Storage → Heating Demand
                                             → Direct Heating

No grid connection. Supply-demand balance is guaranteed by constraints;
the optimizer minimises curtailment and battery stress while maximising
H2 storage.

Decision variable layout  (17 * Np total, all per-step blocks of length Np)
-------------------------------------------------------------------------------
Continuous (16 blocks x Np):
  [ 0]  Pb_c        battery charge power           kW
  [ 1]  Pb_d        battery discharge power        kW
  [ 2]  SOC         battery state of charge        %
  [ 3]  lambda_pv   PV dispatch fraction           0-1
  [ 4]  lambda_wt   Wind dispatch fraction         0-1
  [ 5]  SOC_Hs      H₂ storage state of charge    %
  [ 6]  H_El2Hs     electrolyzer → H₂ storage     kW_th
  [ 7]  H_El2Hd     electrolyzer → direct heat    kW_th
  [ 8]  H_Hs2Hd     H₂ storage → heat demand      kW_th
  [ 9]  Pel_in      electrolyzer power input       kW
  [10]  P_RE_B      RE → battery                  kW
  [11]  P_RE_Ele    RE → electrolyzer              kW
  [12]  P_RE_L      RE → electrical load           kW   ← new
  [13]  P_B_Ele     battery → electrolyzer         kW
  [14]  P_B_L       battery → electrical load      kW   ← new
  [15]  P_curtail   RE curtailment                 kW   ← new

Binary (1 block x Np):
  [16]  delta_b     1 = charging mode, 0 = discharging mode

Global power balance (verified analytically):
    P_RE + Pb_d  =  Pb_c + Pel_in + P_load + P_curtail
----------------------------------------------------------------------------------
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# ----------------------------------------------------------------------------------
# Helper: block-row index  i * Np … (i+1) * Np
# ----------------------------------------------------------------------------------
def _blk(i, Np):
    return slice(i * Np, (i + 1) * Np)


def MILP_islanded(P_pv, P_wind, P_load, HeatingD,
                  Batt, dt, Np, Capacities, Hs, Cost):
    """
    Solve one MPC window for the islanded microgrid.

    Parameters
    ----------
    P_pv      : array (Np,)   PV available power profile    [kW]
    P_wind    : array (Np,)   Wind available power profile  [kW]
    P_load    : array (Np,)   Electrical load demand        [kW]
    HeatingD  : array (Np,)   Heating demand                [kW_th]
    Batt      : dict          Battery parameters
    dt        : float         Time-step length              [h]
    Np        : int           Prediction horizon            [steps]
    Capacities: dict          Installed capacities
    Hs        : dict          H₂ storage parameters
    Cost      : dict          Objective weights
                  "w_curtail"  penalty per kW curtailed          (default 1.0)
                  "w_batt"     penalty per kW battery throughput (default 0.01)
                  "w_H2"       reward per kW stored as H₂        (default 0.5)

    Returns
    -------
    x         : array (17*Np,)  solution vector
    status    : int             0 = optimal
    fun       : float           objective value
    mip_gap   : float           optimality gap
    n_vars    : int             total number of decision variables
    """

    # --- Parameters --------------------------------------------------------------
    Pb_max    = Batt["Pb_max"]
    Mb        = Pb_max                   # big-M for battery binary

    eta_c     = Batt["eta_c"]
    eta_d     = Batt["eta_d"]
    SOC_init  = Batt["SOC_init"]
    SOC_min   = Batt["SOC_min"]
    SOC_max   = Batt["SOC_max"]
    Qnom      = Batt["Qnom"]

    eta_El    = 0.6                      # electrolyzer efficiency (elec → heat)
    Pnom_El   = Capacities["Pnom_El"]

    Qnom_Hs   = Capacities["Hs"]
    SOC_Hs_min = Hs["SOC_min"]
    SOC_Hs_max = Hs["SOC_max"]
    SOC_Hs_init = Hs["SOC_init"]

    w_curtail = Cost.get("w_curtail", 1.0)
    w_batt    = Cost.get("w_batt",    0.01)
    w_H2      = Cost.get("w_H2",      0.5)

    # --- Variable dimensions ----------------------------------------------------
    N_CONT  = 16          # continuous variable blocks
    N_BIN   = 1           # binary variable blocks
    n_vars  = (N_CONT + N_BIN) * Np

    # Variable block indices  (shorthand: V[i] = slice(i*Np, (i+1)*Np))
    V = [_blk(i, Np) for i in range(N_CONT + N_BIN)]
    # V[0]  Pb_c       V[1]  Pb_d       V[2]  SOC
    # V[3]  lambda_pv  V[4]  lambda_wt  V[5]  SOC_Hs
    # V[6]  H_El2Hs    V[7]  H_El2Hd    V[8]  H_Hs2Hd
    # V[9]  Pel_in     V[10] P_RE_B     V[11] P_RE_Ele
    # V[12] P_RE_L     V[13] P_B_Ele    V[14] P_B_L
    # V[15] P_curtail  V[16] delta_b

    I  = np.eye(Np)
    Aug_pv  = np.diag(P_pv)
    Aug_pw  = np.diag(P_wind)

    # ==========================================================================
    # INEQUALITY CONSTRAINTS
    # ==========================================================================

    # 1. Pb_c(t) ≤ delta_b(t) * Mb   →   Pb_c − Mb·delta_b ≤ 0
    A1 = np.zeros((Np, n_vars))
    A1[:, V[0]]  =  I
    A1[:, V[16]] = -Mb * I
    b1 = np.zeros(Np)

    # 2. Pb_d(t) ≤ (1 − delta_b(t)) * Mb   →   Pb_d + Mb·delta_b ≤ Mb
    A2 = np.zeros((Np, n_vars))
    A2[:, V[1]]  =  I
    A2[:, V[16]] =  Mb * I
    b2 = Mb * np.ones(Np)

    Aineq   = np.vstack([A1, A2])
    bineq   = np.concatenate([b1, b2])
    bl_ineq = np.full(len(bineq), -np.inf)
    bu_ineq = bineq

    # ==========================================================================
    # EQUALITY CONSTRAINTS
    # ==========================================================================

    # Eq1: RE routing
    #   P_RE_B + P_RE_Ele + P_RE_L + P_curtail
    #       = Ppv(t)·lambda_pv(t) + Pwt(t)·lambda_wt(t)
    Aeq1 = np.zeros((Np, n_vars))
    Aeq1[:, V[10]] =  I
    Aeq1[:, V[11]] =  I
    Aeq1[:, V[12]] =  I
    Aeq1[:, V[15]] =  I
    Aeq1[:, V[3]]  = -Aug_pv
    Aeq1[:, V[4]]  = -Aug_pw
    beq1 = np.zeros(Np)

    # Eq2: Battery SOC dynamics
    #   SOC(k) = SOC(k−1) + (100·dt·η_c / Qnom)·Pb_c(k)
    #                      − (100·dt / (η_d·Qnom))·Pb_d(k)
    #   Rearranged: (100·dt·η_c/Qnom)·Pb_c − (100·dt/(η_d·Qnom))·Pb_d
    #               − SOC(k) + SOC(k−1) = 0
    #   For k=0:  SOC(k−1) = SOC_init  →  RHS = −SOC_init
    Aeq2 = np.zeros((Np, n_vars))
    Aeq2[:, V[0]] =  (100.0 / Qnom) * dt * eta_c  * I
    Aeq2[:, V[1]] = -(100.0 / Qnom) * (dt / eta_d) * I
    Aeq2[:, V[2]] = -I
    for k in range(1, Np):
        Aeq2[k, 2*Np + k - 1] = 1.0    # SOC(k−1) term
    beq2 = np.zeros(Np)
    beq2[0] = -SOC_init

    # Eq3: H₂ storage SOC dynamics
    #   SOC_Hs(k) = SOC_Hs(k−1)
    #               + (100·dt / Qnom_Hs)·H_El2Hs(k)
    #               − (100·dt / Qnom_Hs)·H_Hs2Hd(k)
    Aeq3 = np.zeros((Np, n_vars))
    Aeq3[:, V[6]] =  (100.0 / Qnom_Hs) * dt * I
    Aeq3[:, V[8]] = -(100.0 / Qnom_Hs) * dt * I
    Aeq3[:, V[5]] = -I
    for k in range(1, Np):
        Aeq3[k, 5*Np + k - 1] = 1.0    # SOC_Hs(k−1) term
    beq3 = np.zeros(Np)
    beq3[0] = -SOC_Hs_init

    # Eq4: Electrolyzer thermal balance
    #   H_El2Hs + H_El2Hd = η_El · Pel_in
    Aeq4 = np.zeros((Np, n_vars))
    Aeq4[:, V[6]] =  I
    Aeq4[:, V[7]] =  I
    Aeq4[:, V[9]] = -eta_El * I
    beq4 = np.zeros(Np)

    # Eq5: Heating demand balance
    #   H_El2Hd + H_Hs2Hd = HeatingD(t)
    Aeq5 = np.zeros((Np, n_vars))
    Aeq5[:, V[7]] = I
    Aeq5[:, V[8]] = I
    beq5 = HeatingD

    # Eq6: Electrical load balance  ← NEW
    #   P_RE_L + P_B_L = P_load(t)
    Aeq6 = np.zeros((Np, n_vars))
    Aeq6[:, V[12]] = I
    Aeq6[:, V[14]] = I
    beq6 = P_load

    # Eq7: Battery charge source routing
    #   Pb_c = P_RE_B
    #   (islanded: battery can only be charged from RE)
    Aeq7 = np.zeros((Np, n_vars))
    Aeq7[:, V[0]]  =  I
    Aeq7[:, V[10]] = -I
    beq7 = np.zeros(Np)

    # Eq8: Battery discharge destination routing
    #   Pb_d = P_B_Ele + P_B_L
    Aeq8 = np.zeros((Np, n_vars))
    Aeq8[:, V[1]]  =  I
    Aeq8[:, V[13]] = -I
    Aeq8[:, V[14]] = -I
    beq8 = np.zeros(Np)

    # Eq9: Electrolyzer input source routing
    #   Pel_in = P_RE_Ele + P_B_Ele
    Aeq9 = np.zeros((Np, n_vars))
    Aeq9[:, V[9]]  =  I
    Aeq9[:, V[11]] = -I
    Aeq9[:, V[13]] = -I
    beq9 = np.zeros(Np)

    Aeq = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4, Aeq5,
                     Aeq6, Aeq7, Aeq8, Aeq9])
    beq = np.concatenate([beq1, beq2, beq3, beq4, beq5,
                          beq6, beq7, beq8, beq9])
    bl_eq = beq
    bu_eq = beq

    # ==========================================================================
    # COMBINED CONSTRAINT MATRIX
    # ==========================================================================
    A   = np.vstack([Aineq, Aeq])
    bl  = np.concatenate([bl_ineq, bl_eq])
    bu  = np.concatenate([bu_ineq, bu_eq])
    constraints = LinearConstraint(A, lb=bl, ub=bu)

    # ==========================================================================
    # BOUNDS
    # ==========================================================================
    lb_vars = np.zeros(n_vars)
    ub_vars = np.full(n_vars, np.inf)

    # Battery SOC bounds
    lb_vars[V[2]] = SOC_min
    ub_vars[V[2]] = SOC_max

    # RE dispatch fractions
    ub_vars[V[3]] = 1.0    # lambda_pv
    ub_vars[V[4]] = 1.0    # lambda_wt

    # H₂ storage SOC bounds
    lb_vars[V[5]] = SOC_Hs_min
    ub_vars[V[5]] = SOC_Hs_max

    # Electrolyzer capacity
    ub_vars[V[9]] = Pnom_El

    # Binary: delta_b ∈ {0, 1}
    ub_vars[V[16]] = 1.0

    bounds = Bounds(lb_vars, ub_vars)

    # ==========================================================================
    # INTEGRALITY
    # ==========================================================================
    integrality = np.zeros(n_vars)
    integrality[V[16]] = 1.0    # delta_b only

    # ==========================================================================
    # OBJECTIVE FUNCTION
    # ==========================================================================
    # Minimise:
    #   w_curtail · P_curtail   (penalise wasted RE)
    #   w_batt · (Pb_c + Pb_d) (penalise battery cycling)
    # Maximise:
    #   w_H2 · H_El2Hs          (reward long-term H₂ storage)
    # --------------------------------------------------------------------------
    c = np.zeros(n_vars)
    c[V[15]] =  w_curtail       # curtailment penalty
    c[V[0]]  =  w_batt          # battery charge stress
    c[V[1]]  =  w_batt          # battery discharge stress
    c[V[6]]  = -w_H2            # reward H₂ storage

    # ==========================================================================
    # SOLVE
    # ==========================================================================
    res = milp(c=c, constraints=constraints,
               bounds=bounds, integrality=integrality)

    if res.status != 0:
        import warnings
        warnings.warn(
            f"MILP solver returned status {res.status}: {res.message}\n"
            f"  Check that installed capacities are sufficient to meet load.",
            RuntimeWarning
        )

    return res.x, res.status, res.fun, res.mip_gap, n_vars


# -----------------------------------------------------------------------------
# Variable extraction helper
# (use in main.py after calling MILP_islanded)
# -----------------------------------------------------------------------------
def extract_first_step(x, Np):
    """
    Extract the first step of each decision variable from solution vector x.
    Returns a dict matching the results structure in main.py.

    Parameters
    ----------
    x   : solution vector from MILP_islanded
    Np  : prediction horizon

    Returns
    -------
    dict with scalar values for each variable at t=0
    """
    return {
        "Pb_c":       x[0  * Np],
        "Pb_d":       x[1  * Np],
        "SOC":        x[2  * Np],
        "lambda_pv":  x[3  * Np],
        "lambda_wt":  x[4  * Np],
        "SOC_Hs":     x[5  * Np],
        "H_El2Hs":    x[6  * Np],
        "H_El2Hd":    x[7  * Np],
        "H_Hs2Hd":    x[8  * Np],
        "Pel_in":     x[9  * Np],
        "P_RE_B":     x[10 * Np],
        "P_RE_Ele":   x[11 * Np],
        "P_RE_L":     x[12 * Np],
        "P_B_Ele":    x[13 * Np],
        "P_B_L":      x[14 * Np],
        "P_curtail":  x[15 * Np],
        "delta_b":    x[16 * Np],
    }


# -----------------------------------------------------------------------------
# Quick feasibility test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import scipy.io as sio, os

    print("Running MILP_islanded feasibility test on Longyearbyen profiles...")

    mat_path = os.path.join("Data", "Longyearbyen_profiles.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Run data_generator.py first to create {mat_path}"
        )

    data = sio.loadmat(mat_path)
    PV_cf = data["PV"].flatten()
    WT_cf = data["WT"].flatten()
    H2    = data["H2"].flatten()
    ED    = data["ED"].flatten()

    # ── Placeholder capacities (will be optimised by PSO/GA later) 
    C_pv   = 300.0    # kW
    C_wt   = 500.0    # kW
    C_batt = 500.0    # kWh  (Qnom)
    C_el   = 200.0    # kW   (electrolyzer rated power)
    C_hs   = 5000.0   # kWh  (H₂ storage capacity)

    Np = 24
    dt = 1.0

    Batt = {
        "Pb_max":   C_batt * 0.5,   # C-rate 0.5C
        "Pb_min":   0.0,
        "SOC_max":  90.0,
        "SOC_min":  10.0,
        "eta_c":    0.95,
        "eta_d":    0.95,
        "SOC_init": 50.0,
        "Qnom":     C_batt,
    }
    Capacities = {
        "PV":     C_pv,
        "WT":     C_wt,
        "Batt":   C_batt,
        "Pnom_El": C_el,
        "Hs":     C_hs,
    }
    Hs = {"SOC_min": 0.0, "SOC_max": 100.0, "SOC_init": 20.0}
    Cost = {"w_curtail": 1.0, "w_batt": 0.01, "w_H2": 0.5}

    # Test on hour 0 window
    start = 0
    P_pv_win    = PV_cf[start:start+Np] * C_pv
    P_wind_win  = WT_cf[start:start+Np] * C_wt
    P_load_win  = ED[start:start+Np]
    HeatingD_win = H2[start:start+Np]

    x, status, fval, mip_gap, n_vars = MILP_islanded(
        P_pv_win, P_wind_win, P_load_win, HeatingD_win,
        Batt, dt, Np, Capacities, Hs, Cost
    )

    print(f"\n  Status   : {status}  (0 = optimal)")
    print(f"  Obj val  : {fval:.4f}")
    print(f"  MIP gap  : {mip_gap:.6f}")
    print(f"  n_vars   : {n_vars}")

    step = extract_first_step(x, Np)
    print(f"\n  First-step results:")
    for k, v in step.items():
        print(f"    {k:<12} = {v:8.3f}")

    # Verify load balance
    load_met = step["P_RE_L"] + step["P_B_L"]
    load_req = P_load_win[0]
    print(f"\n  Load balance check: met={load_met:.3f}  required={load_req:.3f}  "
          f"error={abs(load_met - load_req):.2e}")

    # Verify RE routing balance
    re_avail = P_pv_win[0] * step["lambda_pv"] + P_wind_win[0] * step["lambda_wt"]
    re_used  = step["P_RE_B"] + step["P_RE_Ele"] + step["P_RE_L"] + step["P_curtail"]
    print(f"  RE routing check : avail={re_avail:.3f}  used={re_used:.3f}  "
          f"error={abs(re_avail - re_used):.2e}")

    print("\n  All checks passed." if abs(load_met - load_req) < 1e-4 else
          "\n  WARNING: Load balance check failed.")
