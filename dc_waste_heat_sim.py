#!/usr/bin/env python3
"""
dc_waste_heat_sim.py  --  physics simulation of data-centre waste-heat reuse

Everything flagged ASSUMPTION is not public data -- replace with measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

HOURS, DT_S = 8760, 3600.0

# ---- constants (workbook: Heat Supply Waterfall constants block) --------------
CP_W, RHO_W = 4186.0, 1000.0
K_INS, K_SOIL, H_BURY, T_GROUND = 0.024, 1.5, 1.0, 9.0
R1_SHEET, R2_SHEET = 0.075, 0.125          # fixed DN150 geometry used in the sheet
INS_THICK, WALL, V_DESIGN = 0.05, 0.005, 1.5  # sized-pipe mode: insulation, steel wall, m/s
ETA_HP, COP_MAX, PINCH = 0.45, 15.0, 2.0   # Offtake Viability D3; practical COP cap; HX pinch (K)
CF_DEFAULT = 0.40                          # Offtake Viability D4
ORC_CALIB = [(43.0, 25.0, 0.019), (81.0, 25.0, 0.046)]  # IEEE lab ORC points
ORC_T_COLD_SHEET, ORC_APPROACH, ORC_T_MIN_STUDY = 15.0, 5.0, 55.0
FLOW_FLOOR = 0.25                          # ASSUMPTION: DH min flow = 25 % of design
# digester (workbook: ORC & AD Energy Balance)
AD = dict(t_set=37.0, t_feed=12.0, t_amb=10.0, mdot=5.0, cp=3800.0, d=20.0, h=15.0, u=0.6)
AD_SUPPLY, AD_DT, AD_BAND = 45.0, 10.0, (35.0, 40.0)   # coil water temp (ASSUMPTION), mesophilic band
LOAD_MEAN = 0.85                           # ASSUMPTION: mean IT load / design


def carnot_cop(t_sink, t_src):
    return (t_sink + 273.15) / (t_sink - t_src)


def cop_array(t_sink, t_hot, eta=ETA_HP):
    """Heat-pump COP; inf where the loop is already hot enough (direct heat exchange)."""
    t_hot = np.asarray(t_hot, float)
    cop = np.minimum(eta * (t_sink + 273.15) / np.maximum(t_sink - t_hot, 0.5), COP_MAX)
    return np.where(t_hot >= t_sink + PINCH, np.inf, cop)


def mdot_for(q_w, dt):
    return q_w / (CP_W * dt)


def pipe_u_prime(r1, r2):
    r_ins = math.log(r2 / r1) / (2 * math.pi * K_INS)
    r_soil = math.acosh(H_BURY / r2) / (2 * math.pi * K_SOIL)
    return 1.0 / (r_ins + r_soil)


@dataclass(frozen=True)
class Pipe:
    r1: float
    r2: float

    @property
    def u_prime(self):
        return pipe_u_prime(self.r1, self.r2)


def size_pipe(q_design_w, dt):
    area = mdot_for(q_design_w, dt) / (RHO_W * V_DESIGN)
    r1 = max(math.sqrt(area / math.pi), 0.0125) + WALL
    return Pipe(r1, r1 + INS_THICK)


def standing_loss_w(q_w, ts, dt, length_m, pipe, mdot_floor):
    """Supply+return heat loss (W). Each pipe: m Cp (T-Tg)(1-exp(-U'L/(m Cp)))."""
    m = np.maximum(mdot_for(q_w, dt), mdot_floor)
    x = pipe.u_prime * length_m / (m * CP_W)
    return m * CP_W * (-np.expm1(-x)) * np.maximum((ts - T_GROUND) + (ts - dt - T_GROUND), 0.0)


def pipe_velocity(pipe, mdot):
    """Actual mean water velocity (m/s); differs from V_DESIGN when the pipe hits its minimum size."""
    return mdot / (RHO_W * math.pi * (pipe.r1 - WALL) ** 2)


def supply_temp(t_sink, t_hot):
    """Water temperature in the supply pipe: heat-pump output, or loop water (minus HX pinch) when used directly."""
    t_hot = np.asarray(t_hot, float)
    return np.where(t_hot >= t_sink + PINCH, t_hot - PINCH, t_sink)


def orc_f():
    return float(np.mean([a / (1 - (c + 273.15) / (h + 273.15)) for h, c, a in ORC_CALIB]))


def orc_eta(t_hot, t_cold):
    return orc_f() * np.maximum(1 - (t_cold + 273.15) / (np.asarray(t_hot, float) + 273.15), 0.0)


def pump_power_w(q_design_w, dt, length_m, pipe, f=0.02, eta=0.7):
    """Darcy-Weisbach, supply + return, at the pipe's actual velocity."""
    di = 2 * (pipe.r1 - WALL)
    m = mdot_for(q_design_w, dt)
    v = pipe_velocity(pipe, m)
    dp = 2 * f * (length_m / di) * RHO_W * v ** 2 / 2
    return dp * m / RHO_W / eta


# ---- facilities (inputs mirror the workbook; climate/demand shape = ASSUMPTION) ----
@dataclass(frozen=True)
class Facility:
    name: str
    short: str
    t_src: float       # Offtake Viability B   design hot-side loop temp (C)
    t_sink: float      # Offtake Viability C   offtake supply temp (C)
    q_cap_mw: float    # Offtake Viability F (atNorth derived from households)
    capture: float     # Offtake Viability J
    dt_loop: float     # Waterfall S2 D
    dt_off: float      # Waterfall S3 D
    dist_km: float     # Waterfall S3 C
    demand_mwh: float  # annual offtake demand (see note in README block of main)
    kind: str          # 'dh' | 'building' | 'aqua'
    climate: tuple     # (annual mean C, annual amplitude K, diurnal amplitude K)  ASSUMPTION
    t_bal: float = 15.0
    dhw: float = 0.15
    it_mw: float | None = None   # disclosed IT capacity (MW), where public -- used only as a consistency check


FACILITIES = [
    Facility("Google - Hamina, Finland", "Hamina", 32, 70, 7.5, .75, 10, 30, .3, 40000 / .80, "dh", (5.0, 12.5, 3.0)),
    Facility("atNorth - DEN01, Ballerup", "atNorth", 40, 70, 8000 * 15 / 8760 / CF_DEFAULT, .75, 15, 30, .3,
             8000 * 15, "dh", (9.0, 8.5, 3.0), it_mw=22.5),
    Facility("NREL - ESIF, Golden CO", "NREL", 38, 30, 10, .97, 14, 10, .02, 10 * 8760 * CF_DEFAULT,
             "building", (10.5, 13.0, 7.0), 12.0, 0.05, it_mw=10.0),
    Facility("EcoDataCenter - Falun", "EcoDC", 32, 70, 80, .75, 12, 30, .2, 80 * 8760 * CF_DEFAULT, "dh",
             (4.5, 12.5, 4.0)),
    Facility("Green Mountain - Rjukan", "GreenMtn", 30, 14, 1.75, .85, 8, 5, .8, 1.75 * 8760 * .85, "aqua",
             (3.5, 11.0, 4.0)),
    Facility("Aquasar - ETH Zurich/IBM", "Aquasar", 65, 45, .009, .80, 5, 15, .02, .009 * 8760 * CF_DEFAULT,
             "building", (10.0, 10.0, 4.5), 15.0, 0.10),
]


def make_timeseries(fac, seed, load_mean=LOAD_MEAN):
    rng = np.random.default_rng(seed)
    burn = 2000                                                          # AR(1) filters start at 0: discard spin-up
    ar = lambda pole, sd: lfilter([1], [1, -pole], rng.normal(0, sd, HOURS + burn))[burn:]
    h = np.arange(HOURS)
    doy, hod = h / 24.0, h % 24
    tm, ay, ad = fac.climate
    slow = tm - ay * np.cos(2 * np.pi * (doy - 25) / 365)
    weather = ar(0.995, 0.30)                                            # ~3 K std, multi-day persistence
    t_amb = slow + ad * np.cos(2 * np.pi * (hod - 15) / 24) + weather
    load = np.clip(load_mean + 0.04 * np.cos(2 * np.pi * (hod - 14) / 24)
                   + ar(0.98, 0.008), 0.40, 1.0)
    if fac.kind == "aqua":
        shape = 0.7 + 0.3 * np.clip((15 - t_amb) / 30, 0, 1)
    else:
        t_design = tm - ay - 5
        shape = fac.dhw + (1 - fac.dhw) * np.clip((fac.t_bal - t_amb) / (fac.t_bal - t_design), 0, 1.3)
    demand = fac.demand_mwh * shape / shape.sum()                        # MW, sums to annual MWh
    return dict(t_amb=t_amb, t_slow=slow, load=load, demand_mw=demand)


def hot_side(fac, ts):
    """First Law + fixed-flow loop: heat captured and loop temperature track IT load."""
    f = ts["load"]
    t_hot = fac.t_src - fac.dt_loop * (1 - f)
    cop_d = float(cop_array(fac.t_sink, fac.t_src))
    p_design = fac.q_cap_mw * (1 - 1 / cop_d) / fac.capture              # MW heat from IT at design
    return dict(t_hot=t_hot, q_ex=fac.capture * p_design * f * 1e6, p_design_mw=p_design)


def run_heat_offtake(fac, ts, hs, pipe):
    cop = cop_array(fac.t_sink, hs["t_hot"])
    q_pot = hs["q_ex"] / (1 - 1 / cop)
    d = ts["demand_mw"] * 1e6
    qd = fac.q_cap_mw * 1e6
    floor = FLOW_FLOOR * mdot_for(qd, fac.dt_off)
    kw = dict(ts=supply_temp(fac.t_sink, hs["t_hot"]), dt=fac.dt_off, length_m=fac.dist_km * 1e3, pipe=pipe, mdot_floor=floor)
    q = np.minimum(q_pot, d)
    for _ in range(8):
        q = np.minimum(q_pot, d + standing_loss_w(q, **kw))
    loss = np.minimum(standing_loss_w(q, **kw), q)
    served = q - loss
    elec = q / cop
    waste = q - elec
    checks = dict(dc_overdraw_w=float((waste - hs["q_ex"]).max()),          # heat taken from DC beyond what was captured
                  delivery_closure_w=float(np.abs(served + loss - q).max()),  # served + lost must equal delivered
                  oversupply_w=float((served - d).max()))                     # never serve more than demanded
    return dict(cop=cop, q_pot=q_pot, q_hp=q, loss=loss, served=served, elec=elec, waste_used=waste * served / np.maximum(q, 1e-9),
                demand=d, checks=checks)


def dig_consts():
    a = AD
    area = math.pi * a["d"] * a["h"] + 2 * math.pi * (a["d"] / 2) ** 2
    ua = a["u"] * area
    q_req = a["mdot"] * a["cp"] * (a["t_set"] - a["t_feed"]) + ua * (a["t_set"] - a["t_amb"])
    return dict(ua=ua, c=RHO_W * math.pi / 4 * a["d"] ** 2 * a["h"] * a["cp"], q_req_sheet=q_req, kp=0.2 * q_req)


def digester_step(T, q_avail, t_feed, t_amb, k):
    """One hour of the lumped digester: C dT/dt = Q_in - m Cp (T - T_feed) - UA (T - T_amb); P-controller on Q_in."""
    a = AD
    q_cmd = a["mdot"] * a["cp"] * (a["t_set"] - t_feed) + k["ua"] * (a["t_set"] - t_amb) + k["kp"] * (a["t_set"] - T)
    q_i = np.clip(q_cmd, 0.0, q_avail)
    return T + DT_S / k["c"] * (q_i - a["mdot"] * a["cp"] * (T - t_feed) - k["ua"] * (T - t_amb)), q_i


def run_digesters(fac, ts, hs, rhos):
    """Sweep digester count n = rho*n0; each digester is a lumped CSTR with a proportional heat controller."""
    a = AD
    k = dig_consts()
    q_req_sheet = k["q_req_sheet"]
    cop = cop_array(AD_SUPPLY, hs["t_hot"])
    q_hp = hs["q_ex"] / (1 - 1 / cop)
    qd = fac.q_cap_mw * 1e6
    pipe = size_pipe(qd, AD_DT)
    loss = standing_loss_w(q_hp, supply_temp(AD_SUPPLY, hs["t_hot"]), AD_DT, fac.dist_km * 1e3, pipe, FLOW_FLOOR * mdot_for(qd, AD_DT))
    q_av = np.maximum(q_hp - loss, 0.0)
    n0 = q_av.mean() / q_req_sheet
    n = rhos * n0
    t_feed = np.maximum(ts["t_slow"] + 2.0, 5.0)
    # the sheet sizes digesters on a constant 12 C feed / 10 C ambient; real demand follows the weather
    dem_ratio = float(np.mean(a["mdot"] * a["cp"] * (a["t_set"] - t_feed) + k["ua"] * (a["t_set"] - ts["t_amb"])) / q_req_sheet)
    T = np.full(len(rhos), a["t_set"])
    tlog, qin = np.empty((HOURS, len(rhos))), np.empty((HOURS, len(rhos)))
    for h in range(HOURS):
        T, q_i = digester_step(T, q_av[h] / n, t_feed[h], ts["t_amb"][h], k)
        tlog[h], qin[h] = T, q_i
    cov = ((tlog >= AD_BAND[0]) & (tlog <= AD_BAND[1])).mean(axis=0)
    ok = np.where(cov >= 0.95)[0]
    i_rel = int(ok.max()) if len(ok) else -1            # -1: no tested size holds the band 95 % of hours
    heat = (qin * n).sum(axis=0) / 1e6
    waste = ((qin * n) * (1 - 1 / cop)[:, None]).sum(axis=0) / 1e6
    elec = ((qin * n) / cop[:, None]).sum(axis=0) / 1e6
    return dict(rhos=rhos, n0=n0, n=n, cov=cov, i_rel=i_rel, tlog=tlog, heat_mwh=heat, waste_mwh=waste,
                elec_mwh=elec, q_req_kw=q_req_sheet / 1e3, cop=cop, dem_ratio=dem_ratio)


def run_orc(fac, ts, hs):
    t_cold = ts["t_amb"] + ORC_APPROACH
    eta = orc_eta(hs["t_hot"], t_cold)
    p = hs["q_ex"] * eta
    return dict(eta=eta, p_el=p, hours_above_55=float((hs["t_hot"] >= ORC_T_MIN_STUDY).mean()))


def distance_curve(fac, lengths_km):
    qd = fac.q_cap_mw * 1e6
    pipe = size_pipe(qd, fac.dt_off)
    ts = float(supply_temp(fac.t_sink, fac.t_src))
    out = [float(standing_loss_w(qd, ts, fac.dt_off, L * 1e3, pipe, 0.0)) / qd for L in lengths_km]
    return np.array(out), pipe


def simulate_facility(fac, seed):
    ts = make_timeseries(fac, seed)
    hs = hot_side(fac, ts)
    pipe = size_pipe(fac.q_cap_mw * 1e6, fac.dt_off)
    ho = run_heat_offtake(fac, ts, hs, pipe)
    rhos = np.round(np.arange(0.30, 1.501, 0.01), 2)
    ad = run_digesters(fac, ts, hs, rhos)
    orc = run_orc(fac, ts, hs)
    mwh = lambda x: float(np.sum(x) / 1e6)
    e_ex = mwh(hs["q_ex"])
    i = ad["i_rel"]
    n_rel = float(ad["n"][i]) if i >= 0 else 0.0
    ad_waste = float(ad["waste_mwh"][i]) if i >= 0 else 0.0
    ad_elec = float(ad["elec_mwh"][i]) if i >= 0 else 0.0
    L = np.logspace(-3, 1.5, 400)                               # 1 m to 31.6 km
    phi, _ = distance_curve(fac, L)
    d5 = float(np.interp(0.05, phi, L)) if phi.max() > 0.05 else float("inf")
    qd = fac.q_cap_mw * 1e6
    s = dict(
        facility=fac.name, short=fac.short, mean_load=float(ts["load"].mean()), mean_t_amb_C=float(ts["t_amb"].mean()),
        captured_waste_heat_MWh=e_ex, mean_cop=float(np.nan if np.isinf(ho["cop"]).all() else
                                                      (ho["q_hp"][~np.isinf(ho["cop"])].sum() / ho["elec"][~np.isinf(ho["cop"])].sum())),
        offtake_demand_MWh=mwh(ho["demand"]), offtake_served_MWh=mwh(ho["served"]),
        demand_coverage_pct=100 * mwh(ho["served"]) / mwh(ho["demand"]),
        hours_demand_unmet_pct=100 * float((ho["served"] < 0.98 * ho["demand"]).mean()),
        hp_electricity_MWh=mwh(ho["elec"]), transit_loss_MWh=mwh(ho["loss"]),
        transit_loss_pct_of_hp_output=100 * mwh(ho["loss"]) / max(mwh(ho["q_hp"]), 1e-12),
        pipe_r1_m=pipe.r1, pipe_U_W_per_mK=pipe.u_prime,
        transit_time_s=fac.dist_km * 1e3 / pipe_velocity(pipe, mdot_for(qd, fac.dt_off)),
        pipe_velocity_m_s=pipe_velocity(pipe, mdot_for(qd, fac.dt_off)),
        heat_use_share_pct=100 * mwh(ho["waste_used"]) / e_ex,
        ad_digesters_spreadsheet_style_n0=ad["n0"], ad_digesters_reliable_n=n_rel,
        ad_inband_pct_at_n0=100 * float(ad["cov"][np.argmin(abs(ad["rhos"] - 1.0))]),
        ad_weather_demand_vs_sheet=ad["dem_ratio"],
        ad_use_share_pct=100 * ad_waste / e_ex, ad_hp_electricity_MWh=ad_elec,
        orc_electricity_MWh=mwh(orc["p_el"]), orc_mean_eta_pct=100 * float(orc["p_el"].sum() / hs["q_ex"].sum()),
        orc_use_share_pct=100 * mwh(orc["p_el"]) / e_ex, hours_loop_above_55C_pct=100 * orc["hours_above_55"],
        distance_for_5pct_loss_km=d5, pump_kW_per_km=pump_power_w(qd, fac.dt_off, 1e3, pipe) / 1e3,
        implied_it_power_MW=hs["p_design_mw"], disclosed_it_power_MW=(fac.it_mw if fac.it_mw is not None else float("nan")),
        max_dc_overdraw_W=ho["checks"]["dc_overdraw_w"], max_delivery_closure_W=ho["checks"]["delivery_closure_w"],
        max_oversupply_W=ho["checks"]["oversupply_w"],
    )
    return s, dict(ts=ts, hs=hs, ho=ho, ad=ad, orc=orc)


# ---- bench-scale (Raspberry-Pi) model: same lumped-capacitance physics --------------
BENCH = dict(m_water=2.0, c_loop=700.0, ua_coil_water=6.0, mu_ratio=100.0, ua_loss=0.35, eta_capture=0.8,
             p_idle=2.7, p_mid=5.0, p_full=7.5, t_amb=21.0)   # ALL ASSUMPTIONS -- replace with fitted values


def simulate_bench(power_w, dt=2.0, fluid="water", p=BENCH):
    ua_coil = p["ua_coil_water"] * (1.0 if fluid == "water" else p["mu_ratio"] ** -0.25)  # h ~ mu^-1/4 (natural convection)
    c_b = p["m_water"] * CP_W + 150.0
    tl = tb = p["t_amb"]
    out = np.empty((len(power_w), 2))
    for k, pw in enumerate(power_w):
        q = ua_coil * (tl - tb)
        tl += dt / p["c_loop"] * (pw * p["eta_capture"] - q)
        tb += dt / c_b * (q - p["ua_loss"] * (tb - p["t_amb"]))
        out[k] = tl, tb
    return out, c_b, ua_coil


def fit_cooling_ua(t_s, temp, t_amb, c_b):
    slope, _ = np.polyfit(t_s, np.log(temp - t_amb), 1)       # log-linear Newton cooling fit
    return -slope * c_b


def bench_tests(seed):
    rng = np.random.default_rng(seed)
    dt, n_heat, n_cool = 2.0, 3 * 3600 // 2, 8 * 3600 // 2
    res = {}
    for fluid in ("water", "xanthan"):
        pw = np.r_[np.full(n_heat, BENCH["p_full"]), np.zeros(n_cool)]
        y, c_b, ua_coil = simulate_bench(pw, dt, fluid)
        noisy = y[:, 1] + rng.normal(0, 0.05, len(y))          # 0.05 K sensor noise
        t = np.arange(n_cool) * dt
        win = (t[: n_cool // 2], noisy[n_heat: n_heat + n_cool // 2], BENCH["t_amb"])
        # the loop water + waterblock cool with the bath, so the regression must use bath + loop capacitance
        res[fluid] = dict(y=y, pw=pw, ua_coil=ua_coil,
                          ua_fit=fit_cooling_ua(*win, c_b + BENCH["c_loop"]),
                          ua_fit_bath_only=fit_cooling_ua(*win, c_b))
    levels = np.array([BENCH["p_idle"], BENCH["p_mid"], BENCH["p_full"]])
    blocks = rng.integers(0, 3, 12 * 3600 // (20 * 60))
    pw_var = np.repeat(levels[blocks], 20 * 60 // 2)
    pw_const = np.full_like(pw_var, pw_var.mean())
    yv, _, _ = simulate_bench(pw_var, dt, "xanthan")
    yc, _, _ = simulate_bench(pw_const, dt, "xanthan")
    dev = (yv[:, 1] - yc[:, 1])[3600 // 2:]                     # swing about the constant-power run (drops warm-up trend)
    unbuf = BENCH["eta_capture"] * float(pw_var.std()) / BENCH["ua_loss"]   # swing if the bath had no thermal mass
    res["testC"] = dict(var=yv[:, 1], const=yc[:, 1], pw=pw_var, mean_p=float(pw_var.mean()),
                        std_dev=float(dev.std()), max_dev=float(np.abs(dev).max()), unbuffered_std=unbuf,
                        buffering_pct=100 * (1 - float(dev.std()) / unbuf))
    res["dt"] = dt
    return res


# ---- validation against the workbook (spreadsheet semantics) ------------------------
# Cell map of sheet "3 Steady-State Model" (first facility row of each table; the six
# facilities follow in order). The workbook builder uses the SAME map to place the cells.
WB_SHEET = "3 Steady-State Model"
WB = dict(viab=15, flow=24, uprime=(34, 3), pipe=38, f_orc=(49, 6), orc=53, qreq=(65, 3), ad=68)


def check_plan():
    """List of (label, python value, (row, col) on WB_SHEET): what the code predicts and where the workbook holds it."""
    plan = []
    u = pipe_u_prime(R1_SHEET, R2_SHEET)
    plan.append(("U' buried-pipe loss coefficient", u, WB["uprime"]))
    plan.append(("f_ORC calibrated Carnot fraction", orc_f(), WB["f_orc"]))
    qreq = AD["mdot"] * AD["cp"] * (AD["t_set"] - AD["t_feed"]) / 1e3 + \
        AD["u"] * (math.pi * AD["d"] * AD["h"] + 2 * math.pi * (AD["d"] / 2) ** 2) * (AD["t_set"] - AD["t_amb"]) / 1e3
    for i, fac in enumerate(FACILITIES):
        q = fac.q_cap_mw
        if fac.t_sink > fac.t_src:
            cc = carnot_cop(fac.t_sink, fac.t_src)
            plan.append((f"{fac.short}: Carnot COP", cc, (WB["viab"] + i, 4)))
            plan.append((f"{fac.short}: actual COP", cc * ETA_HP, (WB["viab"] + i, 5)))
        plan.append((f"{fac.short}: loop mass flow kg/s", mdot_for(q * 1e6, fac.dt_loop), (WB["flow"] + i, 5)))
        m_dh = mdot_for(q * 1e6, fac.dt_off)
        f_ret = math.exp(-u * fac.dist_km * 1e3 / (m_dh * CP_W))
        plan.append((f"{fac.short}: distribution flow kg/s", m_dh, (WB["pipe"] + i, 5)))
        plan.append((f"{fac.short}: fraction retained after L", f_ret, (WB["pipe"] + i, 6)))
        plan.append((f"{fac.short}: heat after transit MW", q * f_ret, (WB["pipe"] + i, 8)))
        eta = orc_f() * (1 - (ORC_T_COLD_SHEET + 273.15) / (fac.t_src + 273.15))
        plan.append((f"{fac.short}: ORC efficiency", eta, (WB["orc"] + i, 4)))
        plan.append((f"{fac.short}: ORC electric MW (sheet basis)", q * f_ret * eta, (WB["orc"] + i, 6)))
        plan.append((f"{fac.short}: digesters suppliable", q * f_ret / (qreq / 1e3), (WB["ad"] + i, 4)))
    plan.append(("Digester Q_required kW", qreq, WB["qreq"]))
    return plan


def validate(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True)[WB_SHEET]
    checks = []
    for label, mine, (r, c) in check_plan():
        ref = ws.cell(r, c).value
        ok = isinstance(ref, (int, float)) and bool(np.isclose(mine, ref, rtol=1e-6, atol=1e-9))
        checks.append(dict(check=label, code=float(mine), workbook=float(ref) if isinstance(ref, (int, float)) else None, ok=ok))
    return checks


def selftest(summaries, bench):
    """Physics self-checks that need no workbook."""
    checks = []

    def chk(label, mine, ref, rtol):
        checks.append(dict(check=label, code=float(mine), reference=float(ref), ok=bool(np.isclose(mine, ref, rtol=rtol, atol=1.0 if ref == 0 else 1e-6))))

    k, a = dig_consts(), AD
    # (1) supply == demand, constant weather -> digester holds its setpoint
    T = a["t_set"]
    for _ in range(24 * 60):
        T, _q = digester_step(T, k["q_req_sheet"], a["t_feed"], a["t_amb"], k)
    chk("digester holds 37 C when supply = Q_required", T, a["t_set"], 1e-6)
    # (2) supply = 90 % of demand -> settles at the analytic steady state T* = (Q + m Cp Tf + UA Ta)/(m Cp + UA)
    q = 0.9 * k["q_req_sheet"]
    T = a["t_set"]
    for _ in range(24 * 365):
        T, _q = digester_step(T, q, a["t_feed"], a["t_amb"], k)
    mcp = a["mdot"] * a["cp"]
    chk("digester settles at analytic T* under 90 % supply", T, (q + mcp * a["t_feed"] + k["ua"] * a["t_amb"]) / (mcp + k["ua"]), 1e-4)
    # (3) energy accounting, every hour, every facility (tolerance 1 W on MW-scale flows)
    chk("heat drawn from DC never exceeds heat captured (max excess, W)", max(max(s["max_dc_overdraw_W"], 0.0) for s in summaries), 0.0, 0.0)
    chk("served + transit loss = delivered (max error, W)", max(s["max_delivery_closure_W"] for s in summaries), 0.0, 0.0)
    chk("served never exceeds demand (max excess, W)", max(max(s["max_oversupply_W"], 0.0) for s in summaries), 0.0, 0.0)
    # (4) the cooling-curve regression recovers the true UA when loop mass is included
    for fl in ("water", "xanthan"):
        chk(f"bench: fitted UA_loss recovers truth ({fl})", bench[fl]["ua_fit"], BENCH["ua_loss"], 0.03)
    return checks


def hamina_consistency(n=20, seed0=5000):
    """Hamina publicly reports its heat pump covers ~80 % of local DH demand (40,000 of ~50,000 MWh).
    The model was never tuned to that; compare its predicted coverage across weather seeds."""
    f = FACILITIES[0]
    cov = [simulate_facility(f, seed0 + s)[0]["demand_coverage_pct"] for s in range(n)]
    return float(np.mean(cov)), float(np.std(cov))


# ---- plotting ----------------------------------------------------------------------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CTX = "#9a9994"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_facecolor(SURF)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def make_figures(outdir, summaries, sims, bench):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "text.color": INK})
    paths = []

    def save(fig, name):
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=160, facecolor=SURF, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # Fig 1 -- Hamina: supply vs demand
    k = [f.short for f in FACILITIES].index("Hamina")
    s = sims[k]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    day = lambda x: np.asarray(x).reshape(365, 24).mean(1) / 1e6
    for ax in (a1, a2):
        _style(ax)
    d = np.arange(365)
    a1.plot(d, day(s["ho"]["demand"]), color=CTX, lw=1.6, label="Offtake demand")
    a1.plot(d, day(s["ho"]["q_pot"]), color=SLOT[0], lw=1.6, label="Heat available from data centre")
    a1.plot(d, day(s["ho"]["served"]), color=SLOT[1], lw=1.9, label="Heat actually delivered")
    a1.set_xlabel("Day of year"); a1.set_ylabel("Daily mean thermal power (MW)")
    a1.set_title("Winter demand outruns supply; summer supply outruns demand", loc="left", fontsize=10.5, color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    w = slice(10 * 24, 17 * 24)
    hh = np.arange(7 * 24) / 24
    for key, col, lw in (("demand", CTX, 1.6), ("q_pot", SLOT[0], 1.6), ("served", SLOT[1], 1.9)):
        a2.plot(hh, np.asarray(s["ho"][key])[w] / 1e6, color=col, lw=lw)
    a2.set_xlabel("Days from 11 Jan"); a2.set_ylabel("MW")
    a2.set_title("One winter week, hourly", loc="left", fontsize=10.5, color=INK)
    fig.suptitle("Google Hamina - heat-pump offtake to district heating (simulated)", x=0.01, y=1.04, ha="left", fontsize=12, color=INK)
    save(fig, "fig1_hamina_year.png")

    # Fig 2 -- pathways
    fig, ax = plt.subplots(figsize=(9.5, 4.2)); _style(ax)
    x = np.arange(len(FACILITIES)); w_ = 0.26
    series = [("Heat offtake (as configured)", "heat_use_share_pct"), ("Anaerobic digestion (reliable count)", "ad_use_share_pct"),
              ("ORC electricity", "orc_use_share_pct")]
    for j, (lab, key) in enumerate(series):
        v = [sm[key] for sm in summaries]
        b = ax.bar(x + (j - 1) * (w_ + 0.02), v, w_, color=SLOT[j], label=lab)
        for xi, vi in zip(x + (j - 1) * (w_ + 0.02), v):
            ax.text(xi, vi + 1.0, f"{vi:.0f}" if vi >= 10 else f"{vi:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f.short for f in FACILITIES]); ax.set_ylabel("Share of captured waste heat put to use (%)")
    rng_ = lambda k: (min(sm[k] for sm in summaries), max(sm[k] for sm in summaries))
    (a0, a1_), (h0, h1), (o0, o1) = rng_("ad_use_share_pct"), rng_("heat_use_share_pct"), rng_("orc_use_share_pct")
    ax.set_title(f"Digesters put {a0:.0f}-{a1_:.0f}% of captured heat to use; hot-water offtakes {h0:.0f}-{h1:.0f}%; ORC {o0:.0f}-{o1:.0f}%",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.01, -0.06, "Digester bars assume as many digesters as the heat can reliably support (counts in summary.csv); "
             "hot-water bars are limited by the offtake's own demand.", fontsize=8, color=INK2)
    save(fig, "fig2_pathway_use_share.png")

    # Fig 3 -- digester sizing
    fig, ax = plt.subplots(figsize=(9.5, 4.2)); _style(ax)
    for i, f in enumerate(FACILITIES):
        ad = sims[i]["ad"]
        ax.plot(ad["rhos"], 100 * ad["cov"], color=SLOT[i], lw=1.8, label=f.short)
    rel = [float(sims[i]["ad"]["rhos"][sims[i]["ad"]["i_rel"]]) for i in range(len(FACILITIES)) if sims[i]["ad"]["i_rel"] >= 0]
    if rel:
        r_lo, r_hi = min(rel), max(rel)
        ax.axvspan(r_lo, r_hi, color=GRID, alpha=0.6, lw=0) if r_hi > r_lo else ax.axvline(r_lo, color=INK2, lw=1, ls=":")
        ax.text(r_lo - 0.01, 30, f"reliable limit for all six:\nn = {r_lo:.2f}-{r_hi:.2f} n0", fontsize=8.5, color=INK2, ha="right")
    ax.axhline(95, color=MUTED, lw=1, ls="--"); ax.text(1.34, 90, "95 % target", fontsize=8, color=INK2, ha="right")
    ax.set_xlim(0.6, 1.36)
    ax.set_xlabel("Digesters supplied, relative to the sheet's mean-heat balance (n / n0)")
    ax.set_ylabel("Hours inside 35-40 C band (%)")
    ax.set_title(f"Digesters hold temperature until {100*min(rel):.0f}-{100*max(rel):.0f}% of the sheet's count, then fall off a cliff",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "fig3_digester_sizing.png")

    # Fig 4 -- distance
    fig, ax = plt.subplots(figsize=(9.5, 4.4)); _style(ax)
    L = np.logspace(-2, 1.5, 200)
    for i, f in enumerate(FACILITIES):
        phi, _ = distance_curve(f, L)
        size = f"{f.q_cap_mw * 1e3:.0f} kW" if f.q_cap_mw < 0.1 else f"{f.q_cap_mw:.3g} MW"
        ax.plot(L, 100 * phi, color=SLOT[i], lw=1.8, label=f"{f.short} ({size})")
        pe, _ = distance_curve(f, [f.dist_km])
        ax.plot(f.dist_km, 100 * pe[0], "o", ms=7, color=SLOT[i], mec=SURF, mew=2)
    ax.axhline(5, color=MUTED, lw=1, ls="--"); ax.text(30, 5.6, "5 % loss", fontsize=8, color=INK2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.003, 100)
    ax.set_xlabel("Pipe distance to offtake (km, log)"); ax.set_ylabel("Heat lost in supply + return pipes (% of design output)")
    ax.set_title("Thermal loss per km falls as a facility gets bigger (dots = actual siting)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "fig4_distance_loss.png")

    # Fig 5 -- bench
    fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax in (a1, a2, a3):
        _style(ax)
    t_h = np.arange(len(bench["water"]["y"])) * bench["dt"] / 3600
    lab_w = f"Test A - water (fitted UA {bench['water']['ua_fit']:.2f} W/K)"
    lab_x = f"Test B - xanthan (fitted UA {bench['xanthan']['ua_fit']:.2f} W/K)"
    a1.plot(t_h, bench["water"]["y"][:, 1], color=SLOT[0], lw=1.9, label=lab_w)
    a1.plot(t_h, bench["xanthan"]["y"][:, 1], color=SLOT[1], lw=1.9, label=lab_x)
    a1.axvline(3, color=MUTED, lw=1, ls="--"); a1.text(3.1, BENCH["t_amb"] + 0.5, "load off", fontsize=8, color=INK2)
    a1.set_xlabel("Hours"); a1.set_ylabel("Bath temperature (C)")
    a1.set_title("Bath curves are nearly identical", loc="left", fontsize=10.5, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for fl, col in (("water", SLOT[0]), ("xanthan", SLOT[1])):
        y = bench[fl]["y"]
        a3.plot(t_h, y[:, 0] - y[:, 1], color=col, lw=1.9, label=fl)
    a3.set_xlabel("Hours"); a3.set_ylabel("Loop minus bath temperature (K)")
    a3.set_title("Viscosity shows up in the coil gap", loc="left", fontsize=10.5, color=INK)
    a3.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    c = bench["testC"]; tc = np.arange(len(c["var"])) * bench["dt"] / 3600
    a2.plot(tc, c["const"], color=CTX, lw=1.7, label="Same average power, constant")
    a2.plot(tc, c["var"], color=SLOT[1], lw=1.7, label="Randomised duty cycle (Test C)")
    a2.set_xlabel("Hours"); a2.set_ylabel("Bath temperature (C)")
    a2.set_title(f"Thermal mass absorbs {c['buffering_pct']:.0f} % of the load swing", loc="left", fontsize=10.5, color=INK)
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Pi-rig prediction - illustrative parameters, replace with your fitted values", x=0.01, y=1.04, ha="left", fontsize=11.5, color=INK)
    save(fig, "fig5_bench_prediction.png")
    return paths


# ---- driver -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/user-data/outputs/sim_results")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workbook", default="/mnt/user-data/outputs/dc_waste_heat_offtake_model.xlsx")
    ap.add_argument("--hourly", action="store_true", help="also write hourly CSV per facility")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    report = None
    if os.path.exists(a.workbook):
        report = validate(a.workbook)
        n_ok = sum(c["ok"] for c in report)
        print(f"VALIDATION vs workbook: {n_ok}/{len(report)} checks match")
        for c in report:
            if not c["ok"]:
                print("  MISMATCH", c)
        with open(os.path.join(a.out, "validation.json"), "w") as fh:
            json.dump(report, fh, indent=1)

    summaries, sims = [], []
    for i, fac in enumerate(FACILITIES):
        s, sim = simulate_facility(fac, a.seed * 100 + i)
        summaries.append(s); sims.append(sim)
        if a.hourly:
            with open(os.path.join(a.out, f"hourly_{fac.short}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["hour", "t_amb_C", "load", "t_hot_C", "demand_MW", "hp_potential_MW", "served_MW", "orc_MWe"])
                for h in range(HOURS):
                    w.writerow([h, sim["ts"]["t_amb"][h], sim["ts"]["load"][h], sim["hs"]["t_hot"][h], sim["ho"]["demand"][h] / 1e6,
                                sim["ho"]["q_pot"][h] / 1e6, sim["ho"]["served"][h] / 1e6, sim["orc"]["p_el"][h] / 1e6])
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)

    bench = bench_tests(a.seed)
    figs = make_figures(a.out, summaries, sims, bench)

    tests = selftest(summaries, bench)
    print("SELF-TESTS: %d/%d pass" % (sum(t["ok"] for t in tests), len(tests)))
    for t in tests:
        print("  [%s] %s  (code %.6g vs ref %.6g)" % ("ok" if t["ok"] else "FAIL", t["check"], t["code"], t["reference"]))
    hm, hs_ = hamina_consistency()
    print("HAMINA: model predicts heat pump covers %.1f%% +/- %.1f%% of local DH demand across 20 weather years "
          "(publicly reported: ~80%%)" % (hm, hs_))
    with open(os.path.join(a.out, "selftests.json"), "w") as fh:
        json.dump(dict(selftests=tests, hamina_coverage_pct=dict(mean=hm, std=hs_, reported=80.0)), fh, indent=1)

    for s in summaries:
        if not math.isnan(s["disclosed_it_power_MW"]):
            r_ = s["implied_it_power_MW"] / s["disclosed_it_power_MW"]
            flag = "OK" if r_ <= 1.05 else "INCONSISTENT"
            print("IT POWER CHECK %-8s implied %.1f MW vs disclosed %.1f MW (x%.2f) -> %s" % (
                s["short"], s["implied_it_power_MW"], s["disclosed_it_power_MW"], r_, flag))
    worst = max(summaries, key=lambda s: s["transit_time_s"])
    print("PIPE DYNAMICS: pipes are treated as quasi-steady (no transport delay). Longest transit %.0f s (%s) = %.0f%% of the "
          "1 h step: fine for annual totals, not for minute-scale transients." % (
              worst["transit_time_s"], worst["short"], 100 * worst["transit_time_s"] / DT_S))
    print("\nANNUAL RESULTS (hourly simulation, seed %d)" % a.seed)
    km = lambda v: ">31" if math.isinf(v) else ("%.3f" % v if v < 1 else "%.1f" % v)
    print("facility   demand%  loss%  heatUse%  AD n(rel)  AD n0   ADuse%  ORCuse%  km@5%loss")
    for s in summaries:
        print("%-9s  %7.1f  %5.2f  %8.1f  %9.3f  %6.3f  %6.1f  %7.2f  %9s" % (
            s["short"], s["demand_coverage_pct"], s["transit_loss_pct_of_hp_output"], s["heat_use_share_pct"],
            s["ad_digesters_reliable_n"], s["ad_digesters_spreadsheet_style_n0"], s["ad_use_share_pct"],
            s["orc_use_share_pct"], km(s["distance_for_5pct_loss_km"])))
    c = bench["testC"]
    print("\nBENCH (illustrative parameters): true UA_loss %.3f W/K | fitted: water %.3f, xanthan %.3f "
          "(ignoring loop mass: %.3f / %.3f)" % (BENCH["ua_loss"], bench["water"]["ua_fit"], bench["xanthan"]["ua_fit"],
                                               bench["water"]["ua_fit_bath_only"], bench["xanthan"]["ua_fit_bath_only"]))
    print("       Test C: swing about constant-power run std %.2f K (max %.2f K) vs %.2f K with no thermal mass -> %.0f%% buffered"
          % (c["std_dev"], c["max_dev"], c["unbuffered_std"], c["buffering_pct"]))
    print("files:", a.out, "|", ", ".join(os.path.basename(p) for p in figs))
    return summaries, report, bench


if __name__ == "__main__":
    main()
