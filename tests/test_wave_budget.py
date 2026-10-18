"""
Manufactured-solution tests for the wave-coherent budget (phase budget + time budget).
    python test_wave_budget.py
    pytest -q test_wave_budget.py
"""
import functools

import numpy as np
import sympy as sp
import xarray as xr

from padeopsIO.gridslice import GridDataset
from padeopsIO.budget_addons import Wave_x, Wave_y, Wave_z
from padeopsIO.utils import fluids_utils as fluids

X, Y, Z, T = sp.symbols("x y z t", real=True)
XYZ = (X, Y, Z)
DIMS = ("phase", "x", "y", "z")
PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
WAVE = [Wave_x, Wave_y, Wave_z]
TERMS = ("storage", "adv", "prod", "wavewave", "rs_phase", "rs_time", "prss", "sgs", "adm")
TOL_EXACT, TOL_FD = 1e-10, 0.15


# =============================================================================
# Manufactured fields
# =============================================================================
def manufactured(U0=1.0, D=0.3, sig=0.5, a=0.08, b=0.3, c=0.03, lam=2.0,
                 gam=0.3, f=0.5, r0=0.02, p0=0.05, t0=0.005):
    """Analytic phase-averaged fields <f>(x, y, z, t) and exact wave-budget terms."""
    th = 2 * sp.pi / lam * X - 2 * sp.pi * f * T          # phase increases with t
    env = sp.exp(-(Y**2 + Z**2) / sig**2)
    gz, hy = sp.exp(-Z**2 / sig**2), sp.exp(-Y**2 / sig**2)

    Ubar = [U0 - D * env, sp.Integer(0), sp.Integer(0)]                    # div-free wake
    psi = a * gz * sp.exp(-gam * X) * (sp.sin(th) + b * sp.sin(2 * th))     # (x, z, t)
    chi = c * hy * sp.exp(-gam * X) * sp.cos(th)                             # (x, y, t)
    uw = [hy * sp.diff(psi, Z) + gz * sp.diff(chi, Y),                       # div-free wave
          -gz * sp.diff(chi, X),
          -hy * sp.diff(psi, X)]
    u = [Ubar[n] + uw[n] for n in range(3)]

    # <u_i'u_j'>_phi (about the phase mean) and SGS stresses, symmetric
    R = [[r0*env*(1 + 0.5*sp.cos(th)), 0.1*r0*env*Y*sp.sin(th), -0.3*r0*env*Z*(1 + 0.3*sp.sin(th))],
         [None, 0.7*r0*env*(1 + 0.3*sp.sin(th)), 0.05*r0*env*sp.cos(th)],
         [None, None, 0.5*r0*env*(1 + 0.4*sp.cos(2*th))]]
    tau = [[t0*env*sp.cos(th), 0.2*t0*env*Y, -t0*env*Z*(1 + sp.sin(th))],
           [None, 0.5*t0*env, 0.1*t0*env*sp.sin(th)],
           [None, None, 0.3*t0*env*sp.cos(th)]]
    for M in (R, tau):
        for r in range(3):
            for s in range(r):
                M[r][s] = M[s][r]
    p = p0 * env * (1 + sp.cos(th)) - 0.01 * X

    # PadeOps budget-1 terms (RHS convention); turbine forcing closes the equation
    nonlin = [-sum(u[j] * sp.diff(u[i], XYZ[j]) + sp.diff(R[i][j], XYZ[j]) for j in range(3))
              for i in range(3)]
    prss = [-sp.diff(p, XYZ[i]) for i in range(3)]
    sgs = [-sum(sp.diff(tau[i][j], XYZ[j]) for j in range(3)) for i in range(3)]
    adm = [sp.diff(u[i], T) - nonlin[i] - prss[i] - sgs[i] for i in range(3)]

    fields = {"ubar": u[0], "vbar": u[1], "wbar": u[2], "pbar": p,
              "DuDt": nonlin[0], "DvDt": nonlin[1], "DwDt": nonlin[2],
              "dpdx": prss[0], "dpdy": prss[1], "dpdz": prss[2],
              "xSGS": sgs[0], "ySGS": sgs[1], "zSGS": sgs[2],
              "xAD": adm[0], "yAD": adm[1], "zAD": adm[2]}
    for (r, s), k in zip(PAIRS, fluids.rs_keys_list):
        fields[k] = R[r][s]
    for (r, s), k in zip(PAIRS, fluids.tau_keys_list):
        fields[k] = tau[r][s]

    exact = {i: {"storage": -sp.diff(uw[i], T),
                 "adv": -sum(Ubar[j] * sp.diff(uw[i], XYZ[j]) for j in range(3)),
                 "prod": -sum(uw[j] * sp.diff(Ubar[i], XYZ[j]) for j in range(3)),
                 "wavewave": -sum(uw[j] * sp.diff(uw[i], XYZ[j]) for j in range(3)),
                 "rs_phase": -sum(sp.diff(R[i][j], XYZ[j]) for j in range(3)),
                 "prss": prss[i], "sgs": sgs[i], "adm": adm[i], "nonlin": nonlin[i]}
             for i in range(3)}
    return fields, exact, f, uw


def build_case(n=(48, 24, 24), nphase=10, xlim=(0.0, 4.0), ylim=(-1.5, 1.5),
               zlim=(-1.5, 1.5), directions=(0, 1, 2), **params):
    """Returns (ds_phase [phase,x,y,z], ds_time [x,y,z], exact {i: Dataset}, period)."""
    fields, exact, f, uw = manufactured(**params)
    period = 1.0 / f
    x, y, z = (np.linspace(*lim, m) for lim, m in zip((xlim, ylim, zlim), n))
    phase = np.arange(nphase) / nphase
    tg, xg, yg, zg = np.meshgrid(phase * period, x, y, z, indexing="ij")

    def ev(expr):
        out = sp.lambdify((X, Y, Z, T), expr, "numpy")(xg, yg, zg, tg)
        return np.broadcast_to(np.asarray(out, dtype=float), xg.shape).copy()

    ds = GridDataset(x=x, y=y, z=z).assign_coords(phase=phase)
    for k, e in fields.items():
        ds[k] = (DIMS, ev(e))
    ds = ds.transpose(*DIMS)

    # Synthetic time budget. The phase mean is exact here (all harmonics < nphase).
    # Like PadeOps, the time-averaged uu includes the wave stress mean(u~ u~).
    ds_time = ds.mean("phase")
    UW = [ev(e) for e in uw]
    for (a, b), k in zip(PAIRS, fluids.rs_keys_list):
        ds_time[k] = ds_time[k] + xr.DataArray((UW[a] * UW[b]).mean(axis=0), dims=("x", "y", "z"))

    ex = {}
    for i in directions:
        e = {k: ev(v) for k, v in exact[i].items()}
        exi = ds.drop_vars(list(ds.data_vars))
        for k in ("storage", "adv", "prod", "wavewave", "rs_phase"):
            exi[k] = (DIMS, e[k])
        # +d_j R^time = +d_j mean(R) + d_j mean(u~u~) = -mean(rs_phase) - mean(wavewave)
        exi["rs_time"] = (("x", "y", "z"), -(e["rs_phase"] + e["wavewave"]).mean(axis=0))
        for k in ("prss", "sgs", "adm", "nonlin"):
            exi[k] = (DIMS, e[k] - e[k].mean(axis=0, keepdims=True))
        ex[i] = exi
    return ds, ds_time, ex, period


@functools.lru_cache(maxsize=None)
def default_case():
    return build_case()


# =============================================================================
# Helpers
# =============================================================================
def interior(da, m=3):
    return da.isel({d: slice(m, -m) for d in ("x", "y", "z") if d in da.dims})


def rms(da):
    return float(np.sqrt((da ** 2).mean()))


def err(num, ref, crop=interior):
    """Relative RMS error (absolute if the exact term is zero)."""
    nref = rms(crop(ref))
    e = rms(crop(num - ref))
    return (e / nref, "rel") if nref > 1e-14 else (e, "abs")


def show(name, val, kind, tol):
    ok = val < tol
    print(f"    {name:12s} {kind} {val:9.2e}   (tol {tol:.0e})  {'ok' if ok else 'FAIL'}")
    return ok


# =============================================================================
# Checks
# =============================================================================
def check_A_les_closure(case):
    """Storage from neighbouring phases vs. solver RHS terms: no FD -> machine precision."""
    ds, ds_time, _, period = case

    def res(d, T_, **kw):
        les = fluids.compute_wave_les_budget(d, ds_time, 0, T_, verbose=False, **kw)
        return rms(les["residual"]) / rms(les["storage"])

    rev = ds.assign_coords(phase=((1.0 - ds.phase) % 1.0).values).sortby("phase")
    print("[A] LES-term closure: residual / |storage|")
    ok = show("spectral", res(ds, period), "rel", TOL_EXACT)
    print(f"    central      rel {res(ds, period, ddt_method='central'):9.2e}   (2nd-order bias)")
    r_T, r_rev = res(ds, 1.1 * period), res(rev, period)
    ctrl = r_T > 1e-2 and r_rev > 0.5
    print(f"    controls: period x1.1 -> {r_T:.2e}, reversed phase -> {r_rev:.2e} "
          f"(must be large)  {'ok' if ctrl else 'FAIL'}")
    return ok and ctrl


def check_B_terms(case, direction=0, storage="solver", single_phase=None, label=None, **kw):
    ds, ds_time, ex, period = case
    dp, exd = ds, ex[direction]
    if single_phase is not None:
        dp = ds.isel(phase=single_phase, drop=True)
        exd = exd.isel(phase=single_phase, drop=True)
    wb = WAVE[direction](dp, ds_time, storage=storage, period=period, verbose=False, **kw)
    wb.compute()
    print(f"[B] direction {direction}, storage={storage}, "
          f"{'phase ' + str(single_phase) if single_phase is not None else 'all phases'} "
          f"{label or kw or ''}")
    ok = True
    for k in TERMS:
        if k not in wb.data_vars:
            print(f"    {k:12s} missing  FAIL"); ok = False; continue
        val, kind = err(wb[k], exd[k])
        solver_based = k in ("storage", "adm") or (k in ("prss", "sgs")
                                                    and kw.get("prefer_solver_terms", True))
        tol = 1e-12 if kind == "abs" else (TOL_EXACT if solver_based else TOL_FD)
        ok &= show(k, val, kind, tol)
    ok &= show("residual", rms(interior(wb["residual"])) / rms(interior(wb["storage"])),
               "rel", TOL_FD)
    if single_phase is None:
        ok &= "phase" in wb["adv"].dims
    return ok


def check_B_storage_agree(case):
    """storage='solver' (one phase at a time) must equal storage='phases'."""
    ds, ds_time, _, period = case
    a = Wave_x(ds, ds_time, storage="solver", verbose=False); a.compute()
    b = Wave_x(ds, ds_time, storage="phases", period=period, verbose=False); b.compute()
    print("[B] storage: solver vs phases")
    return show("storage", *err(a["storage"], b["storage"]), TOL_EXACT)


def check_B_split(case):
    """totaladv (agg -1) = adv+prod+wavewave+rs_phase+rs_time must equal wave(DuDt)."""
    ds, ds_time, ex, _ = case
    wb = Wave_x(ds, ds_time, verbose=False)
    wb.compute(aggregate=-1)
    print("[B] split: totaladv vs wave(DuDt)")
    return show("split", *err(wb["totaladv"], ex[0]["nonlin"]), TOL_FD)


def check_B_fallbacks(case):
    """Zeroed solver dpdx/xSGS must be ignored and replaced by FD from pbar/tau_ij."""
    ds, ds_time, ex, period = case
    ds2 = ds.copy()
    for k in ("dpdx", "xSGS"):
        ds2[k] = xr.zeros_like(ds[k])
    wb = Wave_x(ds2, ds_time, storage="phases", period=period, verbose=False)
    wb.compute()
    print("[B] fallbacks: dpdx, xSGS zeroed -> FD")
    ok = show("prss(FD)", *err(wb["prss"], ex[0]["prss"]), TOL_FD)
    ok &= show("sgs(FD)", *err(wb["sgs"], ex[0]["sgs"]), TOL_FD)
    try:  # storage='solver' must refuse broken solver terms
        Wave_x(ds2, ds_time, storage="solver", verbose=False).compute()
        print("    storage='solver' accepted zeroed terms  FAIL"); ok = False
    except ValueError:
        print("    storage='solver' rejects zeroed terms   ok")
    return ok


def check_B_aggregation(case):
    ds, ds_time, _, _ = case
    wb = Wave_x(ds, ds_time, verbose=False)
    wb.compute(aggregate=1)
    a1, j_ok = wb["adv"].sum("j"), "j" in wb["adv"].dims
    wb.compute(aggregate=0)
    tot0 = sum(wb[k] for k in ("adv", "prod", "wavewave", "rs_phase", "rs_time"))
    d0 = float(np.abs(a1 - wb["adv"]).max())
    wb.compute(aggregate=-1)
    d1 = float(np.abs(wb["totaladv"] - tot0).max())
    print("[B] aggregation")
    ok = show("sum_j(agg1)", d0, "abs", 1e-12) & show("totaladv", d1, "abs", 1e-12)
    ok &= j_ok and "phase" in wb["residual"].dims and list(wb.data_vars)[-1] == "residual"
    print(f"    j kept at agg=1, phase kept, residual last  {'ok' if ok else 'FAIL'}")
    return ok


def check_C_convergence():
    """FD path converges at 2nd order (same physical sub-box on both grids)."""
    keys = ("adv", "prod", "wavewave", "rs_phase", "rs_time", "prss", "sgs")
    box = dict(x=slice(0.4, 1.6), y=slice(-0.9, 0.9), z=slice(-0.9, 0.9))
    crop = lambda da: da.sel(box)
    errs = []
    for n in ((17, 25, 25), (33, 49, 49)):        # dx exactly halves
        ds, ds_time, ex, T_ = build_case(n=n, nphase=6, xlim=(0, 2), ylim=(-1.2, 1.2),
                                         zlim=(-1.2, 1.2), directions=(0,))
        wb = Wave_x(ds, ds_time, storage="phases", period=T_,
                    prefer_solver_terms=False, verbose=False)
        wb.compute()
        errs.append({k: err(wb[k], ex[0][k], crop=crop)[0] for k in keys})
    print("[C] grid convergence (FD path)")
    ok = True
    for k in keys:
        order = np.log2(errs[0][k] / errs[1][k])
        good = 1.5 < order < 2.6
        ok &= good
        print(f"    {k:9s} {errs[0][k]:.2e} -> {errs[1][k]:.2e}  order {order:.2f}  "
              f"{'ok' if good else 'FAIL'}")
    return ok


def check_D_binning(f=0.5, dt=0.01, ncycles=200, nbins=10, tol=0.05, noise=0.1, seed=0):
    """Single-point replica of the Fortran binning (check_runBudget + DumpBudget0)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(ncycles / (f * dt)))) * dt
    phi = np.mod(f * t, 1.0)
    A1, A2 = 0.2, 0.05
    u = 1 + A1*np.cos(2*np.pi*phi) + A2*np.sin(4*np.pi*phi) + noise*rng.standard_normal(t.size)
    att = lambda m: np.sinc(2 * m * tol)
    print(f"[D] binning: nbins={nbins}, tol={tol}, ~{2*tol/(f*dt):.1f} samples/bin/cycle, "
          f"{ncycles} cycles; harmonic-1 attenuation {1-att(1):.2%}")
    print("    phase   n     <u> - filtered     R_uu/noise^2")
    used, Ub, nb = np.zeros(t.size, bool), [], []
    for p0 in np.arange(nbins) / nbins:
        m = np.abs(((phi - p0 + 0.5) % 1.0) - 0.5) < tol
        used |= m
        U, R = u[m].mean(), (u[m]**2).mean() - u[m].mean()**2
        U_ex = 1 + A1*att(1)*np.cos(2*np.pi*p0) + A2*att(2)*np.sin(4*np.pi*p0)
        Ub.append(U); nb.append(m.sum())
        print(f"    {p0:5.3f} {m.sum():5d}   {U - U_ex:+.1e}            {R/noise**2:.3f}")
    Ub, nb = np.array(Ub), np.array(nb)
    d = abs((nb * Ub).sum() / nb.sum() - u.mean())
    print(f"    samples in no bin: {(~used).sum()} of {t.size}")
    print(f"    count-weighted bin mean vs time mean: {d:.1e}  (exact only if no gaps)")
    return d < 1e-12 if not (~used).any() else True


def check_E_phase_vs_time(case):
    """Check 0 helper on the synthetic data."""
    ds, ds_time, _, _ = case
    c0 = fluids.compare_phase_to_time(ds, ds_time, keys=["ubar", "DuDt", "xAD"])
    print("[E] Check 0 helper")
    return all(show(k, v, "rel", 1e-12) for k, v in c0.items())


# =============================================================================
# pytest entry points
# =============================================================================
def test_A():           assert check_A_les_closure(default_case())
def test_B_x():         assert check_B_terms(default_case(), 0)
def test_B_y():         assert check_B_terms(default_case(), 1)
def test_B_z():         assert check_B_terms(default_case(), 2)
def test_B_single():    assert check_B_terms(default_case(), 0, single_phase=3)
def test_B_phases():    assert check_B_terms(default_case(), 0, storage="phases")
def test_B_cons():      assert check_B_terms(default_case(), 0, adv_form="conservative")
def test_B_fdpath():    assert check_B_terms(default_case(), 0, prefer_solver_terms=False)
def test_B_storage():   assert check_B_storage_agree(default_case())
def test_B_split():     assert check_B_split(default_case())
def test_B_fallbacks(): assert check_B_fallbacks(default_case())
def test_B_agg():       assert check_B_aggregation(default_case())
def test_C():           assert check_C_convergence()
def test_D():           assert check_D_binning()
def test_E():           assert check_E_phase_vs_time(default_case())


if __name__ == "__main__":
    case = default_case()
    results = {
        "A": check_A_les_closure(case),
        "B_x": check_B_terms(case, 0),
        "B_y": check_B_terms(case, 1),
        "B_z": check_B_terms(case, 2),
        "B_single": check_B_terms(case, 0, single_phase=3),
        "B_phases": check_B_terms(case, 0, storage="phases"),
        "B_cons": check_B_terms(case, 0, adv_form="conservative"),
        "B_fdpath": check_B_terms(case, 0, prefer_solver_terms=False),
        "B_storage": check_B_storage_agree(case),
        "B_split": check_B_split(case),
        "B_fallbacks": check_B_fallbacks(case),
        "B_agg": check_B_aggregation(case),
        "C": check_C_convergence(),
        "D": check_D_binning(f=0.5, dt=0.01, nbins=10, tol=0.05),  # use your surge_freq, dt
        "E": check_E_phase_vs_time(case),
    }
    print("\nSUMMARY:", {k: ("ok" if v else "FAIL") for k, v in results.items()})
    assert all(results.values())