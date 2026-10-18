"""
Fluids-specific math operations, etc. 

Kirby Heck
2024 August 01
"""

import numpy as np
import xarray as xr

from . import math_utils as math
from ..gridslice import GridDataset


# some helpful key pairings:
uvw_keys = ["ubar", "vbar", "wbar"]
rs_keys = [["uu", "uv", "uw"], ["uv", "vv", "vw"], ["uw", "vw", "ww"]]
rs_keys_list = ["uu", "uv", "uw", "vv", "vw", "ww"]
tau_keys = [
    ["tau11", "tau12", "tau13"],
    ["tau12", "tau22", "tau23"],
    ["tau13", "tau23", "tau33"],
]
tau_keys_list = ["tau11", "tau12", "tau13", "tau22", "tau23", "tau33"]
dpdxi_keys = ["dpdx", "dpdy", "dpdz"]
AD_keys = ["xAD", "yAD", "zAD"]


# =======================================
# === FLUIDS POSTPROCESSING FUNCTIONS ===
# =======================================


def compute_vort(ds, in_place=False):
    """see compute_vorticity()"""
    return compute_vorticity(ds, in_place=in_place)


def compute_vorticity(ds, uvw_keys=None, in_place=False):
    """
    Computes the vorticity vector w_i = [w_x, w_y, w_z]

    Parameters
    ----------
    ds : xr.Dataset with keys ['ubar', 'vbar', 'wbar']
        Budget object or Slice() from BudgetIO.slice()
    uvw_keys : list, optional
        List of keys to use for u, v, w. Default ['ubar', 'vbar', 'wbar']
    in_place : bool, optional
        Returns result if False. Default False

    Returns
    -------
    w_i : (Nx, Ny, Nz, 3)
        4D Vorticity tensor, if in_place=False
    """
    u_i = math.assemble_tensor_1d(ds, uvw_keys or ["ubar", "vbar", "wbar"])
    dxi = ds.grid.dxi

    # TODO: This can be made ~33% faster if we do not compute along i=j=k
    dukdxj = math.gradient(
        u_i, dxi, axis=(0, 1, 2), stack=-2
    )  # stack along -2 axis so indexing is x, y, z, j, k

    w_i = np.zeros(ds.grid.shape + (3,))
    for i in range(3):  # can this be done w/o a loop?
        w_i[..., i] = np.sum(math.E_ijk[i, ...] * dukdxj, axis=(-2, -1))

    if in_place:
        ds["w_i"] = xr.DataArray(w_i, dims=("x", "y", "z", "i"))
    else:
        return xr.DataArray(w_i, dims=("x", "y", "z", "i"), coords=ds.coords)


def compute_delta_field(
    primary, precursor, budget_terms=None, avg_xy=True, in_place=True
):
    """
    Computes deficit fields between primary and precursor simulations, e.g.:
        Delta u = u_primary - u_precursor

    This definition follows the double decomposition in Martinez-Tossas, et al. (2021).

    Parameters
    ----------
    primary : BudgetIO
        Primary simulation budget object
    precursor : BudgetIO
        Precursor simulation budget object
    budget_terms : list, optional
        Budget terms to compute deficits.
        Default is all shared keys between primary and precursor
    avg_xy : bool, optional
        Averages precursor terms horizontally if True. Default True

    Returns
    -------
    field : dict
        Deficit fields, either in-place if in_place=True, or

    """

    if budget_terms is None:
        budget_terms = set(primary.keys()) & set(precursor.keys())

    # background fields
    if avg_xy:
        tmp = {key: np.mean(precursor[key], (0, 1)) for key in budget_terms}
    else:
        tmp = precursor

    if in_place:
        field = primary
    else:
        field = {}

    # for each key, compute primary - precursor:
    for key in budget_terms:
        dkey = key + "_deficit"
        if dkey in field.keys():
            continue  # skip repeat computations
        field[dkey] = primary[key] - tmp[key]

    return field


def compute_tke(ds):
    """Computes TKE"""
    return 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])


def assemble_Ui(ds, dim="i"): 
    """
    Assembles the velocity tensor U_i from an xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Budget object with keys ['ubar', 'vbar', 'wbar']
    dim : str, optional
        Dimension to assemble along. Default is 'i'.

    Returns
    -------
    U_i : xr.DataArray
        Assembled velocity tensor with dimensions (x, y, z, i)
    """
    return math.assemble_xr_1d(ds, uvw_keys, dim=dim)


def assemble_uiuj(ds, dim_i="i", dim_j="j"): 
    """
    Assemble the reynolds stress tensor

    Parameters
    ----------
    ds : xr.Dataset
        Budget object with keys ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
    dim_i, dim_j : str, optional
    
    Returns
    -------
    uiuj : xr.DataArray
        Assembled reynolds stress tensor with dimensions (x, y, z, dim_i, dim_j)
    """
    return math.assemble_xr_nd(ds, rs_keys, dim=(dim_i, dim_j))


def compute_dUidxj(ds, dim_i="i", dim_j="j"):
    """
    Computes the velocity gradient tensor d(Ui)/dxj.
    """
    U_i = assemble_Ui(ds, dim=dim_i)
    dUidxj = math.xr_gradient(U_i, dim=["x", "y", "z"], concat_along=dim_j)
    return dUidxj


def compute_Sij(ds, dim_i="i", dim_j="j"):
    """
    Computes the symmetric part of the velocity gradient tensor S_ij = 0.5 * (dUidxj + dUjdxi).

    Parameters
    ----------
    ds : xr.Dataset
        Budget object with keys ['ubar', 'vbar', 'wbar']
    dim_i : str, optional
        Dimension for i. Default is 'i'.
    dim_j : str, optional
        Dimension for j. Default is 'j'.

    Returns
    -------
    Sij : xr.DataArray
        Symmetric velocity gradient tensor with dimensions (x, y, z, i, j)
    """
    # compute the velocity gradient tensor
    dUidxj = compute_dUidxj(ds, dim_i=dim_i, dim_j=dim_j)
    return 0.5 * (dUidxj + dUidxj.rename(dict(i=dim_j, j=dim_i)))


# ================ BUDGET COMPUTATIONS ================


def compute_RANS(
    ds,
    direction,
    Ro=None,
    lat=None,
    galpha=0,
    fplane=True,
    is_stratified=True,
    theta0=0,
    Fr=0.4,
):
    """
    Computes RANS momentum budgets in the i direction.

    NOTE: This code was written before the migration to xarray. It
    would be better to use xarray.Dataset and xarray.DataArray objects
    for all computations. #TODO

    Parameters
    ----------
    ds : xr.Dataset
    direction : int
        Direction, either 0, 1, 2 (x, y, z, respectively)
    Ro : float
        Rossby number, defined Ro = G/(Omega L)
    lat : float
        Latitude, in radians. Default None
    galpha : float, optional
        Geostrophic wind direction, in radians. Default 0
    fplane : bool
        Use f-plane approximation if true, default True
    is_stratified : bool
        Adds buoyancy term to z-momentum equations if True. Default True.
    theta0 : float
        Reference potential temperature in buoyancy equation. Default 0.
    Fr : float
        Froude number Fr = U/sqrt(gL). Default 0.4.
    """

    dims = ds.grid.shape
    dxi = ds.grid.dxi
    i = direction
    dpdxi_keys = ["dpdx", "dpdy", "dpdz"]

    # assemble tensors
    u_j = math.assemble_tensor_1d(ds, ["ubar", "vbar", "wbar"])
    uu_ij = math.assemble_tensor_1d(ds, rs_keys[i])
    tau_ij = math.assemble_tensor_1d(ds, tau_keys[i])
    G = [np.cos(galpha), np.sin(galpha), 0]

    # compute gradients and divergence
    duidxj = math.gradient(u_j[..., i], dxi)
    duiujdxj = math.div(uu_ij, dxi)
    sgs_ij = math.div(tau_ij, dxi)
    if dpdxi_keys[i] in ds.data_vars:
        dpdxi = ds[dpdxi_keys[i]]
    else:
        dpdxi = -math.gradient(ds["pbar"], dxi, axis=i)

    ret = dict()
    ret["adv"] = -u_j * duidxj
    ret["prss"] = dpdxi
    ret["rs"] = -duiujdxj
    ret["sgs"] = -sgs_ij
    try:
        ret["adm"] = ds[AD_keys[i]]
    except KeyError:
        pass

    ret["geo"] = (
        -2 / Ro * np.sin(lat) * math.e_ijk(i, 1 - i, 2) * G[1 - i] * np.ones(dims)
    )
    if fplane:
        ret["cor"] = 2 / Ro * np.sin(lat) * math.e_ijk(i, 1 - i, 2) * (u_j[..., 1 - i])
    else:
        raise NotImplementedError("TODO: add full coriolis term")

    if i == 2 and is_stratified and theta0 is not None:  # stratification term:
        Tbar = ds["Tbar"]
        tref = np.mean(Tbar, (0, 1))  # how buoyancy is defined in PadeOps
        ret["buoy"] = (Tbar - tref) / (theta0 * Fr**2)

    # hotfix - cast to xarray
    ret_ds = GridDataset(coords=ds.coords).expand_dims(j=(0, 1, 2))
    for key, val in ret.items():
        dims = ("x", "y", "z", "j")
        ret_ds[key] = xr.DataArray(val, dims=dims[: val.ndim])

    return ret_ds


def deficit_budget(
    ds_full, ds_bkgd, direction, Ro=None, lat=None, fplane=True, avg_xy=True
):
    """
    Computes the streamwise momentum deficit budget

    NOTE: This code was written before the migration to xarray. It
    would be better to use xarray.Dataset and xarray.DataArray objects
    for all computations. #TODO

    Parameters
    ----------
    full : dict
        Dictionary of full velocity fields
    bkgd : dict
        Dictionary of background fields
    direction : int
        Direction to compute budgets (0 -> x, 1 -> y, 2 -> z)
    Ro : float, optional
        Rossby number. Default None (ignores Coriolis terms)
    lat : float, optional
        Latitude in degrees. Default None.
    fplane : bool, optional
        Use f-plane approximation. Default True.
    avg_xy : bool, optional
        xy-averages precursor fields if True. Default True.

    Returns
    -------
    xr.Dataset
        Dataset of of deficit budget fields:
        - adv: Advection of mean deficit by full flow (negative, moved to RHS)
        - prss: Deficit pressure gradient
        - rsfull: Full flow Reynolds stresses
        - rsdeficit: Deficit flow Reynolds stresses
        - sgs: Subgrid stress
        - cor: Coriolis
        - adm: Actuator disk sink
        - wakeadv: Wake advecting the mean flow
    """

    i = direction
    if i == 2:
        raise NotImplementedError("TODO: Delta_w deficit budgets")

    ret = dict()
    dims = ds_full.grid.shape
    dxi = ds_full.grid.dxi

    # compute deficit fields:
    compute_delta_field(ds_full, ds_bkgd, avg_xy=avg_xy, in_place=True)

    # construct tensors
    deltau_j = math.assemble_tensor_1d(
        ds_full, ["ubar_deficit", "vbar_deficit", "wbar_deficit"]
    )
    u_j = math.assemble_tensor_1d(ds_full, ["ubar", "vbar", "wbar"])  # full field
    U_j = math.assemble_tensor_1d(ds_bkgd, ["ubar", "vbar", "wbar"])  # bkgd field
    uu_full_ij = math.assemble_tensor_1d(ds_full, rs_keys[i])
    uu_bkgd_ij = math.assemble_tensor_1d(ds_bkgd, rs_keys[i])
    tau_full_ij = math.assemble_tensor_1d(ds_full, tau_keys[i])
    tau_bkgd_ij = math.assemble_tensor_1d(ds_bkgd, tau_keys[i])

    # numerics:
    dduidxj = math.gradient(deltau_j[..., i], dxi)  # gradient of velocity deficit field
    dUidxj = math.gradient(U_j[..., i], dxi)  # gradient of background velocity field
    dpdxi = math.gradient(ds_full["pbar_deficit"], dxi, axis=i)
    duiujdxj_full = math.div(uu_full_ij, dxi)
    duiujdxj_bkgd = math.div(uu_bkgd_ij, dxi)
    sgs_ij = math.div(
        tau_full_ij - tau_bkgd_ij, dxi
    )  # could split this into components as well

    # compute momentum deficit terms
    ret["adv"] = -u_j * dduidxj  # deficit advection
    ret["prss"] = -dpdxi
    ret["sgs"] = -sgs_ij
    ret["rsfull"] = -duiujdxj_full
    ret["rsbkgd"] = duiujdxj_bkgd
    ret["wakeadv"] = -deltau_j * dUidxj

    try:
        ret["adm"] = ds_full[AD_keys[i]]
    except KeyError:
        pass

    if fplane:
        ret["cor"] = (
            2 * math.e_ijk(i, 1 - i, 2) * np.sin(lat) / Ro * deltau_j[..., 1 - i]
        )
    else:
        raise NotImplementedError("TODO: Deficit budgets full Coriolis")

    # hotfix - cast to xarray
    ret_ds = GridDataset(coords=ds_full.coords).expand_dims(j=(0, 1, 2))
    for key, val in ret.items():
        dims = ("x", "y", "z", "j")
        ret_ds[key] = xr.DataArray(val, dims=dims[: val.ndim])

    return ret_ds


def compute_vort_budget(
    ds,
    direction,
    Ro=None,
    lat=45.0,
    fplane=True,
    Fr=None,
    theta0=300.0,
    aggregate=1,
):
    """
    Computes the offline vorticity budget in the requested component direction.

    Terms are nd arrays [x, y, z, j, k, m] up to the level of aggregation.
    For example, the default (aggregate=1) will return [x, y, z, j] only.

    Parameters
    ----------
    field_dict : xr.Dataset
        Dataset expecting velocity, temperature, subgrid stresses, and reynolds stresses
    direction : int
        Direction to compute the vorticity budget
    Ro : float
        Rossby number as defined in LES
    lat : float
        Latitude in degrees, default is 45, NOT None.
    fplane : bool
        Use fplane approximation. Default True.
    Fr : float
        Froude number, defined Fr = G/sqrt(g*L_c)
    theta0 : float
        Reference potential temperature, Default 300 [K].
    aggregate : int, optional
        Aggregation level for the output. Default 1.

    Returns
    -------
    xr.Dataset
        Vorticity budget terms, including:
        - adv: Advection due to mean flow (negative, moved to RHS)
        - str: Vortex stretching
        - buoy: Buoyancy torque
        - sgs: Subgrid stress torque
        - rs: Reynolds stress torque
        - cor: Coriolis/planetary vorticity
        - adm: Turbine forcing
    """

    dims = ds.grid.shape
    dxi = ds.grid.dxi
    ii = int(direction)

    # allocate memory for all the tensors
    adv_ij = np.zeros(dims + (3,))
    str_ij = np.zeros(dims + (3,))
    buoy_ij = np.zeros(dims + (3,))
    sgs_ijkm = np.zeros(dims + (3, 3, 3))  # 6D, [x, y, z, j, k, m]
    rs_ijkm = np.zeros(dims + (3, 3, 3))  # also 6D
    if fplane:
        cor_i = np.zeros(dims)
    else:
        cor_i = np.zeros(dims + (3,))

    # check all required tensors exist:  (may throw KeyError)
    u_i = math.assemble_tensor_1d(ds, keys=["ubar", "vbar", "wbar"])
    w_i = compute_vorticity(ds, in_place=False)
    # save vorticity keys to the budget object:
    ds["w_i"] = w_i
    uiuj = math.assemble_tensor_nd(ds, rs_keys)
    tau_ij = math.assemble_tensor_nd(ds, tau_keys)
    Tbar = ds["Tbar"]
    # let's also add turbine forcing
    xAD = ds["xAD"] if "xAD" in ds else np.zeros(dims)
    yAD = ds["yAD"] if "yAD" in ds else np.zeros(dims)
    zAD = ds["zAD"] if "zAD" in ds else np.zeros(dims)
    AD = np.stack([xAD, yAD, zAD], axis=-1)  # this is the actual forcing
    AD_ijk = np.zeros(dims + (3,))  # this is the vorticity budget term
    compute_AD = np.any(AD)  # compute ADM component - boolean

    # compute coriolis
    if fplane:
        cor_i = 2 / Ro * np.sin(lat) * math.gradient(u_i[:, :, :, ii], dxi, axis=2)
    else:
        raise NotImplementedError(
            "compute_vort_budget(): fplane = False not implemented"
        )

    # Compute remaining tensor quantities
    for jj in range(3):
        # advection (on RHS, flipped sign)
        adv_ij[..., jj] = -u_i[..., jj] * math.gradient(w_i[..., ii], dxi, axis=jj)

        # vortex stretching
        str_ij[..., jj] = w_i[:, :, :, jj] * math.gradient(u_i[..., ii], dxi, axis=jj)

        # buoyancy torque
        if theta0 is not None:
            eijk = math.e_ijk(ii, jj, 2)  # buoyancy term has k=3
            if eijk == 0:
                buoy_ij[..., jj] = 0  # save compute time by skipping these
            else:
                buoy_ij[..., jj] = (
                    eijk * math.gradient(Tbar, dxi, axis=jj) / (Fr**2 * theta0)
                )

        for kk in range(3):
            eijk = math.e_ijk(ii, jj, kk)
            # nothing is ijk at the moment, Coriolis w/o trad. approx. is, however

            # so is turbine forcing:
            if compute_AD and eijk != 0:
                AD_ijk = eijk * math.gradient(AD[..., kk], dxi, axis=jj)

            for mm in range(3):
                # compute permutation operator

                if eijk == 0:
                    sgs_ijkm[..., jj, kk, mm] = 0
                    rs_ijkm[..., jj, kk, mm] = 0
                else:
                    sgs_ijkm[..., jj, kk, mm] = eijk * math.gradient(
                        math.gradient(-tau_ij[..., kk, mm], dxi, axis=mm),
                        dxi,
                        axis=jj,
                    )
                    rs_ijkm[..., jj, kk, mm] = eijk * math.gradient(
                        math.gradient(-uiuj[..., kk, mm], dxi, axis=mm),
                        dxi,
                        axis=jj,
                    )

    # hotfix - cast to xarray
    ret = {
        "adv": adv_ij,
        "str": str_ij,
        "buoy": buoy_ij,
        "sgs": sgs_ijkm,
        "cor": cor_i,
        "adm": AD_ijk,
        "rs": rs_ijkm,
    }
    ret_ds = GridDataset(coords=ds.coords).expand_dims(
        j=(0, 1, 2), k=(0, 1, 2), m=(0, 1, 2)
    )
    for key, val in ret.items():
        dims = ["x", "y", "z", "j", "k", "m"][: val.ndim]
        ret_ds[key] = xr.DataArray(val, dims=dims)

    # aggregate down to more manageable dimensions
    math.new_aggregation(ret_ds, base_agg=aggregate, in_place=True)

    return ret_ds


def compute_mke_budget(ds, Fr=None, theta0=300.0, aggregate=("i", "j")):
    """
    Computes the mean kinetic energy budget.

    This is the first budget explicitly written for xarray Datasets.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with time-averaged fields
    Fr : float, optional
        Froude number, defined Fr = U/sqrt(gL). If None,
        no buoyancy term is computed. Default None.
    theta0 : float, optional
        Reference potential temperature. Default 300.0
    aggregate: tuple, optional
        Aggregation level for the output. Default ("i", "j") (sum over all i, j)

    Returns
    -------
    xr.Dataset
        Dataset with budget terms:
        - adv: MKE advection from mean flow (negative, moved to RHS)
        - prss: Pressure work term
        - buoy: Buoyancy term (if Fr is not None)
        - shear: Shear production term
        - transport: Turbulent transport term
        - diss: Dissipation term
        - adm: Turbine forcing term
    """

    u_i = math.assemble_xr_1d(ds, ["ubar", "vbar", "wbar"])
    uiuj = math.assemble_xr_nd(ds, rs_keys, dim=("i", "j"))
    tau_ij = math.assemble_xr_nd(ds, tau_keys, dim=("i", "j"))

    # let's also add turbine forcing
    xAD = ds["xAD"] if "xAD" in ds else xr.zeros_like(ds["ubar"])
    yAD = ds["yAD"] if "yAD" in ds else xr.zeros_like(ds["ubar"])
    zAD = ds["zAD"] if "zAD" in ds else xr.zeros_like(ds["ubar"])
    AD = xr.concat([xAD, yAD, zAD], dim="i")  # this is the actual forcing
    compute_AD = np.any(AD)  # compute ADM component - boolean

    if all([key in ds.data_vars for key in ["dpdx", "dpdy", "dpdz"]]):
        pgf = -math.assemble_xr_1d(ds, ["dpdx", "dpdy", "dpdz"], dim="i")
    else:
        pgf = math.xr_gradient(ds["pbar"], dim=("x", "y", "z"), concat_along="i")

    # Compute budget terms now:
    mke = 0.5 * (u_i ** 2).sum("i")
    ret = (
        GridDataset(coords=ds.coords)
        .expand_dims(i=(0, 1, 2), j=(0, 1, 2))
        .transpose("x", "y", "z", "i", "j")
    )
    ret["adv"] = -u_i * math.xr_gradient(mke, dim=("x", "y", "z"), concat_along="i")
    ret["prss"] = -u_i * pgf
    if Fr is not None and theta0 is not None:
        ret["buoy"] = (
            u_i.sel(i=0)
            * math.xr_gradient(ds["Tbar"], dim=("x", "y", "z"), concat_along="i")
            / (theta0 * Fr**2)
        )
    # ret["shear"] = -u_i * math.xr_div(uiuj, dim="j", sum=False)
    ret["shear"] = uiuj * math.xr_gradient(u_i, dim=("x", "y", "z"), concat_along="j")
    ret["transport"] = -math.xr_div(uiuj * u_i, dim="j", sum=False)
    ret["diss"] = -u_i * math.xr_div(tau_ij, dim="j", sum=False)
    if compute_AD:
        ret["adm"] = u_i * AD

    # aggregate down
    return ret.transpose(..., *aggregate).sum(aggregate)

def compute_residual(ds_budget, in_place=False):
    """
    Computes the residual of a budget dataset.

    Parameters
    ----------
    ds_budget : xr.Dataset
        Dataset with budget terms

    Returns
    -------
    xr.DataArray
        Residual of the budget
    """
    # compute residual
    tmp = math.new_aggregation(ds_budget, base_agg=0)
    residual = sum(tmp[key] for key in tmp)

    if in_place:
        ds_budget["residual"] = residual
    else:
        return residual

# ============================================================================
# ============ WAVE-COHERENT (TRIPLE-DECOMPOSITION) MOMENTUM BUDGET ==========
# ============================================================================
#   f = fbar + f~ + f'      <f>(phi) = fbar + f~      u' = u - <u|phi>
#   fbar from the time-averaged budget, <f> from the phase-averaged budget.

_xyz = ("x", "y", "z")
sgs_keys = ["xSGS", "ySGS", "zSGS"]

# PadeOps budget-1 terms: the full RHS of the resolved momentum equation.
# (zAD is not written by PadeOps; it is listed so tests/other data can supply it.)
les_keys = [
    {"nonlin": "DuDt", "prss": "dpdx", "sgs": "xSGS", "adm": "xAD", "cor": "xCor", "geo": "xGeo"},
    {"nonlin": "DvDt", "prss": "dpdy", "sgs": "ySGS", "adm": "yAD", "cor": "yCor", "geo": "yGeo"},
    {"nonlin": "DwDt", "prss": "dpdz", "sgs": "zSGS", "adm": "zAD", "buoy": "zBuoy"},
]


def _usable(da):
    """False if a field is identically zero (e.g. broken budget-1 output)."""
    return bool(np.abs(da).max() > 0)


def sum_over(da, dims=("j",)):
    """Sum over whichever of `dims` exist in `da`."""
    present = [d for d in dims if d in da.dims]
    return da.sum(present) if present else da


def phase_mean(da, phase_dim="phase", weights=None):
    """Mean over phase bins (diagnostics only)."""
    return da.mean(phase_dim) if weights is None else da.weighted(weights).mean(phase_dim)


def phase_ddt(da, period, phase_period=1.0, phase_dim="phase",
              method="spectral", atol=0.006):
    """
    d/dt of a phase-averaged field from neighbouring phase bins:
        d/dt = (phase_period / period) d/dphi.
    Bins must be sorted, uniform and tile one cycle. "spectral" (FFT) is exact for
    resolved harmonics; "central" is 2nd order.
    """
    phi = np.asarray(da[phase_dim].values, dtype=float)
    n = phi.size
    if n < 3:
        raise ValueError("phase_ddt(): need >= 3 phase bins.")
    dphi = phase_period / n
    if np.max(np.abs(phi - (phi[0] + dphi * np.arange(n)))) > atol * phase_period:
        raise ValueError("phase_ddt(): bins must be sorted, uniform and tile one cycle.")
    scale = phase_period / period
    if method == "central":
        fwd = da.roll({phase_dim: -1}, roll_coords=False)
        bwd = da.roll({phase_dim: 1}, roll_coords=False)
        return (fwd - bwd) / (2 * dphi) * scale
    if method != "spectral":
        raise ValueError("phase_ddt(): method must be 'spectral' or 'central'")
    k = 2j * np.pi * np.fft.fftfreq(n, d=dphi)
    if n % 2 == 0:
        k[n // 2] = 0.0
    _d = lambda a: np.real(np.fft.ifft(k * np.fft.fft(a, axis=-1), axis=-1))
    out = xr.apply_ufunc(_d, da, input_core_dims=[[phase_dim]], output_core_dims=[[phase_dim]])
    return out.transpose(*da.dims) * scale


def _wave_of(ds_phase, ds_time, key, say):
    """Wave part <f> - fbar of a stored field, or None if missing/unusable."""
    if key is None or key not in ds_phase.data_vars or key not in ds_time.data_vars:
        return None
    if not (_usable(ds_phase[key]) and _usable(ds_time[key])):
        say(f"'{key}' is identically zero in the phase or time budget; not using it.")
        return None
    return ds_phase[key] - ds_time[key]


def wave_storage_from_solver(ds_phase, ds_time, direction, verbose=True):
    """
    du~_i/dt = sum_k (S_k^phase - S_k^time) over all PadeOps RHS terms S_k.
    Works for a single phase. Every RHS term with nonzero contribution must be loaded
    (DuDt, dpdx, xSGS, xAD; plus xCor/xGeo if Coriolis matters).
    """
    i = int(direction)
    parts = []
    for name, key in les_keys[i].items():
        if key in ds_phase.data_vars and key in ds_time.data_vars:
            if not (_usable(ds_phase[key]) and _usable(ds_time[key])):
                raise ValueError(f"storage='solver': '{key}' is identically zero; "
                                 "budget-1 output is broken (see Fortran fix).")
            parts.append(ds_phase[key] - ds_time[key])
        elif name == "nonlin":
            raise KeyError(f"storage='solver' needs '{key}' in both phase and time budgets")
    return sum(parts)


def compute_wave_budget(
    ds_phase,
    ds_time,
    direction,
    storage="solver",
    period=None,
    phase_period=1.0,
    phase_dim="phase",
    adv_form="advective",
    coriolis=None,
    buoyancy=False,
    prefer_solver_terms=True,
    ddt_method="spectral",
    verbose=True,
):
    """
    Wave-coherent momentum budget at one or more phases. All terms on the RHS (sum -> 0):

      storage  : -d u~_i/dt              (storage = "solver" | "phases" | None)
      adv      : -ubar_j d u~_i/dx_j     (wave advected by mean flow)
      prod     : -u~_j d ubar_i/dx_j     (mean shear acting on the wave)
      wavewave : -u~_j d u~_i/dx_j       (wave-wave interaction at this phase)
      rs_phase : -d <u_i'u_j'>_phi/dx_j  (turbulent stress at this phase)
      rs_time  : +d R^time_ij/dx_j       (time-avg total stress = mean wave + mean turb.)
      prss     : wave part of pressure gradient
      sgs      : wave part of SGS (+ wall) stress divergence
      adm      : wave part of turbine forcing
      cor/buoy : wave part of Coriolis / buoyancy (solver terms; off by default)

    adv + prod + wavewave + rs_phase + rs_time = wave part of DuDt (skew-symmetric).
    adv_form="conservative" writes the first three as divergences (same if div-free).

    Parameters
    ----------
    ds_phase : phase-averaged fields; dims (x, y, z) or (phase, x, y, z)
    ds_time : time-averaged fields; dims (x, y, z)
    direction : 0, 1, 2
    storage : "solver" (one phase OK), "phases" (needs all bins + period), or None
    period : wave period in simulation time units (1/surge_freq)
    """
    i = int(direction)
    if i not in (0, 1, 2):
        raise ValueError("direction must be 0, 1, or 2")

    def say(msg):
        if verbose:
            print("compute_wave_budget():", msg)

    grad = lambda da: math.xr_gradient(da, dim=_xyz, concat_along="j")
    div = lambda F: math.xr_div(F, dim="j", sum=False)
    in_both = lambda k: k in ds_phase.data_vars and k in ds_time.data_vars
    solver = lambda name: (_wave_of(ds_phase, ds_time, les_keys[i].get(name), say)
                           if prefer_solver_terms else None)

    # ---- velocities ----
    u_phase = assemble_Ui(ds_phase, dim="i")   # <u_i>(phi)
    U = assemble_Ui(ds_time, dim="i")          # ubar_i
    u_w = u_phase - U                          # u~_i
    U_j, u_w_j = U.rename(i="j"), u_w.rename(i="j")
    U_i, u_w_i = U.isel(i=i, drop=True), u_w.isel(i=i, drop=True)

    ret = {}

    # ---- storage ----
    if storage == "solver":
        ret["storage"] = -wave_storage_from_solver(ds_phase, ds_time, i, verbose)
    elif storage == "phases":
        if period is None or phase_dim not in ds_phase.dims:
            raise ValueError("storage='phases' needs all phase bins (a 'phase' dim) and `period`")
        ret["storage"] = -phase_ddt(u_w_i, period, phase_period, phase_dim, method=ddt_method)
    elif storage is not None:
        raise ValueError("storage must be 'solver', 'phases', or None")

    # ---- nonlinear terms ----
    if adv_form == "advective":
        ret["adv"] = -U_j * grad(u_w_i)
        ret["prod"] = -u_w_j * grad(U_i)
        ret["wavewave"] = -u_w_j * grad(u_w_i)
    elif adv_form == "conservative":
        ret["adv"] = -div(U_j * u_w_i)
        ret["prod"] = -div(u_w_j * U_i)
        ret["wavewave"] = -div(u_w_j * u_w_i)
    else:
        raise ValueError("adv_form must be 'advective' or 'conservative'")
    ret["rs_phase"] = -div(math.assemble_xr_1d(ds_phase, rs_keys[i], dim="j"))
    ret["rs_time"] = div(math.assemble_xr_1d(ds_time, rs_keys[i], dim="j"))

    # ---- pressure ----
    f = solver("prss")
    if f is not None:
        ret["prss"] = f                        # PadeOps dpdx is already -dp/dx
    elif in_both("pbar"):
        ret["prss"] = -(ds_phase["pbar"] - ds_time["pbar"]).differentiate(_xyz[i])
    else:
        say("no pressure term available")

    # ---- SGS ----
    f = solver("sgs")
    if f is not None:
        ret["sgs"] = f
    elif all(in_both(k) for k in tau_keys[i]):
        tau_w = (math.assemble_xr_1d(ds_phase, tau_keys[i], dim="j")
                 - math.assemble_xr_1d(ds_time, tau_keys[i], dim="j"))
        ret["sgs"] = -div(tau_w)
    else:
        say(f"no SGS terms for direction {i}")

    # ---- turbine forcing (only available from the solver) ----
    f = _wave_of(ds_phase, ds_time, les_keys[i].get("adm"), say)
    if f is not None:
        ret["adm"] = f

    # ---- Coriolis / buoyancy (solver terms) ----
    if coriolis == "solver":
        f = _wave_of(ds_phase, ds_time, les_keys[i].get("cor"), say)
        if f is not None:
            ret["cor"] = f
    elif coriolis is not None:
        raise ValueError("coriolis must be None or 'solver'")
    if buoyancy:
        f = _wave_of(ds_phase, ds_time, les_keys[i].get("buoy"), say)
        if f is not None:
            ret["buoy"] = f

    out = ds_phase.drop_vars(list(ds_phase.data_vars))
    order = (phase_dim,) + _xyz
    for key, val in ret.items():
        out[key] = val.transpose(*[d for d in order if d in val.dims], ...)
    return out


def compute_wave_residual(ds_budget, sum_dims=("j",), in_place=False):
    """Residual = sum of all terms (summed over j); the phase dimension is kept."""
    residual = sum(sum_over(ds_budget[k], sum_dims)
                   for k in ds_budget.data_vars if k != "residual")
    if in_place:
        ds_budget["residual"] = residual
    else:
        return residual


def compute_wave_les_budget(ds_phase, ds_time, direction, period, phase_period=1.0,
                            phase_dim="phase", ddt_method="spectral", verbose=True):
    """
    Check A (needs all phase bins): storage from neighbouring phases vs. the wave
    parts of the PadeOps RHS terms. No spatial derivatives -> residual ~ 0.
    """
    i = int(direction)
    say = (lambda m: print("compute_wave_les_budget():", m)) if verbose else (lambda m: None)
    out = ds_phase.drop_vars(list(ds_phase.data_vars))
    u_w = ds_phase[uvw_keys[i]] - ds_time[uvw_keys[i]]
    out["storage"] = -phase_ddt(u_w, period, phase_period, phase_dim, method=ddt_method)
    for name, key in les_keys[i].items():
        f = _wave_of(ds_phase, ds_time, key, say)
        if f is not None:
            out[name] = f
    out["residual"] = sum(out[k] for k in list(out.data_vars))
    return out


def split_time_stress(ds_phase, ds_time, direction, phase_dim="phase", weights=None):
    """
    Optional (needs all phase bins): rs_time = rs_wave_mean + rs_turb_mean, where
        rs_wave_mean = +d mean(u~_i u~_j)/dx_j,  rs_turb_mean = +d mean(u_i'u_j')/dx_j.
    """
    i = int(direction)
    div = lambda F: math.xr_div(F, dim="j", sum=False)
    u_w = assemble_Ui(ds_phase, dim="i") - assemble_Ui(ds_time, dim="i")
    wave_stress = phase_mean(u_w.isel(i=i, drop=True) * u_w.rename(i="j"), phase_dim, weights)
    R_time = math.assemble_xr_1d(ds_time, rs_keys[i], dim="j")
    out = ds_time.drop_vars(list(ds_time.data_vars))
    out["rs_wave_mean"] = div(wave_stress)
    out["rs_turb_mean"] = div(R_time - wave_stress)
    return out


def compare_phase_to_time(ds_phase, ds_time, keys=None, phase_dim="phase", weights=None):
    """Check 0: mean over phase bins vs. time file, per key (max relative error)."""
    keys = keys or [k for k in ds_phase.data_vars
                    if k in ds_time.data_vars and k not in rs_keys_list]
    out = {}
    for k in keys:
        ref = float(np.abs(ds_time[k]).max())
        d = float(np.abs(phase_mean(ds_phase[k], phase_dim, weights) - ds_time[k]).max())
        out[k] = d / ref if ref > 0 else np.nan
    return out
