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
        \Delta u = u_primary - u_precursor

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
        Rossby number, defined Ro = G/(\Omega L)
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
    ret["wakeadv"] = -deltau_j * dUidxj

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
