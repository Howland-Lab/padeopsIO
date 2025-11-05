"""
Math utility functions. 

Kirby Heck
2024 July 24
"""

import numpy as np
import xarray as xr
from ..gridslice import GridDataset


def assemble_tensor_1d(field_dict, keys, axis=-1):
    """
    Stacks keys in a 1d tensor in the order they are given.

    Parameters
    ----------
    field_dict : dict
        Dictionary of fields (e.g. {'ubar': <np.ndarray>, 'vbar': <np.ndarray>, 'wbar': <np.ndarray>})
    keys : list
        (Length-3) list of keys
    axis : int, optional
        Axis to stack. Default: -1

    Returns
    -------
    np.ndarray : (Nx, Ny, Nz, len(keys))
    """

    return np.stack([field_dict[key] for key in keys], axis=axis)


def assemble_tensor_nd(field_dict, keys):
    """
    Assembles an n-dimensional tensor from a dictionary of fields.

    Parameters
    ----------
    field_dict : dict
        Dictionary of fields
        (e.g. {'ubar': <np.ndarray>, 'vbar': <np.ndarray>, 'wbar': <np.ndarray>})
    keys : list of lists
        (Length-3) list of lists (of lists of lists...) of keys
    basedim : int, optional
        Number of base dimensions, default 3 (x, y, z).

    Returns
    -------
    np.ndarray
    """

    # keys is a list or a list of nested lists
    key_ls = list(field_dict.keys())
    try:
        ndim = field_dict[key_ls[0]].ndim
    except AttributeError:
        raise  # ???

    def _assemble_nd(keys):
        """Recursive helper function"""
        if isinstance(keys[0], str):  # TODO: what if the key is not a string?
            # base case, prepend stacks
            return assemble_tensor_1d(field_dict, keys, axis=-1)
        else:
            # recursive call: return stack [of stacks], also prepended
            return np.stack([_assemble_nd(key) for key in keys], axis=ndim)

    # use the recursive calls, which prepends each added index (e.g. [x, y, z, i, j, ...])
    tensors_rev = _assemble_nd(keys)

    return tensors_rev


def new_aggregation(ds, base_agg=0, in_place=False, **kwargs):
    """
    Aggregates all terms into dictionary by summing over extra axes

    Parameters
    ----------
    terms : dict
        Dictionary of np.ndarray
    base_agg : int, optional
        Base aggregation level. Default 0
    in_place : bool, optional
        If True, aggregates in-place. Default False.
    kwargs : dict, optional
        Additional levels of (dis)aggregation for keys `newkey`

    Returns
    -------
    xr.Dataset or None
        Dictionary of aggregated values, only if `ret` is None.
    """
    if base_agg < 0:
        raise ValueError("new_aggregation(): base_agg must be >= 0")

    ret = GridDataset(coords=ds.coords) if not in_place else ds

    for key in ds.data_vars.keys():
        if key in kwargs.keys() and not isinstance(kwargs[key], int):
            # assume these are dimensions to aggregate over
            dims = kwargs[key]
        else:
            if key in kwargs.keys():
                agg = kwargs[key]
            else:
                agg = base_agg

            # pick dimensions to sum over
            # e.g., for base_agg=0, sum over all free indices
            free_indices = [c for c in ds[key].coords if c not in ["x", "y", "z"]]
            free_indices.sort()
            dims = free_indices[agg:]

        ret[key] = ds[key].sum(dims)

    return ret


# ==================== index notation help ==========================


def e_ijk(i, j, k):
    """
    Permutation operator, takes i, j, k in [0, 1, 2]

    returns +1 if even permutation, -1 if odd permutation

    TODO: This is not elegant
    """

    if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        return 1
    elif (i, j, k) in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]:
        return -1
    else:
        return 0


# also E_ijk in tensor form:
E_ijk = np.array(
    [[[e_ijk(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)]
)


def d_ij(i, j):
    """
    Kronecker delta
    """
    return int(i == j)


def gradient(f, dx, axis=(0, 1, 2), stack=-1, edge_order=2, **kwargs):
    """
    Compute gradients of f using numpy.gradient

    Parameters
    ----------
    f : ndarray
    dx : float or tuple
        Float (dx) or vector (dx, dy, dz)
    axis : int or tuple
        Axes to compute gradients over. Default is (0, 1, 2) for x, y, z
    stack : int
        Which axis to stack gradients before returning a tensor.
        Default is -1
    edge_order : int
    """

    if hasattr(axis, "__iter__"):
        args = [dx[k] for k in axis]
    else:
        args = [dx[axis]]

    dfdxi = np.gradient(f, *args, axis=axis, edge_order=edge_order, **kwargs)

    if len(args) > 1:
        return np.stack(dfdxi, axis=stack)
    return dfdxi


def div(f, dxi, axis=-1, sum=False, **kwargs):
    """
    Computes the 3D divergence of vector or tensor field f: dfi/dxi

    Parameters
    ----------
    f : (Nx, Ny, Nz, 3) or (Nx, Ny, Nz, ...) array
        Vector or tensor field f
    dxi : tuple
        Vector (dx, dy, dz)
    axis : int, optional
        Axis to compute divergence, default -1
        (Requires that f.shape[axis] = 3)
    sum : bool, optional
        if True, performs implicit summation over repeated indices.
        Default False

    Returns
    -------
    dfi/dxi : f.shape array (if sum=True) OR drops the `axis`
        axis if sum=False
    """

    res = np.zeros(f.shape)

    def get_slice(ndim, axis, index):
        """
        Helper function to slice axis `axis` from ndarray
        """
        s = [slice(None) for i in range(ndim)]
        s[axis] = slice(index, index + 1)
        return tuple(s)

    # compute partial derivatives:
    for i in range(3):
        s = get_slice(f.ndim, axis, i)

        # TODO fix gradient functions everywhere
        res[s] = np.gradient(f[s], dxi[i], axis=i, **kwargs)

    if sum:
        return np.sum(res, axis=axis)
    else:
        return res


# ================== XARRAY FUNCTIONS ========================


def assemble_xr_1d(ds, keys, dim="i", coords=None, rename=None):
    """Assemble 1d xarray tensor by expanding dimensions"""
    if coords is None:
        coords = {dim: range(len(keys))}
    result = xr.concat([ds[key] for key in keys], dim=dim)
    result.coords[dim] = coords[dim]
    return result.rename(rename)


def assemble_xr_nd(ds, nested_keys, dim=("i", "j"), coords=None, rename=None):
    """Assemble nd xarray tensor by recursively expanding dimensions"""
    try:
        newshape = np.array(nested_keys)
    except ValueError as e:
        raise e

    if isinstance(dim, str):
        dim = (dim,)

    if newshape.ndim != len(dim):
        raise ValueError(
            "Dimension mismatch: cannot reshape"
            f"the array into the requested {newshape.ndim}"
            f"dimensions with {len(dim)} directions given."
        )

    def _assemble(keys, level, coords):
        """Define recursive function call"""
        if isinstance(keys[0], str):
            # base level of recursion
            return assemble_xr_1d(ds, keys, dim=dim[level], coords=coords)
        else:
            # top level recursion
            result = xr.concat(
                [_assemble(key, level - 1, coords) for key in keys], dim=dim[level]
            )
            if coords is None:
                coords = {dim[level]: range(len(keys))}
            result.coords[dim[level]] = coords[dim[level]]
            return result

    return _assemble(nested_keys, 0, coords).rename(rename)


def xr_gradient(data, dim, concat_along="i", raise_errors=True):
    """
    Computes the gradient of an xarray.DataArray along specified dimensions.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    dim : str or list/tuple of str
        Dimensions along which to compute the gradient.
    concat_along : str
        Name of the new dimension to concatenate along (only used if dim is a list).
    raise_errors : bool
        Whether to raise errors on differentiation failure (default True).

    Returns
    -------
    gradient : xarray.DataArray
        Gradient of the input data.
    """
    if isinstance(dim, str):
        # Simple 1D gradient: return directly
        return data.differentiate(dim)

    # Multi-dimensional gradient
    grads = []
    for _dim in dim:
        try:
            g = data.differentiate(_dim)
        except ValueError as e:
            if raise_errors:
                raise e
            else:
                g = xr.zeros_like(data)
        grads.append(g)

    # Assign integer coordinate labels for the new dimension
    gradient = xr.concat(grads, dim=concat_along)
    gradient = gradient.assign_coords({concat_along: list(range(len(dim)))})

    return gradient  # Stack gradients along a new dimension


def xr_d2x(ds, dim):
    """
    Computes second derivates along dimension `dim` for xarray.DataArray `ds`.
    """
    d2fdx2 = xr.zeros_like(ds)
    dx = ds[dim].shift({dim: -1}) - ds[dim]
    dx_last = ds[dim][-1] - ds[dim][-2]

    d2fdx2 = (ds.shift({dim: -1}) - 2 * ds + ds.shift({dim: 1})) / (dx**2)
    first4 = ds.isel(
        {dim: slice(0, 4)}
    )  # if there aren't four points in the dimension, this will fail
    last4 = ds.isel({dim: slice(-4, None)})

    # second order downwind:
    d2fdx2[{dim: 0}] = (
        2 * first4
        - 5 * first4.shift({dim: -1})
        + 4 * first4.shift({dim: -2})
        - first4.shift({dim: -3})
    ).isel({dim: 0}) / (dx[0] ** 2)

    # second order upwind:
    d2fdx2[{dim: -1}] = (
        2 * last4
        - 5 * last4.shift({dim: 1})
        + 4 * last4.shift({dim: 2})
        - last4.shift({dim: 3})
    ).isel({dim: -1}) / (dx_last**2)

    return d2fdx2


def xr_laplacian(ds, dim, concat_along="i", sum=False):
    """
    Computes the laplacian of xarray.DataArray `ds` along dimensions `dim`.

    Parameters
    ----------
    ds : xr.Dataset
    dim : str or tuple
        Dimensions to compute laplacian
    concat_along : str, optional
        Dimension to concatenate along. Default is "i".
        If `sum` = True, this is ignored.
    sum : bool
        If True, performs implicit summation over repeated indices. Default False.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Laplacian of `ds`
    """

    ret = GridDataset(coords=ds.coords)

    ret = []
    if isinstance(dim, str):
        dim = (dim,)

    for _dim in dim:
        ret.append(xr_d2x(ds, _dim))

    # rearrange to DataArray
    laplacian = xr.concat(ret, concat_along)

    if sum:
        return laplacian.sum(concat_along)
    else:
        return laplacian


def xr_div(ds, dim, mapping=None, sum=False):
    """
    Computes the 3D divergence of vector or tensor field f: dfi/dxi

    Parameters
    ----------
    ds : xr.Dataset
        Vector or tensor field
    dim : str
        Dimension to compute divergence
    mapping : dict, optional
        Mapping of indices to dimensions along dimension `dim`.
        Default is {0: "x", 1: "y", 2: "z"}
    sum : bool, optional
        if True, performs implicit summation over repeated indices.
        Default False

    Returns
    -------
    dfi/dxi : f.shape array (if sum=True) OR drops the `axis`
        axis if sum=False
    """

    ret = GridDataset(coords=ds.coords)
    mapping = {0: "x", 1: "y", 2: "z"}

    ret = []
    for i in ds.coords[dim]:
        axis = mapping[int(i)]
        ret.append(ds.sel({dim: i}).differentiate(axis))

    # rearrange to DataArray
    div = xr.concat(ret, dim)

    if sum:
        return div.sum(dim)
    else:
        return div


def xr_permutation_tensor(dims=None, coords=None):
    """
    Returns the Levi-Civita symbol for 3D space.

    Parameters
    ----------
    dims : list of str, optional
        Default is ["i", "j", "k"]
    coords : dict of coordinate values, optional.
        If `None`, fills in [0, 1, 2] for all axes.

    Returns
    -------
    E_ijk : xr.DataArray
        3x3x3 Levi-Civita tensor.
    """
    E_ijk = np.zeros((3, 3, 3))
    E_ijk[0, 1, 2] = E_ijk[1, 2, 0] = E_ijk[2, 0, 1] = 1
    E_ijk[0, 2, 1] = E_ijk[2, 1, 0] = E_ijk[1, 0, 2] = -1

    dims = dims or ["i", "j", "k"]
    coords = coords or {dim: [0, 1, 2] for dim in dims}

    ret = xr.DataArray(
        E_ijk,
        dims=dims,
        coords=coords,
    )

    return ret


if __name__ == "__main__":
    # run basic tests:
    assert e_ijk(0, 0, 1) == 0
    assert e_ijk(0, 2, 1) == -1

    tmp = np.reshape(np.arange(24), (2, 3, 4))

    field = {str(k): np.ones((3, 4)) * k for k in range(8)}
    keys_test = [
        [
            [
                "0",
                "1",
            ],
            ["2", "3"],
        ],
        [["4", "5"], ["6", "7"]],
    ]
    ret = assemble_tensor_nd(field, keys_test)

    print(ret[..., 0, 1, 1])
