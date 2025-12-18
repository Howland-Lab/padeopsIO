"""
Improved slice data using xarray Datasets. 

Kirby Heck
2025 January 17
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from .budgetkey import key_labels

labels = key_labels()


class GridDataset(xr.Dataset):
    """
    Gridded xarray Dataset which always contains axes x, y, z.

    The chief benefit of this derived class is to allow for
    directly setting keys as numpy arrays (enabling back-compatibility),
    for example:

    >>> ds = GridDataset(x=x, y=y, z=z)
    >>> ds['u'] = ufield
    """

    __slots__ = ()

    def __init__(self, *args, x=None, y=None, z=None, coords=None, **kwargs):
        if len(args) > 0 and hasattr(args[0], "coords"):
            coords = args[0].coords
        else:
            coords = coords or dict(x=x, y=y, z=z)
        kwargs.update(coords=coords)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            if not (np.allclose(value.shape, self.grid.nxi)):
                raise ValueError(
                    "Number of dimensions of the array does not match the dataset."
                )

            # first, try to set this to physical coordinates x, y, z
            try:
                super().__setitem__(key, (self.grid.keys(), value))
                return
            except ValueError as e:
                pass  # do nothing, for now

            # Assign the variable using existing dimensions
            super().__setitem__(key, (list(self.dims), value))
        else:
            # TODO: Dimension checking?
            super().__setitem__(key, value)


# ================== decorators ==================


# add "grid" attribute to all xarray Datasets and DataArrays
@xr.register_dataset_accessor("grid")
@xr.register_dataarray_accessor("grid")
class GridAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def x(self):
        return self._obj.coords.get("x", None)

    @property
    def y(self):
        return self._obj.coords.get("y", None)

    @property
    def z(self):
        return self._obj.coords.get("z", None)

    @property
    def nx(self):
        return len(self.x) if self.x is not None and self.x.ndim > 0 else 0

    @property
    def ny(self):
        return len(self.y) if self.y is not None and self.y.ndim > 0 else 0

    @property
    def nz(self):
        return len(self.z) if self.z is not None and self.z.ndim > 0 else 0

    @property
    def nxi(self):
        """Number of grid points along each non-singleton axis"""
        res = np.array([self.nx, self.ny, self.nz])
        return res[(res != 0) & (res != 1)]

    @property
    def dx(self):
        return (
            float(self.x[1] - self.x[0])
            if self.x is not None and self.x.ndim > 0 and self.x.size > 1
            else 0
        )

    @property
    def dy(self):
        return (
            float(self.y[1] - self.y[0])
            if self.y is not None and self.y.ndim > 0 and self.y.size > 1
            else 0
        )

    @property
    def dz(self):
        return (
            float(self.z[1] - self.z[0])
            if self.z is not None and self.z.ndim > 0 and self.z.size > 1
            else 0
        )

    @property
    def dxi(self):
        """Grid spacing along each non-singleton axis"""
        res = np.array([self.dx, self.dy, self.dz])
        return res[res != 0]

    @property
    def dV(self):
        """Grid volume"""
        return np.prod(self.dxi)

    @property
    def Lx(self):
        return self.nx * self.dx

    @property
    def Ly(self):
        return self.ny * self.dy

    @property
    def Lz(self):
        return self.nz * self.dz

    @property
    def Lxi(self):
        res = np.array([self.Lx, self.Ly, self.Lz])
        return res[res != 0]

    @property
    def extent(self):
        """Build extents of non-singleton axes"""
        return np.ravel(
            [
                (float(xi.min()), float(xi.max()))
                for xi in [self.x, self.y, self.z]
                if xi is not None and xi.ndim > 0
            ]
        )

    @property
    def xi(self):
        return [
            _x.to_numpy()
            for _x in [self.x, self.y, self.z]
            if _x is not None and _x.ndim > 0
        ]

    @property
    def shape(self):
        return tuple(xi.size for xi in self.xi)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def to_meshgrid(self, drop_singleton=True):
        """Expand x, y, z coordinates to meshgrid"""
        coords = [
            coord
            for coord in [self.x, self.y, self.z]
            if coord is not None and (coord.ndim > 0 or not drop_singleton)
        ]
        return np.meshgrid(*coords, indexing="ij")

    def to_dict(self):
        return {
            key: _x.to_numpy()
            for key, _x in zip(["x", "y", "z"], [self.x, self.y, self.z])
            if _x is not None and _x.ndim > 0
        }

    def keys(self):
        """Returns non-singleton axis names"""
        return [
            key
            for key, _x in zip(["x", "y", "z"], [self.x, self.y, self.z])
            if _x is not None and _x.ndim > 0
        ]

    def __repr__(self):
        return "Gridded data: " + repr(self.to_dict())


@xr.register_dataset_accessor("slice")
@xr.register_dataarray_accessor("slice")
class Slicer:
    """
    Slices into xarray using the same syntax as the previous Slice() class

    For back-compatability only. Recommended: instead of using ds.slice(),
    use the xarray select function (ds.sel()).
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, xlim=None, ylim=None, zlim=None, keys=None, **extra_kwargs):
        """Returns a slice of the original array"""
        x = self._obj.grid.x.to_numpy() if "x" in self._obj.coords else None
        y = self._obj.grid.y.to_numpy() if "y" in self._obj.coords else None
        z = self._obj.grid.z.to_numpy() if "z" in self._obj.coords else None

        xids, yids, zids = get_xids(
            x=x,
            y=y,
            z=z,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            return_slice=True,
            return_none=True,
        )
        kwargs = dict(x=xids, y=yids, z=zids)
        valid_indexers = {
            dim: idx
            for dim, idx in kwargs.items()
            if idx is not None and dim in self._obj.dims
        }

        if isinstance(self._obj, xr.DataArray):
            return self._obj.isel(**valid_indexers).sel(**extra_kwargs)
        elif keys:
            return self._obj.isel(**valid_indexers)[keys].sel(**extra_kwargs)
        else:
            return self._obj.isel(**valid_indexers).sel(**extra_kwargs)


@xr.register_dataset_accessor("imshow")
@xr.register_dataarray_accessor("imshow")
class XRImshow:
    """Add plotting function `imshow` for 2d arrays"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ax=None, cbar=True, figsize=None, **kwargs):
        if isinstance(self._obj, xr.Dataset):
            if len(self._obj.keys()) > 1:
                raise ValueError("Cannot plot type `Dataset` with more than 1 key")
            else:
                self._obj[next(iter(self._obj))].imshow(ax=ax, cbar=cbar, **kwargs)

        else:
            if self._obj.ndim != 2:
                raise AttributeError("imshow() requires 2D data")

            if ax is None:
                _, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(
                self._obj.T, extent=self._obj.grid.extent, origin="lower", **kwargs
            )
            axes = self._obj.grid.keys()
            ax.set_xlabel(labels[axes[0]])
            ax.set_ylabel(labels[axes[1]])
            if cbar:
                if self._obj.name in labels.keys():
                    label = labels[self._obj.name]
                else:
                    label = self._obj.name
                plt.colorbar(im, ax=ax, label=label)
            return im


@xr.register_dataset_accessor("pcolormesh")
@xr.register_dataarray_accessor("pcolormesh")
class XRPcolormesh:
    """
    Add plotting function `pcolormesh` for 2d arrays
    
    This is similar to `imshow` except useful for non-uniform grids. It 
    is also similar to the native `plot` function, but plots the transpose
    automatically and sets the apsect ratio to 1.0 by default.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ax=None, cbar=True, figsize=None, aspect=1, **kwargs):
        if isinstance(self._obj, xr.Dataset):
            if len(self._obj.keys()) > 1:
                raise ValueError("Cannot plot type `Dataset` with more than 1 key")
            else:
                self._obj[next(iter(self._obj))].pcolormesh(ax=ax, cbar=cbar, **kwargs)

        else:
            if self._obj.ndim != 2:
                raise AttributeError("pcolormesh() requires 2D data")
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)

            im = ax.pcolormesh(
                *self._obj.grid.xi, self._obj.T, **kwargs
            )
            axes = self._obj.grid.keys()
            ax.set_xlabel(labels[axes[0]])
            ax.set_ylabel(labels[axes[1]])
            ax.set_aspect(aspect)
            if cbar:
                if self._obj.name in labels.keys():
                    label = labels[self._obj.name]
                else:
                    label = self._obj.name
                plt.colorbar(im, ax=ax, label=label)
            return im


# ================= helper functions =================


def get_xids(
    x=None,
    y=None,
    z=None,
    xlim=None,
    ylim=None,
    zlim=None,
    return_none=False,
    return_slice=False,
):
    """
    Translates x, y, and z limits in the physical domain to indices
    based on axes x, y, z.

    Can be replaced by xarray's isel() function.

    Parameters
    ---------
    x, y, z : 1d array
    xlim, ylim, zlim : float or iterable (tuple, list, etc.)
        Physical locations to return the nearest index
    return_none : bool
        If True, populates output tuple with None if input is None. Default False.
    return_slice : bool
        If True, returns a tuple of slices instead a tuple of lists. Default False.

    Returns
    -------
    xid, yid, zid : list or tuple of lists
        Indices for the requested x, y, z, args in the order: x, y, z.
        If, for example, y and z are requested, then the returned tuple
        will have (yid, zid) lists. If only one value (float or int) is
        passed in for e.g. x, then an integer will be passed back in xid.
    """

    ret = ()

    # iterate through x, y, z, index matching for each term
    for s, s_ax in zip([xlim, ylim, zlim], [x, y, z]):
        if s is not None:
            if s_ax is None:
                raise AttributeError("Axis keyword not provided")

            if hasattr(s, "__iter__"):
                s = np.atleast_1d(s)  # fixes 0-D ndarry issues.
                xids = [np.argmin(np.abs(s_ax - xval)) for xval in s]
            else:
                xids = np.argmin(np.abs(s_ax - s))

            xids = np.squeeze(np.unique(xids))

            if return_slice and xids.ndim > 0:  # append slices to the return tuple
                ret = ret + (slice(np.min(xids), np.max(xids) + 1),)

            else:  # append index list to the return tuple
                ret = ret + (xids,)

        elif return_none:  # fill with None or slice(None)
            if return_slice:
                ret = ret + (slice(None),)

            else:
                ret = ret + (None,)

    if len(ret) == 1:
        return ret[0]  # don't return a length one tuple
    else:
        return ret


if __name__ == "__main__":
    pass
