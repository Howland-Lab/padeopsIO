"""
Helper functions for PadeopsIO

Kirby Heck
2024 July 15
"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import warnings

from . import io_utils


def get_logfiles(path, search_str="*.o[0-9]*", id=-1): 
    """
    Searches for all logfiles formatted "*.o[0-9]" (Stampede3 format)
    and returns the entire list if `id` is None, otherwise returns 
    the specific `id` requested. 

    Parameters
    ----------
    path : path-like
        Directory to search for logfiles
    search_str : str
        Pattern to attempt to match
    id : int
        If multiple logfiles exist, selects this index of the list
    """
    logfiles = list(path.glob(search_str))

    if len(logfiles) == 0: 
        warnings.warn("No logfiles found, returning")
        return []
    elif id is None: 
        return logfiles
    else: 
        return logfiles[id]


def get_ustar(self=None, logfile=None, search_str="*.o[0-9]*", crop_budget=True, average=True):
    """
    Gleans ustar from the logfile.

    Parameters
    ----------
    self : BudgetIO object, optional
        if None, then logfile must be a full path (and not a filename)
    logfile : path-like, optional
        Path to logfile. 
    search_str : str, optional
        String to match. Default: searches for all files ending in '.o[0-9]*'.
    crop_budget : bool, optional
        Crops time axis to budgets. Defaults to True.
    average : bool, optional
        Time averages. Defaults to True.
    """
    if self is not None: 
        logfile = get_logfiles(self.dir_name, search_str=search_str, id=-1)
    elif logfile is not None:
        logfile = Path(logfile)
    else: 
        raise ValueError("get_ustar(): Requires either BudgetIO object `self` or path-like `logfile`.")

    # match the last one and read ustar... could fix this later
    ret = io_utils.query_logfile(logfile, search_terms=["u_star", "TIDX", "Time"])

    nml = self.input_nml
    if crop_budget and nml["budget_time_avg"]["do_budgets"]:
        try:
            time_budget_st = nml["budget_time_avg"]["time_budget_start"]
            filt = ret["Time"] > time_budget_st
            return np.mean(ret["u_star"][filt])
        except KeyError as e:
            raise  # fix this later

    if average:
        return np.mean(ret["u_star"])
    else:
        return ret["u_star"]


def get_uhub(self, z_hub=0, use_fields=False, **slice_kwargs):
    """Compute the hub height velocity"""
    if use_fields:
        s = self.xy_avg(field_terms=["u", "v"], **slice_kwargs)
        U = np.sqrt(s["u"] ** 2 + s["v"] ** 2)
    else:
        s = self.xy_avg(budget_terms=["ubar", "vbar"], **slice_kwargs)
        U = np.sqrt(s["ubar"] ** 2 + s["vbar"] ** 2)

    Uinf = np.interp(z_hub, s.grid.z, U)
    return Uinf


def get_phihub(self, z_hub=0, return_degrees=False, use_fields=False, **slice_kwargs):
    """Interpolate hub height wind direction (radians)."""
    if use_fields:
        ret = self.xy_avg(field_terms=["u", "v"], **slice_kwargs)
        phi = np.arctan2(ret["v"], ret["u"])
    else:
        ret = self.xy_avg(budget_terms=["ubar", "vbar"], **slice_kwargs)
        phi = np.arctan2(ret["vbar"], ret["ubar"])

    if return_degrees:
        return np.rad2deg(np.interp(z_hub, self.grid.z, phi))
    else:
        return np.interp(z_hub, self.grid.z, phi)


def get_timekey(self, budget=False):
    """
    Returns a dictionary matching time keys [TIDX in PadeOps] to non-dimensional times.

    Arguments
    ----------
    self : BudgetIO object
    budget : bool
        If true, matches budget times from BudgetIO.unique_budget_tidx(). Default false.

    Returns
    -------
    dict
        matching {TIDX: time} dictionary
    """
    tidxs = self.unique_tidx()
    times = self.unique_times()

    timekey = {tidx: time for tidx, time in zip(tidxs, times)}

    if budget:
        keys = self.unique_budget_tidx(return_last=False)
        return {key: timekey[key] for key in keys}
    else:
        return timekey


def get_time_ax(self, return_tidx=False, missing_init_ok=True):
    """
    Interpolates a time axis between Time IDs

    Parameters
    ----------
    self : BudgetIO object
    return_tidx : bool (optional)
        If True, returns tidx, time axes. Default False
    missing_init_ok : bool (optional)
        If True, then info files do not need to be written on initialization,
        uses a workaround to find the restarts. Default True.

    Returns
    -------
    tidx, time
        if `return_tidx` is True
    time
        if `return_tidx` is False
    """
    times = self.unique_times()
    tidx = self.unique_tidx()
    if missing_init_ok and io_utils.key_search_r(self.input_nml, "userestartfile"):
        first_tid = min(
            io_utils.key_search_r(self.input_nml, "restartfile_tid"), tidx[0]
        )
    else:
        first_tid = tidx[0]

    new_tidx = np.arange(first_tid, np.max(tidx) + 1, 1)
    extrapolator = interp1d(tidx, times, kind="linear", fill_value="extrapolate")
    new_time = extrapolator(new_tidx)

    if return_tidx:
        return new_tidx, new_time
    else:
        return new_time


def get_dt(self):
    """Computes a mean time step dt for a simulation"""
    time = get_time_ax(self)
    return np.mean(np.diff(time))


def moving_avg(signal, kernel_size=3):
    """Smoothed 1D signal, default kernel size = 3"""
    kernel = np.ones(kernel_size)
    normfact = np.convolve(np.zeros_like(signal) + 1, kernel, mode="same")
    smoothed = np.convolve(signal, kernel, mode="same")
    return smoothed / normfact


def window_agg(x, length=1, func=np.mean, original_size=False):
    """
    Aggregates a signal `x` into groups of length `length` and applies a function
    to each group.

    Example:
    window_agg(x, length=5)
        Returns the signal x, averaged over groups of 5.

    Parameters
    ----------
    x : ndarray or list
        Signal or list of signals to aggregate
    length : int
        Length of aggregation window
    func : function
        Aggregation function, default np.mean()
    original_size : bool
        If True, repeats the result `length` times to arrive at a vector the same length
        as the original signal. Default False
    """
    if length == 1:
        return x  # no averaging needed
    elif length < 1:
        raise ValueError("`length` must be an integer greater than zero.")

    def _helper(_signal):
        N = len(_signal)
        new_N = (N // length) * length

        arr = np.reshape(_signal[:new_N], (-1, length))  # reshape
        ret = func(arr, axis=1)  # apply function over the axis which was just created

        if original_size:
            return np.pad(np.repeat(ret, length), pad_width=(0, N - new_N), mode="edge")
        else:
            return ret

    if isinstance(x, list):
        res = []
        for _signal in x:
            res.append(_helper(_signal))

    else:
        res = _helper(x)

    return res
