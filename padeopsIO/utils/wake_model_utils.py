# Additional wake functions (some work-in-progress)
# Possible usage: from padeopsIO.wake_model_utils import *

import numpy as np
import os
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy import optimize

from .. import ActuatorDisk


import warnings

warnings.warn("wake_model_utils.py: This is being deprecated in a future release. ")

# ===================== WAKE FITTING =========================


def gaussian_wake(r, sigma, u0, r0):
    """
    Gaussian function defined as:
    u_w(r) = u0 * exp(-(r-r0)^2/(2*sigma^2))
    """

    return u0 * np.exp(-((r - r0) ** 2) / (2 * sigma**2))


def gaussian_wake_fit_con(y, wake, p0=0.5, u0=None, y0=None):
    """
    Curve fits a 1D gaussian wake with constrained (fixed) y0, u0

    Parameters
    ----------
    y : 1d array
        y-axis along hub plane
    wake : 1d array
        wake deficit u_inf-u(x=x',y,z=zhub) along the given y-axis
    p0 : float, optional
        initial guess for sigma(x). Defaults to 0.5.
    u0 : float, optional
        maximum wake deficit. If None, then this is computed as max(wake).
    y0 : float, optional
        location of maximum deficit. If None, then this is computed as y[np.argmax(wake)].
        If another value (e.g. the lateral turbine location) is given, then this will be
        forced into the curve fit.

    Returns
    -------
    sigma (float) : best fit standard deviation
    """

    if u0 is None:
        u0 = max(wake)

    if y0 is None:
        yid = np.argmax(wake)
        y0 = y[yid]

    def _gaussian_wake(y, sigma):
        return gaussian_wake(y, sigma, u0, y0)  # u0 * np.exp(-(y-y0)**2 / (2*sigma**2))

    ret = curve_fit(_gaussian_wake, y, wake, p0)
    return ret[0]


# ============ additional brute-force wake fitting functions ===============


def _compare_wm(x, xG, yG, uwake_ref, CT, yaw, order=2, phi_hub=0, mask_thresh=None):
    """
    xy-plane helper function
    """
    kw, sigma = x
    wake = ActuatorDisk.MITWake(CT, yaw, kw=kw, sigma=sigma, phi_hub=phi_hub)

    if mask_thresh is None:
        mask = 1
    else:
        mask = uwake_ref > mask_thresh

    uwake = wake.deficit(xG, yG)
    diff = np.ravel(abs(uwake - uwake_ref) * mask)

    res = np.linalg.norm(diff, ord=order)
    return res


def calibrate_wm(xax, yax, uwake_ref, ct, yaw, order=2, phi_hub=0, mask_thresh=None):
    """
    Calibrate kw, sigma0 to a reference slice in the xy-plane.

    Parameters
    ----------
    xax : (Nx, )
        Array of x-values for the truth data uwake_ref.
    yax : (Ny, )
        Array of y-values for the truth data uwake_ref.
    uwake_ref : (Nx, Ny)
        Array of \Delta u (wake deficit field) values.
    ct : float
        C_T' modified thrust coefficient for the leading turbine.
    yaw : float
        yaw (radians) of the leading turbine.
    order : int, optional
        Order for taking norms; Default 2.
    phi_hub : float, optional
        Hub height wind direction, with respect to +x axis (radians). Default 0
    mask_thresh : float, optional
        Threshold to cut off uwake_ref field. Default None (no mask)

    Returns
    -------
    res : OptimizeResult
        res.x contains the optimal (kw, sigma0)
    """
    x0 = [0.03, 0.25]
    xG, yG = np.meshgrid(xax, yax, indexing="ij")

    res = optimize.minimize(
        _compare_wm,
        x0,
        args=(xG, yG, uwake_ref, ct, yaw, order, phi_hub, mask_thresh),
        bounds=[(1e-3, 0.2), (1e-3, 0.5)],
    )
    #     print(res.x)
    return res


# Calibrate to 2D (or 3D) flow field
def _compare_wm2(
    x, xG, yG, zG, uwake_ref, CT, yaw, order=2, phi_hub=0, mask_thresh=None
):
    kw, sigma = x
    wake = ActuatorDisk.MITWake(CT, yaw, kw=kw, sigma=sigma)
    if mask_thresh is not None:
        mask = uwake_ref > mask_thresh
    else:
        mask = 1.0

    uwake = wake.deficit(xG, yG, z=zG)
    diff = abs(uwake - uwake_ref) * mask_thresh

    res = np.linalg.norm(np.ravel(diff), ord=order)
    return res


def calibrate_wm2(
    xax, yax, zax, uwake_ref, ct, yaw, order=2, phi_hub=0, mask_thresh=None, x0=None
):
    if x0 is None:
        x0 = [0.08, 0.25]
    xG, yG, zG = np.meshgrid(xax, yax, zax, indexing="ij")
    res = optimize.minimize(
        _compare_wm2,
        x0,
        args=(xG, yG, zG, uwake_ref, ct, yaw, order, phi_hub, mask_thresh),
        bounds=[(1e-3, 0.2), (1e-3, 0.5)],
    )
    return res


# Calibrate to reference turbine efficiency
def calibrate_wm_p(xt, yt, p_ref, ct, yaw, ct2=2.0, yaw2=0, sigma=0.25):
    x0 = [0.03]  # don't fit sigma0, assume = 0.25

    res = optimize.minimize(
        _compare_wm_p,
        x0,
        args=(xt, yt, p_ref, ct, yaw, ct2, yaw2 * np.pi / 180, sigma),
        bounds=[(1e-3, 0.2)],  # bounds on k_w
    )
    return res


def _compare_wm_p(x, xt, yt, p_ref, CT, yaw, ct2, yaw2, sigma):
    # leading turbine
    wake = ActuatorDisk.MITWake(CT, yaw, kw=x, sigma=sigma)
    rews = wake.REWS(xt, yt)

    a, _, _ = ActuatorDisk.calculate_induction(ct2, yaw2)
    cp = ct2 * ((1 - a) * np.cos(yaw2)) ** 3
    eta2 = cp * rews**3

    return abs(p_ref - eta2)


def get_uwake(CT, yaw, kw, sigma, xax, yax, zax=0, phi_hub=0):
    """
    Returns the wake field computed with the gaussian wake model
    """
    xG, yG, zG = np.meshgrid(xax, yax, zax, indexing="ij")
    wake = ActuatorDisk.MITWake(CT, yaw, kw=kw, sigma=sigma, phi_hub=phi_hub)
    uwake = wake.deficit(xG, yG, z=zG)
    return np.squeeze(uwake)


def CTprime_CT(CT_prime):
    a = CT_prime / (4 + CT_prime)
    return 4 * a * (1 - a)


def CT_sigma(CT):
    beta = 0.5 * (1 + np.sqrt(1 - CT)) / np.sqrt(1 - CT)
    return 0.2 * np.sqrt(beta)


def _compare_con(x, xG, yG, uwake_ref, CT, yaw, order=2, phi_hub=0, mask_thresh=None):
    """
    Helper function for constrained optimization
    """
    kw = x
    ct = CTprime_CT(CT)  # CT is C_T'
    sigma = CT_sigma(ct)
    wake = ActuatorDisk.MITWake(CT, yaw, kw=kw, sigma=sigma, phi_hub=phi_hub)

    if mask_thresh is None:
        mask = 1
    else:
        mask = uwake_ref > mask_thresh

    uwake = wake.deficit(xG, yG)
    diff = (uwake - uwake_ref) * mask

    res = np.linalg.norm(np.ravel(diff), ord=order)
    return res


def calibrate_con(case, u_hub, xlim=[4.0, 5.0], order=1, phi_hub=0, mask_thresh=None):
    """Constrained calibration for sigma"""

    sl = case.slice(budget_terms=["ubar"], xlim=xlim, zlim=0)
    adm = case.turbineArray.turbines[0]
    uwake_ref = (u_hub - sl["ubar"]) / u_hub

    x0 = 0.02
    xG, yG = np.meshgrid(sl["x"], sl["y"], indexing="ij")

    res = optimize.minimize(
        _compare_con,
        x0,
        args=(
            xG,
            yG,
            uwake_ref,
            adm.ct,
            np.deg2rad(adm.yaw),
            order,
            phi_hub,
            mask_thresh,
        ),
        bounds=[(1e-3, 0.2)],
    )
    CT = CTprime_CT(adm.ct)
    sigma = CT_sigma(CT)
    res.x = np.array([res.x[0], sigma])
    return res
