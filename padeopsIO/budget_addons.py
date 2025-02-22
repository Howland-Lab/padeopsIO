"""
Budget term computations; return the most disaggregated form

Kirby Heck
2024 July 24
"""

import numpy as np
from abc import ABC

from .utils import math_utils as math
from .utils import fluids_utils as fluids
from .gridslice import GridDataset


# =============== NewBudget interface ================


class NewBudget(GridDataset, ABC):
    """
    Informal interface for new budget classes to add (e.g. RANS).

    Aggregation is primarly defined based on how repeated indices are summed (or not)
    in the budget computation. For a term like the reynolds stress divergence,
        d<ui'uj'>/dxj,
    the base level of aggregation (level 0) is to sum over j. If we want one level lower
    of diaggregation (level 1), the j-indices are not summed.
    """

    __slots__ = ("budget", "base_terms")
    req_keys = []  # required keys go here (e.g. 'ubar', etc...)
    opt_keys = []  # optional keys (e.g. 'xAD', etc... )

    # ====== the following functions should not need to change (much) =======

    def __init__(self, budget, base_agg=0):
        """Blueprint for budgets"""
        super().__init__(coords=budget.coords)
        self.budget = budget
        self.base_terms = None  # these are computed in self.compute()
        self.attrs["base_agg"] = (
            base_agg  # base level of aggregation, 0 is the most aggregated
        )

    def clear(self, keys=None):
        """Clear all terms"""
        for var in list(keys or self.data_vars.keys()):
            try:
                del self[var]  # delete these keys
            except KeyError as e:
                pass  # skip these

    def pop(self, key):
        """Pop a key"""
        var = self[key]
        del self[key]
        return var

    def compute(self, aggregate=None, custom=False, **kwargs):
        """Compute disaggregated budget terms"""
        # make sure the terms exist to compute budgets
        self._check_terms()

        # carry out budget computation
        self._compute_budget()

        # aggregate terms and save them in the dictionary
        self.aggregate(aggregate=aggregate, custom=custom, **kwargs)

    def aggregate(self, aggregate=None, custom=False, **kwargs):
        """Aggregate similar budget terms"""

        self.clear()  # clear previous aggregation
        if aggregate is None:  # default aggregation
            aggregate = self.base_agg

        if not custom:
            try:
                ret = math.new_aggregation(
                    self.base_terms,
                    base_agg=aggregate,
                    **kwargs,
                )
            except ValueError as e:
                # catch errors in new_aggregation()
                raise ValueError(
                    "aggregate(): cannot take base_agg < 0. Perhaps try passing custom=True"
                )

            for key in ret.data_vars.keys():
                self[key] = ret[key]

        else:  # add a custom aggregation on top of base level
            self._aggregate_custom(aggregate, **kwargs)

    def _check_terms(self, budget=None):
        """
        Ensure all the required terms exist before computing budgets

        Parameters
        ----------
        budget : budget.Budget object
            Checks terms in self.budget by default
        """

        if budget is None:
            budget = self.budget
        # first check required keys:
        missing_keys = [key for key in self.req_keys if key not in budget.keys()]

        if len(missing_keys) > 0:  # try to load missing keys
            try:
                budget._read_budgets(missing_keys)

            except AttributeError as e:
                print(
                    "NewBudget._check_terms(): missing",
                    missing_keys,
                    ", no source files associated.",
                )
                raise e

            except KeyError as e:
                raise e  # budgets in BudgetIO.read_budgets() do not exist

        # then check optional keys:
        missing_keys_opt = [key for key in self.opt_keys if key not in budget.keys()]

        if len(missing_keys_opt) > 0:  # try also to load optional keys:
            try:
                budget.read_budgets(missing_keys_opt)

            except AttributeError as e:
                print(
                    "NewBudget._check_terms(): missing optional keys",
                    missing_keys_opt,
                    ", no source files associated.",
                )
                # but do not raise these errors

            except KeyError as e:
                still_missing = [
                    key for key in self.opt_keys if key not in budget.keys()
                ]
                print(
                    "NewBudget._check_terms(): missing optional keys: ", still_missing
                )

    # the following functions will need to be overwritten

    def _aggregate_custom(self, level):
        """Custom aggregation for this budget"""
        raise Exception("_aggregate_custom(): No function definition. ")

    def _compute_budget(self):
        """Fill in budget computations here"""
        raise Exception("_compute_budget(): No function definition. ")


# ==================== LES Momentum budgets =========================


class LESMomentum(NewBudget):
    """
    Class for LES momentum budgets ("read directly from PadeOps")
    """

    __slots__ = ()

    def _compute_budget(self):
        """
        Assembles LES momentum budgets directly from PadeOps.
        """
        terms = GridDataset(coords=self.coords)
        for key in self.req_keys + self.opt_keys:
            try:
                terms[key] = self.budget[key]
            except KeyError as e:
                print("_compute_budget(): could not find term", key)

        # compute residual
        fluids.compute_residual(terms, in_place=True)

        self.base_terms = terms


class LESMomentum_x(LESMomentum):
    """
    Reads LES x-momentum budgets.
    """

    __slots__ = ()
    req_keys = ["DuDt", "dpdx", "xSGS"]
    opt_keys = ["xAD", "xCor", "xGeo"]

    def _aggregate_custom(self, aggregate, **kwargs):
        """
        Combines the coriolis and geostrophic terms. This is the "-1" level of aggregation
        (combining non-tensor terms).

        Parameters
        ----------
        aggregate : Any
            Dictionary {'coriolis': -1} or int (-1) to aggregate Coriolis terms
        """
        # may throw KeyError
        if aggregate == -1:
            kwargs = {"coriolis": -1}
            aggregate = 0

        self.aggregate(aggregate)

        if "coriolis" in kwargs and kwargs["coriolis"] == -1:
            self["coriolis"] = self["xCor"] + self["xGeo"]
            del self["xCor"], self["xGeo"]

        self["residual"] = self.pop("residual")  # move the residual to the end


class LESMomentum_y(LESMomentum):
    """
    Reads LES y-momentum budgets.
    """

    __slots__ = ()
    req_keys = ["DvDt", "dpdy", "ySGS"]
    opt_keys = ["yAD", "yCor", "yGeo"]

    def _aggregate_custom(self, aggregate, **kwargs):
        """
        Combines the coriolis and geostrophic terms. This is the "-1" level of aggregation
        (combining non-tensor terms).

        Parameters
        ----------
        level : int
            Base level of aggregation
        kwargs : dict
        """
        # may throw KeyError
        if aggregate == -1:
            kwargs = {"coriolis": -1}
            aggregate = 0

        self.aggregate(aggregate)

        if "coriolis" in kwargs and kwargs["coriolis"] == -1:
            self["coriolis"] = self["yCor"] + self["yGeo"]
            del self["yCor"], self["yGeo"]

        self["residual"] = self.pop("residual")  # move the residual to the end


class LESMomentum_z(LESMomentum):
    """
    Reads LES z-momentum budgets.
    """

    __slots__ = ()
    req_keys = ["DwDt", "dpdz", "zSGS"]
    opt_keys = ["zBuoy"]


# =========================== RANS Budgets ===============================


class RANSBudget(NewBudget):
    """
    Base class for filtered RANS budgets.
    """

    __slots__ = ()
    req_keys = [
        fluids.uvw_keys + fluids.rs_keys[0] + fluids.tau_keys[0] + ["pbar"],
        fluids.uvw_keys + fluids.rs_keys[1] + fluids.tau_keys[1] + ["pbar"],
        fluids.uvw_keys + fluids.rs_keys[2] + fluids.tau_keys[2] + ["pbar"],
    ]  # required keys in x, y, z
    opt_keys = [["xAD", "dpdx"], ["yAD", "dpdy"], ["Tbar", "dpdz"]]  # optional keys

    def __init__(
        self,
        budget,
        base_agg=0,
        Ro=None,
        Fr=None,
        lat=None,
        galpha=0,
        is_stratified=True,
        theta0=None,
        # TODO: what if we don't want to use Coriolis?
    ):
        """
        Initialize non-dimensional RANS budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        Fr : float
            Froude number, defined U/\sqrt{g L}
        lat : float
            Latitude, in degrees
        galpha : float
            Geostrophic wind direction, in degrees
        is_stratified : bool
            Use stratification if True. Default True.
        theta0 : float
            Reference potential temperature (K)
        """
        super().__init__(budget, base_agg)
        self.attrs["Ro"] = Ro
        self.attrs["Fr"] = Fr
        self.attrs["lat"] = lat * np.pi / 180 if lat else None
        self.attrs["galpha"] = galpha * np.pi / 180 if galpha else 0
        self.attrs["is_stratified"] = is_stratified
        self.attrs["theta0"] = theta0
        self.attrs["direction"] = None

    def _compute_budget(self):
        """
        Computes RANS momentum budgets in x.
        """
        self.base_terms = fluids.compute_RANS(
            self.budget,
            self.direction,
            Ro=self.Ro,
            lat=self.lat,
            galpha=self.galpha,
            is_stratified=self.is_stratified,
            Fr=self.Fr,
            theta0=self.theta0,
        )
        fluids.compute_residual(self.base_terms, in_place=True)

    def _aggregate_custom(self, aggregate, **kwargs):
        """
        For RANS Budgets, custom aggregation allows for the following:
            Passing `totaladv=-1` aggregates mean advection and Reynolds stresses
            Passing `coriolis=-1` aggregates Coriolis + Geostrophic
        """
        if aggregate == -1:
            kwargs = {"totaladv": -1, "coriolis": -1}
            aggregate = 0

        self.aggregate(aggregate, **kwargs)
        if "totaladv" in kwargs.keys() and kwargs["totaladv"] == -1:
            self["totaladv"] = self["adv"] + self["rs"]
            del self["rs"], self["adv"]

        if "coriolis" in kwargs.keys() and kwargs["coriolis"] == -1:
            self["coriolis"] = self["cor"] + self["geo"]
            del self["geo"], self["cor"]

        self["residual"] = self.pop("residual")  # move the residual to the end


class RANS_x(RANSBudget):
    """
    Computes the RANS budgets in the y-direction.
    """

    __slots__ = ()
    req_keys = RANSBudget.req_keys[0]
    opt_keys = RANSBudget.opt_keys[0]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 0  # x-direction


class RANS_y(RANSBudget):
    """
    Computes the RANS budgets in the y-direction.
    """

    __slots__ = ()
    req_keys = RANSBudget.req_keys[1]
    opt_keys = RANSBudget.opt_keys[1]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 1  # y-direction


class RANS_z(RANSBudget):
    """
    Computes the RANS budgets in the y-direction.
    """

    __slots__ = ()
    req_keys = RANSBudget.req_keys[2]
    opt_keys = RANSBudget.opt_keys[2]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 2  # z-direction


# =========================== RANS Deficit Budgets ===============================


class BudgetDeficit(NewBudget):
    """
    Base class for filtered RANS deficit budgets.
    """

    __slots__ = ()
    req_keys = RANSBudget.req_keys
    opt_keys = RANSBudget.opt_keys

    def __init__(
        self,
        budget,
        bkgd_budget,
        base_agg=0,
        Ro=None,
        Fr=None,
        lat=None,
    ):
        """
        Initialize non-dimensional RANS-deficit budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        bkdg_budget : budget.Budget object
            Budget object for the background flow (precursor sim.)
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        Fr : float
            Froude number, defined U/\sqrt{g L}
        lat : float
            Latitude, in degrees
        """
        super().__init__(budget, base_agg)
        self.attrs["bkgd"] = bkgd_budget
        self.attrs["Ro"] = Ro
        self.attrs["Fr"] = Fr
        self.attrs["lat"] = lat * np.pi / 180
        self.attrs["direction"] = None  # overwrite this in sub-classes

    def _compute_budget(self):
        """
        Computes RANS momentum budgets in x.
        """
        self.base_terms = fluids.deficit_budget(
            self.budget,
            self.bkgd,
            self.direction,
            Ro=self.Ro,
            lat=self.lat,
        )
        fluids.compute_residual(self.base_terms, in_place=True)

    def _check_terms(self):
        """Check terms for background and primary budget objects"""
        super()._check_terms()
        super()._check_terms(budget=self.bkgd)

    def _aggregate_custom(self, level, **kwargs):
        """Custom aggregation for level -1 (combined base terms)"""
        if level == -1:
            kwargs = {"rs": -1}
            level = 0

        if "rs" in kwargs.keys():
            self.aggregate(level, rsfull=kwargs["rs"], rsbkgd=kwargs["rs"], **kwargs)
            if kwargs["rs"] <= 0:
                # combine advection with reynolds stresses
                self["rs"] = self["rsfull"] + self["rsbkgd"]
                del self["rsfull"], self["rsbkgd"]
            elif kwargs["rs"] == 1:
                for k in range(1, 4):
                    self[f"rs_{k}"] = self[f"rsfull_{k}"] + self[f"rsbkgd_{k}"]
                    del self[f"rsfull_{k}"], self[f"rsbkgd_{k}"]
        else:
            self.aggregate(level, **kwargs)

        self["residual"] = self.pop("residual")  # move the residual to the end


class BudgetDeficit_x(BudgetDeficit):
    """
    Computes the RANS deficit budgets in the x-direction.
    """

    __slots__ = ()
    req_keys = BudgetDeficit.req_keys[0]
    opt_keys = BudgetDeficit.opt_keys[0]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 0  # x-direction


class BudgetDeficit_y(BudgetDeficit):
    """
    Computes the RANS deficit budgets in the y-direction.
    """

    __slots__ = ()
    req_keys = BudgetDeficit.req_keys[1]
    opt_keys = BudgetDeficit.opt_keys[1]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 1  # y-direction


# ======================= Vorticity budgets ==========================


class BudgetVorticity(NewBudget):
    """
    Base class for filtered vorticity budgets.
    """

    __slots__ = ()
    req_keys = fluids.uvw_keys + fluids.rs_keys_list + fluids.tau_keys_list + ["Tbar"]
    opt_keys = ["xAD", "yAD"]

    def __init__(
        self, budget, base_agg=0, Ro=None, lat=None, fplane=True, Fr=None, theta0=None
    ):
        """
        Initialize non-dimensional vorticity budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        lat : float
            Latitude, in degrees
        fplane : bool
            Use f-plane approx. Default True
        Fr : float
            Froude number, defined U/\sqrt{g L}
        theta0 : float
            Reference potential temperature (K)
        """
        super().__init__(budget, base_agg)
        self.attrs["Ro"] = Ro
        self.attrs["Fr"] = Fr
        self.attrs["lat"] = lat * np.pi / 180
        self.attrs["fplane"] = fplane
        self.attrs["theta0"] = theta0
        self.attrs["direction"] = None  # overwrite this in sub-classes

    def _compute_budget(self):
        """
        Computes vorticity budgets
        """
        self.base_terms = fluids.compute_vort_budget(
            self.budget,
            self.direction,
            Ro=self.Ro,
            lat=self.lat,
            fplane=self.fplane,
            Fr=self.Fr,
            theta0=self.theta0,
        )
        fluids.compute_residual(self.base_terms, in_place=True)

    def _aggregate_custom(self, level, **kwargs):
        """Define custom aggregation here"""
        if level == -1:
            kwargs = {"turb_sgs": -1}
            level = 0

        self.aggregate(level, **kwargs)
        if "turb_sgs" in kwargs.keys() and kwargs["turb_sgs"] == -1:
            self["turb_sgs"] = self["sgs"] + self["rs"]
            del self["rs"], self["sgs"]

        self["residual"] = self.pop("residual")  # move the residual to the end


class BudgetVorticity_x(BudgetVorticity):
    """
    Vorticity budgets in x
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 0  # x-direction


class BudgetVorticity_y(BudgetVorticity):
    """
    Vorticity budgets in y
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 1  # y-direction


class BudgetVorticity_z(BudgetVorticity):
    """
    Vorticity budgets in z
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.attrs["direction"] = 2  # z-direction


# ====================== MKE Budget ========================


class BudgetMKE(NewBudget):
    """
    MKE budget, RANS formulation
    """

    __slots__ = ()

    req_keys = (
        fluids.uvw_keys + fluids.rs_keys_list + fluids.tau_keys_list + ["pbar", "Tbar"]
    )
    opt_keys = ["xAD", "yAD", "dpdx", "dpdy", "dpdz"]

    def __init__(self, budget, base_agg=0, Fr=None, theta0=None):
        """
        Initialize non-dimensional MKE budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        lat : float
            Latitude, in degrees
        fplane : bool
            Use f-plane approx. Default True
        Fr : float
            Froude number, defined U/\sqrt{g L}
        theta0 : float
            Reference potential temperature (K)
        """
        super().__init__(budget, base_agg)
        self.attrs["Fr"] = Fr
        self.attrs["theta0"] = theta0

    def _compute_budget(self):
        """
        Computes MKE budgets
        """
        self.base_terms = fluids.compute_mke_budget(
            self.budget,
            Fr=self.Fr,
            theta0=self.theta0,
        )
        fluids.compute_residual(self.base_terms, in_place=True)
