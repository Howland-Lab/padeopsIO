"""
Budget class for computing and aggregating budget terms. 

Kirby Heck
2024 July 24
"""

import numpy as np
import xarray as xr
from .utils.io_utils import key_search_r
from .budgetIO import BudgetIO
from .budget_addons import *
from .gridslice import GridDataset


class Budget(GridDataset):
    """
    Computes "offline" budgets and links to a BudgetIO object.
    """

    __slots__ = (
        "src",
        "full_arrays",
        "rans_x",
        "rans_y",
        "rans_z",
        "momentumLES_x",
        "momentumLES_y",
        "momentumLES_z",
        "vorticity_x",
        "vorticity_y",
        "vorticity_z",
    )

    def __init__(self, src):
        """
        Initialize a Budget object

        Parameters
        ----------
        src : Any
            If type(src) is a BudgetIO file, links the Budget class to the IO object
            If type(src) is a dictionary, only associates the current fields and axes
                For dictionary, expects keys ['x', 'y', 'z', <terms>, ...]
        """

        if isinstance(src, BudgetIO):
            super().__init__(src.budget)
            self.src = src
            self.attrs["Ro"] = src.Ro
            self.attrs["lat"] = src.lat
            self.attrs["galpha"] = src.galpha
            self.attrs["Fr"] = src.Fr
            self.attrs["is_stratified"] = (
                key_search_r(src.input_nml, "isstratified") or False
            )
            self.attrs["theta0"] = key_search_r(src.input_nml, "tref")
            self.full_arrays = src.budget  # keep "Full" BudgetIO domain size

        elif isinstance(src, Budget):
            super().__init__(src)

        else:
            raise TypeError(f"`src` must be type BudgetIO, not {type(src)}")

    def set_xlim(self, xlim=None, ylim=None, zlim=None):
        """
        Updates the grid and slices into main fields

        If no keyword arguments are given, resets the budget object
        grid to the source (LES) dimensions.
        """
        newslice = self.full_arrays.slice(xlim=xlim, ylim=ylim, zlim=zlim)
        attrs = self.attrs
        src = self.src
        super().__init__(newslice)  # create a new parent object
        self.attrs = attrs  # restore attributes
        self.src = src  # restore source

    def read_budgets(self, budget_terms):
        """Not sure why _read_budgets() was made hidden."""
        self._read_budgets(budget_terms)

    def _read_budgets(self, budget_terms):
        """
        Reads budgets using BudgetIO, if linked.
        """
        if isinstance(self.src, BudgetIO):
            ds = self.src.slice(
                budget_terms=budget_terms,
                xlim=self.grid.x.to_numpy(),
                ylim=self.grid.y.to_numpy(),
                zlim=self.grid.z.to_numpy(),
            )
            for key in ds.data_vars.keys():
                self[key] = ds[key]
            self.full_arrays = self.src.budget

        else:
            raise AttributeError(
                "_read_budgets(): Reading budgets requires a linked BudgetIO object."
            )

    # ================= Add budgets to interface with here ===================

    def init_momentumLES_x(self, **kwargs):
        self.momentumLES_x = LESMomentum_x(self, **kwargs)
        return self.momentumLES_x

    def init_momentumLES_y(self, **kwargs):
        self.momentumLES_y = LESMomentum_y(self, **kwargs)
        return self.momentumLES_y

    def init_momentumLES_z(self, **kwargs):
        self.momentumLES_z = LESMomentum_z(self, **kwargs)
        return self.momentumLES_z

    def init_rans_x(
        self,
        base_agg=0,
        Ro=None,
        lat=None,
        galpha=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if lat is None:
            lat = self.lat
        if galpha is None:
            galpha = self.galpha
        self.rans_x = RANS_x(
            self,
            base_agg=base_agg,
            Ro=Ro,
            lat=lat,
            galpha=galpha,
        )
        return self.rans_x

    def init_rans_y(
        self,
        base_agg=0,
        Ro=None,
        lat=None,
        galpha=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if lat is None:
            lat = self.lat
        if galpha is None:
            galpha = self.galpha
        self.rans_y = RANS_y(
            self,
            base_agg=base_agg,
            Ro=Ro,
            lat=lat,
            galpha=galpha,
        )
        return self.rans_y

    def init_rans_z(
        self,
        base_agg=0,
        Ro=None,
        lat=None,
        galpha=None,
        Fr=None,
        theta0=None,
        is_stratified=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        if galpha is None:
            galpha = self.galpha
        if Fr is None:
            Fr = self.Fr
        if theta0 is None:
            theta0 = self.theta0
        if is_stratified is None:
            is_stratified = self.is_stratified
        self.rans_z = RANS_z(
            self,
            base_agg=base_agg,
            Ro=Ro,
            lat=lat,
            galpha=galpha,
            Fr=Fr,
            theta0=theta0,
            is_stratified=is_stratified,
        )
        return self.rans_z

    def init_deficit_x(
        self,
        bkgd_budget,
        base_agg=0,
        Ro=None,
        Fr=None,
        lat=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        self.deficit_x = BudgetDeficit_x(
            self,
            bkgd_budget,
            base_agg=base_agg,
            Ro=Ro,
            Fr=Fr,
            lat=lat,
        )
        return self.deficit_x

    def init_deficit_y(self, bkgd_budget, base_agg=0, Ro=None, Fr=None, lat=None):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        self.deficit_y = BudgetDeficit_y(
            self,
            bkgd_budget,
            base_agg=base_agg,
            Ro=Ro,
            Fr=Fr,
            lat=lat,
        )
        return self.deficit_y

    def init_vorticity_x(
        self, base_agg=0, Ro=None, lat=None, fplane=True, Fr=None, theta0=None
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        if theta0 is None and isinstance(self.src, BudgetIO):
            theta0 = key_search_r(self.src.input_nml, "tref")
        self.vorticity_x = BudgetVorticity_x(
            self, base_agg=base_agg, Ro=Ro, lat=lat, fplane=fplane, Fr=Fr, theta0=theta0
        )
        return self.vorticity_x


    def init_vorticity_y(
        self, base_agg=0, Ro=None, lat=None, fplane=True, Fr=None, theta0=None
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        if theta0 is None and isinstance(self.src, BudgetIO):
            theta0 = key_search_r(self.src.input_nml, "tref")
        self.vorticity_y = BudgetVorticity_y(
            self, base_agg=base_agg, Ro=Ro, lat=lat, fplane=fplane, Fr=Fr, theta0=theta0
        )
        return self.vorticity_y


    def init_vorticity_z(
        self, base_agg=0, Ro=None, lat=None, fplane=True, Fr=None, theta0=None
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        if theta0 is None and isinstance(self.src, BudgetIO):
            theta0 = key_search_r(self.src.input_nml, "tref")
        self.vorticity_z = BudgetVorticity_z(
            self, base_agg=base_agg, Ro=Ro, lat=lat, fplane=fplane, Fr=Fr, theta0=theta0
        )
        return self.vorticity_z


if __name__ == "__main__":
    pass  # TODO add unit tests
