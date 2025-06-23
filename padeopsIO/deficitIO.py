"""
Read deficit budgets.

Copied from Kerry Klemmer's branch
2024 December 16
"""

import numpy as np
import os
import re
import warnings
from scipy.io import loadmat

from . import budgetIO as pio
from . import deficitkey as deficitkey  # defines key pairing
from .utils.ksk_utils import *
from .utils.io_utils import key_search_r


class DeficitIO(pio.BudgetIO):

    key = deficitkey.get_key()
    key_xy = deficitkey.get_key_xy()

    """
    Class that extends BudgetIO to read deficit budgets 
    """

    def __init__(self, dir_name, **kwargs):
        """
        Calls the constructor of BudgetIO
        """

        super().__init__(dir_name, **kwargs)

        if self.verbose:
            print("Initialized DeficitIO object")

    def existing_budgets(self):
        """
        Checks file names for which budgets were output.
        """
        if self.associate_padeops:
            # capturing *_budget(\d+)* in filenames
            budget_list = self.unique_tidx(
                search_str=rf"Run{self.runid:02d}.*_deficit_budget(\d+).*"
            )

        else:
            if self.associate_npz:
                filename = self.dirname / self.fname_budgets.format("npz")
                with np.load(filename) as npz:
                    t_list = npz.files  # load all the budget filenames
            if self.associate_mat:
                filename = self.dirname / self.fname_budgets.format("mat")
                ret = loadmat(filename)
                t_list = [
                    key for key in ret if key[0] != "_"
                ]  # ignore `__header__`, etc.

            budget_list = [self.key[t][0] for t in t_list]

        if len(budget_list) == 0:
            warnings.warn("existing_budgets(): No associated budget files found. ")

        return np.unique(budget_list)

    def existing_terms(self, budget=None):
        """
        Checks file names for a particular budget and returns a list of all the existing terms.

        Arguments
        ---------
        budget (integer) : optional, default None. If provided, searches a particular budget for existing terms.
            Otherwise, will search for all existing terms. `budget` can also be a list of integers.
            Budget 0: mean statistics
            Budget 1: momentum
            Budget 2: MKE
            Budget 3: TKE
            Budget 4: Reynolds stress

        Returns
        -------
        t_list (list) : list of tuples of budgets found

        """

        t_list = []

        budget4_comp_dict = {11: 0, 22: 10, 33: 20, 13: 30, 23: 40}

        # if no budget is given, look through all saved budgets
        if budget is None:
            budget_list = self.existing_budgets()

        else:
            # convert to list if integer is given
            if hasattr(budget, "__iter__"):
                budget_list = [budget]
            else:
                budget_list = budget

        if self.associate_padeops:
            # find budgets by name matching with PadeOps output conventions
            tup_list = []
            # loop through budgets
            for b in budget_list:
                search_str = rf"Run{self.runid:02d}_deficit_budget{b:01d}_term(\d+).*"
                terms = self.unique_tidx(search_str=search_str)
                tup_list += [((b, term)) for term in terms]  # these are all tuples

                # reynolds stress budgets
                if b == 4:
                    for component in budget4_comp_dict:
                        search_str = rf"Run{self.runid:02d}_deficit_budget{b:01d}_{component:01d}_term(\d+).*"
                        terms = self.unique_tidx(search_str=search_str)
                        tup_list += [
                            ((b, term)) for term in terms
                        ]  # these are all tuples

            # convert tuples to keys
            t_list = [self.key.inverse[key][0] for key in tup_list]

        else:
            # find budgets matching .npz convention in write_npz()
            if self.associate_npz:
                filename = self.dirname / self.fname_budgets.format("npz")
                with np.load(filename) as npz:
                    all_terms = npz.files

            elif self.associate_mat:
                filename = self.dirname / self.fname_budgets.format("mat")
                ret = loadmat(filename)
                all_terms = [
                    key for key in ret if key[0] != "_"
                ]  # ignore `__header__`, etc.

            else:
                raise AttributeError("existing_budgets(): How did you get here? ")

            if budget is None:  # i.e. requesting all budgets
                return all_terms  # we can stop here without sorting through each budget

            tup_list = [self.key[t] for t in all_terms]  # list of associated tuples
            t_list = []  # this is the list to be built and returned

            for b in budget_list:
                t_list += [tup for tup in tup_list if tup[0] == b]

        # else:
        if len(t_list) == 0:
            warnings.warn("existing_terms(): No terms found for budget " + str(budget))

        return t_list

    def _read_budgets_padeops(self, key_subset, tidx):
        """
        Uses a method similar to ReadVelocities_Budget() in PadeOpsViz to read and store full-field budget terms.
        """

        budget4_components = [11, 22, 33, 13, 23]

        if tidx is None:
            if self.budget or self.budget_tidx is not None:
                # if there are budgets loaded, continue loading from that TIDX
                tidx = self.budget_tidx
            else:
                # load budgets from the last available TIDX
                tidx = self.unique_budget_tidx(return_last=True)

        elif tidx not in self.all_budget_tidx:
            # find the nearest that actually exists
            tidx_arr = np.array(self.all_budget_tidx)
            closest_tidx = tidx_arr[np.argmin(np.abs(tidx_arr - tidx))]

            self.print(
                "Requested budget tidx={:d} could not be found. Using tidx={:d} instead.".format(
                    tidx, closest_tidx
                )
            )
            tidx = closest_tidx

        # Additional deficitIO steps to parse budget terms here:
        for key in key_subset:
            budget, term = DeficitIO.key[key]
            if budget == 4:
                component = budget4_components[int(np.floor((term - 1) / 10))]

                if term > 10:
                    term = term % 10
                    if term == 0:
                        term = 10
                searchstr = f"Run{self.runid:02d}_deficit_budget{budget:01d}_{component:02d}_term{term:02d}_t{tidx:06d}_*.s3D"

            else:
                searchstr = f"Run{self.runid:02d}_deficit_budget{budget:01d}_term{term:02d}_t{tidx:06d}_*.s3D"

            try:
                u_fname = next(self.dirname.glob(searchstr))
            except StopIteration as e:
                raise FileNotFoundError(
                    f"No matching files found at {self.dirname / searchstr}"
                )

            self.budget_n = int(
                re.findall(r".*_t\d+_n(\d+)", str(u_fname))[0]
            )  # extract n from string
            self.budget_tidx = tidx

            temp = np.fromfile(u_fname, dtype=np.dtype(np.float64), count=-1)
            self.budget[key] = temp.reshape(
                (self.grid.nx, self.grid.ny, self.grid.nz), order="F"
            )  # reshape into a 3D array

        if self.verbose and len(key_subset) > 0:
            print(
                "PadeOpsViz loaded the deficit budget fields at time:"
                + "{:.06f}".format(tidx)
            )

    def unique_budget_tidx(self, return_last=False):
        """
        Pulls all the unique tidx values from a directory.

        Parameters
        ----------
        return_last (bool) : If False, returns only the largest TIDX associated with budgets.
            Else, returns an entire list of unique tidx associated with budgets. Default False
        """

        # TODO: fix for .npz

        return self.unique_tidx(
            return_last=return_last, search_str=r"Run{:02d}.*deficit_budget.*_t(\d+).*"
        )

    # =============== TODO: MOVE BUDGET COMPUTATION TO SEPARATE FILE ===============

    def grad_stress_calc(self, tidx=None, Lref=1):
        """
        Calculates the velocity and reynolds stress gradients
        """

        self.budget["ddxk_delta_uiuj"] = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 3, 3, 3])
        self.budget["ddxk_delta_ui_base_uj"] = np.zeros(
            [self.grid.nx, self.grid.ny, self.grid.nz, 3, 3, 3]
        )

        tmp_delta_uiuj = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 3, 3])
        tmp_delta_ui_base_uj = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 3, 3])

        tmp_delta_uiuj[:, :, :, 0, 0] = self.budget["delta_uu"]
        tmp_delta_uiuj[:, :, :, 0, 1] = self.budget["delta_uv"]
        tmp_delta_uiuj[:, :, :, 0, 2] = self.budget["delta_uw"]
        tmp_delta_uiuj[:, :, :, 1, 1] = self.budget["delta_vv"]
        tmp_delta_uiuj[:, :, :, 1, 2] = self.budget["delta_vw"]
        tmp_delta_uiuj[:, :, :, 2, 2] = self.budget["delta_ww"]

        tmp_delta_ui_base_uj[:, :, :, 0, 0] = self.budget["delta_u_base_u"]
        tmp_delta_ui_base_uj[:, :, :, 0, 1] = self.budget["delta_u_base_v"]
        tmp_delta_ui_base_uj[:, :, :, 0, 2] = self.budget["delta_u_base_w"]
        tmp_delta_ui_base_uj[:, :, :, 1, 0] = self.budget["base_u_delta_v"]
        tmp_delta_ui_base_uj[:, :, :, 1, 1] = self.budget["delta_v_base_v"]
        tmp_delta_ui_base_uj[:, :, :, 1, 2] = self.budget["delta_v_base_w"]
        tmp_delta_ui_base_uj[:, :, :, 2, 0] = self.budget["base_u_delta_w"]
        tmp_delta_ui_base_uj[:, :, :, 2, 1] = self.budget["base_v_delta_w"]
        tmp_delta_ui_base_uj[:, :, :, 2, 2] = self.budget["delta_w_base_w"]

        for j in range(3):
            for k in range(3):
                print(
                    np.shape(
                        np.gradient(
                            tmp_delta_uiuj[:, :, :, j, k],
                            self.xLine * Lref,
                            self.yLine * Lref,
                            self.zLine * Lref,
                        )
                    )
                )
                self.budget["ddxk_delta_uiuj"][:, :, :, :, j, k] = np.transpose(
                    np.gradient(
                        tmp_delta_uiuj[:, :, :, j, k],
                        self.xLine * Lref,
                        self.yLine * Lref,
                        self.zLine * Lref,
                    ),
                    [1, 2, 3, 0],
                )
                self.budget["ddxk_delta_ui_base_uj"][:, :, :, :, j, k] = np.transpose(
                    np.gradient(
                        tmp_delta_ui_base_uj[:, :, :, j, k],
                        self.xLine * Lref,
                        self.yLine * Lref,
                        self.zLine * Lref,
                    ),
                    [1, 2, 3, 0],
                )

        return

    def rans_calc(self, tidx=None):
        """
        Calculates the rans budget terms (splits the advection term)
        """
        if "xadv_mean" in self.budget:
            return

        self.budget["xadv_mean"] = (
            self.budget["xAdv_delta_delta_mean"] + self.budget["xAdv_base_delta_mean"]
        )
        self.budget["yadv_mean"] = (
            self.budget["yAdv_delta_delta_mean"] + self.budget["yAdv_base_delta_mean"]
        )
        self.budget["zadv_mean"] = (
            self.budget["zAdv_delta_delta_mean"] + self.budget["zAdv_base_delta_mean"]
        )

        self.budget["xturb"] = (
            self.budget["xAdv_delta_delta_fluc"]
            + self.budget["xAdv_base_delta_fluc"]
            + self.budget["xAdv_delta_base_fluc"]
        )
        self.budget["yturb"] = (
            self.budget["yAdv_delta_delta_fluc"]
            + self.budget["yAdv_base_delta_fluc"]
            + self.budget["yAdv_delta_base_fluc"]
        )
        self.budget["zturb"] = (
            self.budget["zAdv_delta_delta_fluc"]
            + self.budget["zAdv_base_delta_fluc"]
            + self.budget["zAdv_delta_base_fluc"]
        )

    def xmom_budget_calc(self, pre, prim):
        """
        Calculates the terms in the streamwise momentum budget
        grouped according to the forward marching a priori analysis
        """

        # read in necessary terms for deficit budget
        def_budget0_terms = [term for term in self.key if self.key[term][0] == 0]
        def_budget1_terms = [term for term in self.key if self.key[term][0] == 1]

        self.read_budgets(budget_terms=def_budget0_terms)
        self.read_budgets(budget_terms=def_budget1_terms)

        # read in necessary terms for base and primary budget
        budget0_terms = ["ubar", "vbar", "wbar"]
        pre.read_budgets(budget_terms=budget0_terms)
        prim.read_budgets(budget_terms=budget0_terms)

        # calculate gradients
        self.grad_calc()
        pre.grad_calc()
        prim.grad_calc()

        # perform RANS calculation
        self.rans_calc()

        # separate out advection terms
        self.budget["xadv_mean_x"] = -prim.budget["ubar"] * self.budget["dUdx"]
        self.budget["xadv_mean_y_delta"] = -self.budget["delta_v"] * self.budget["dUdy"]
        self.budget["xadv_mean_z_delta"] = -self.budget["delta_w"] * self.budget["dUdz"]
        self.budget["xadv_mean_y_base"] = -pre.budget["vbar"] * self.budget["dUdy"]
        self.budget["xadv_mean_z_base"] = -pre.budget["wbar"] * self.budget["dUdz"]

        self.budget["xadv_mean_yz_base"] = (
            self.budget["xadv_mean_y_base"] + self.budget["xadv_mean_z_base"]
        )
        self.budget["xadv_mean_yz_delta"] = (
            self.budget["xadv_mean_y_delta"] + self.budget["xadv_mean_z_delta"]
        )
        self.budget["xadv_mean_yz_delta_total"] = (
            self.budget["xadv_mean_y_delta"]
            + self.budget["xadv_mean_z_delta"]
            + self.budget["xAdv_delta_base_mean"]
        )

    def wake_tke_budget_calc(self, pre, prim):
        """
        Calculate term sin the wake tKE budget
        """
        # # read in necessary terms for precursor and primary budgets
        budget_terms = [term for term in pre.key if pre.key[term][0] == 3]

        dx = self.dx
        dy = self.dy
        dz = self.dz

        # define TKE
        pre.budget["TKE"] = 0.5 * (
            pre.budget["uu"] + pre.budget["vv"] + pre.budget["ww"]
        )
        prim.budget["TKE"] = 0.5 * (
            prim.budget["uu"] + prim.budget["vv"] + prim.budget["ww"]
        )

        self.budget["TKE_wake"] = prim.budget["TKE"] - pre.budget["TKE"]

        # calculate TKE wake budget (prim - pre)
        for term in budget_terms:
            self.budget[term + "_wake"] = prim.budget[term] - pre.budget[term]

        # addition base advection term needs to be removed from TKE_adv_wake
        self.budget["TKE_adv_delta_base_k_wake"] = -advection(
            [self.budget["delta_u"], self.budget["delta_v"], self.budget["delta_w"]],
            pre.budget["TKE"],
            dx,
            dy,
            dz,
        )

    def tke_budget_calc(self, pre, prim):
        """
        Calculates terms in the TKE budget grouped together
        """

        # read in necessary terms for deficit budget
        def_budget_terms = [term for term in self.key if self.key[term][0] == 3]
        self.read_budgets(budget_terms=def_budget_terms)
        def_budget0_terms = [
            "delta_u",
            "delta_v",
            "delta_w",
            "delta_uu",
            "delta_vv",
            "delta_ww",
            "delta_uv",
            "delta_vw",
            "delta_uw",
            "delta_u_base_u",
            "delta_u_base_v",
            "base_u_delta_v",
            "delta_u_base_w",
            "base_u_delta_w",
            "delta_v_base_v",
            "delta_v_base_w",
            "base_v_delta_w",
            "delta_w_base_w",
            "delta_tau11",
            "delta_tau12",
            "delta_tau13",
            "delta_tau22",
            "delta_tau23",
            "delta_tau33",
        ]
        self.read_budgets(budget_terms=def_budget0_terms)

        # read in necessary terms for precursor and primary budgets
        budget_terms = [term for term in pre.key if pre.key[term][0] == 3]
        pre.read_budgets(budget_terms=budget_terms)
        prim.read_budgets(budget_terms=budget_terms)

        budget0_terms = [
            "ubar",
            "vbar",
            "wbar",
            "uu",
            "vv",
            "ww",
            "uv",
            "uw",
            "vw",
            "tau11",
            "tau12",
            "tau22",
            "tau23",
            "tau33",
            "tau13",
        ]
        pre.read_budgets(budget_terms=budget0_terms)
        prim.read_budgets(budget_terms=budget0_terms)

        # calculate velocity gradients
        pre.grad_calc()
        prim.grad_calc()
        self.grad_calc()

        dx = self.dx
        dy = self.dy
        dz = self.dz

        # define TKE
        pre.budget["TKE"] = 0.5 * (
            pre.budget["uu"] + pre.budget["vv"] + pre.budget["ww"]
        )
        prim.budget["TKE"] = 0.5 * (
            prim.budget["uu"] + prim.budget["vv"] + prim.budget["ww"]
        )

        self.budget["TKE_wake"] = prim.budget["TKE"] - pre.budget["TKE"]
        self.budget["delta_TKE"] = 0.5 * (
            self.budget["delta_uu"] + self.budget["delta_vv"] + self.budget["delta_ww"]
        )
        self.budget["delta_ui_base_ui"] = (
            self.budget["delta_u_base_u"]
            + self.budget["delta_v_base_v"]
            + self.budget["delta_w_base_w"]
        )

        # make stress tensors
        delta_ui_base_uj = construct_delta_ui_base_uj(self)
        # print(delta_ui_base_uj)

        base_uiuj = construct_uiuj(pre)

        # make velocity gradient tensors
        base_duidxj = construct_duidxj(pre)

        delta_duidxj = construct_duidxj(self)

        # calculate TKE wake budget (prim - pre)
        for term in budget_terms:
            self.budget[term + "_wake"] = prim.budget[term] - pre.budget[term]

        # addition base advection term needs to be removed from TKE_adv_wake
        self.budget["TKE_adv_delta_base_k_wake"] = -advection(
            [self.budget["delta_u"], self.budget["delta_v"], self.budget["delta_w"]],
            pre.budget["TKE"],
            dx,
            dy,
            dz,
        )

        # self.budget['TKE_adv_wake'] = self.budget['TKE_adv_wake'] - self.budget['TKE_adv_delta_base_k_wake']

        # advection terms
        self.budget["mixed_TKE_base_adv"] = -advection(
            [pre.budget["ubar"], pre.budget["vbar"], pre.budget["wbar"]],
            self.budget["delta_ui_base_ui"],
            dx,
            dy,
            dz,
        )
        self.budget["mixed_TKE_delta_adv"] = -advection(
            [self.budget["delta_u"], self.budget["delta_v"], self.budget["delta_w"]],
            self.budget["delta_ui_base_ui"],
            dx,
            dy,
            dz,
        )

        self.budget["mixed_TKE_adv_delta_base_k"] = self.budget[
            "TKE_adv_delta_base_k_wake"
        ]
        self.budget["mixed_TKE_adv_delta_base"] = (
            self.budget["mixed_TKE_base_adv"] - self.budget["TKE_adv_delta_base"]
        )

        self.budget["mixed_TKE_adv"] = (
            self.budget["TKE_adv_wake"]
            - self.budget["TKE_adv"]
            - self.budget["TKE_adv_delta_base"]
        )

        # production terms
        self.budget["mixed_TKE_prod_base_base_delta"] = -np.sum(
            np.multiply(base_uiuj, delta_duidxj), axis=(3, 4)
        )
        self.budget["mixed_TKE_prod_base_delta_delta"] = -np.sum(
            np.multiply(np.transpose(delta_ui_base_uj, [0, 1, 2, 4, 3]), delta_duidxj),
            axis=(3, 4),
        )
        self.budget["mixed_TKE_prod_base_delta_base"] = -np.sum(
            np.multiply(np.transpose(delta_ui_base_uj, [0, 1, 2, 4, 3]), base_duidxj),
            axis=(3, 4),
        )
        self.budget["mixed_TKE_prod_delta_base_base"] = -np.sum(
            np.multiply(delta_ui_base_uj, base_duidxj), axis=(3, 4)
        )

        self.budget["mixed_TKE_prod"] = (
            self.budget["TKE_shear_production_wake"]
            - self.budget["TKE_production"]
            - self.budget["TKE_prod_delta_base"]
        )

        # transport terms
        self.budget["mixed_TKE_p_transport"] = (
            self.budget["TKE_p_transport_wake"] - self.budget["TKE_p_transport"]
        )
        self.budget["mixed_TKE_SGS_transport"] = (
            self.budget["TKE_SGS_transport_wake"] - self.budget["TKE_SGS_transport"]
        )
        self.budget["mixed_TKE_turb_transport"] = (
            self.budget["TKE_turb_transport_wake"]
            - self.budget["TKE_turb_transport"]
            - self.budget["TKE_turb_transport_delta_base"]
        )

        # buoyancy
        self.budget["mixed_TKE_buoyancy"] = (
            self.budget["TKE_buoyancy_wake"] - self.budget["TKE_buoyancy"]
        )

        # dissipation
        self.budget["mixed_TKE_dissipation"] = (
            self.budget["TKE_dissipation_wake"] - self.budget["TKE_dissipation"]
        )

    def non_dim_vel(self):
        vel_keys = ["delta_u", "delta_v", "delta_w"]

        Ug = key_search_r(self.input_nml, "g_geostrophic")

        for key in vel_keys:
            self.budget[key] /= Ug
