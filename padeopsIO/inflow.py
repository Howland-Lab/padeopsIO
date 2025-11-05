"""
Functions for a BudgetIO object to extract the inflow profile from offline 
computation given the inflow parameters, or from the time-averaged velocity 
inlet profiles. 

Deprecated 2025 Jan 15
"""

import warnings
import numpy as np


class InflowParser:
    """
    This class is an interface with supplemental functions that is not meant to be instantiated.
    """

    def inflow_offline(**kwargs):
        """
        Parse inflow from the namelist variables found in Fortran. This function should be kept
        consistent with initialize.F90 in PadeOps.

        Arguments
        ---------
        InflowProfileType (int) : Profile type:
            0: Uniform streamwise inflow
            1: tanh streamwise inflow
            2: tanh streamwise and lateral inflow
            3: piecewise linear streamwise inflow               (deprecated)
            4: piecewise linear lateral inflow                  (deprecated)
            5: piecewise linear streamwise and lateral inflow   (deprecated)
            6: piecewise linear veered inflow
            7: piecewise linear sheared and veered inflow
        InflowProfileThick (float) : thickness of linear section
        InflowProfileAmplit (float) : amplitude of tanh curve
        uInflow (float) : normalization on u (nominally = 1.0)
        vInflow (float) : controls the degree of veer
        zLine (array) : array of z-values in the mesh discretization
        buffer (float) : value between 0 and 1 to prevent u<0 (reverse flow). Hard-coded as 0.8 in initialize.F90

        Returns
        -------
        u, v (array) : [nz x 1] array of streamwise and lateral velocities at the inlet

        Raises
        ------
        NameError if insufficient or incompatible input arguments

        """

        #         print(kwargs)

        # check keyword arguments
        if "zLine" in kwargs:
            zLine = kwargs["zLine"]
        else:
            warnings.warn("InflowParser.inflow_offline(): No z-coordinates specified!")
            return

        # because not all the profiles need all these variables, try to parse variables individually
        # may throw NameError if a required variable is not provided

        # have been having trouble with case-sensivity on kwargs:
        if "inflowprofiletype" in kwargs:
            InflowProfileType = kwargs["inflowprofiletype"]
        if "inflowprofilethick" in kwargs:
            InflowProfileThick = kwargs["inflowprofilethick"]
        if "inflowprofileamplit" in kwargs:
            InflowProfileAmplit = kwargs["inflowprofileamplit"]
        if "uinflow" in kwargs:
            uInflow = kwargs["uinflow"]
        if "vinflow" in kwargs:
            vInflow = kwargs["vinflow"]

        if "buffer" in kwargs:
            buffer = kwargs["buffer"]
        else:
            buffer = 0.8

        # set "default" values/allocate memory/ensure variables exist
        u = np.ones(zLine.shape)
        v = np.zeros(zLine.shape)

        # because zLine is cell-centered, the average is the quickest way to recover zMid
        zMid = np.mean(zLine)

        # compute profile based on type:

        if InflowProfileType == 0:  # TODO - Do these need to be cast into strings?
            # 0: Uniform streamwise inflow
            u = u * uInflow
            # scale u

        elif InflowProfileType == 1:
            # 1: tanh streamwise inflow
            u = uInflow * (
                1.0 + InflowProfileAmplit * np.tanh((zLine - zMid) / InflowProfileThick)
            )

        elif InflowProfileType == 2:
            # 2: tanh streamwise and lateral inflow
            # note this is consistent with PadeOps but not "postiive" veer
            u = uInflow * (
                1.0 + InflowProfileAmplit * np.tanh((zLine - zMid) / InflowProfileThick)
            )
            v = vInflow * np.tanh((zLine - zMid) / InflowProfileThick)

        elif InflowProfileType == 3:
            # 3: piecewise linear streamwise inflow (deprecated)
            u = uInflow * (1.0 + (zLine - zMid) / InflowProfileThick)
            u[u < 0.5] = 0.5
            u[u > 1.5] = 1.5

        elif InflowProfileType == 4:
            # 4: piecewise linear lateral inflow (deprecated)
            u = uInflow * u
            v = vInflow * (zMid - zLine) / InflowProfileThick

        elif InflowProfileType == 5:
            # 5: piecewise linear streamwise and lateral inflow (deprecated)
            u = uInflow * (1.0 + (zLine - zMid) / InflowProfileThick)
            u[u < 0.5] = 0.5
            u[u > 1.5] = 1.5
            v = vInflow * (zMid - zLine) / InflowProfileThick

        elif InflowProfileType == 6:
            # 6: piecewise linear veered inflow
            alpha = (zMid - zLine) / InflowProfileThick * vInflow
            a_max = np.pi / 2 * buffer
            alpha[alpha > a_max] = a_max
            alpha[alpha < (-a_max)] = -a_max

            u = uInflow * np.cos(alpha)
            v = vInflow * np.sin(alpha)

        elif InflowProfileType == 7:
            # 7: piecewise linear sheared and veered inflow
            # written slightly differently but produces the same result

            g = uInflow * ((zLine - zMid) / InflowProfileThick + 1.0)
            alpha = (zMid - zLine) / InflowProfileThick * vInflow
            a_max = np.pi / 2 * buffer
            g_min = max(1.0 - buffer, 1.0 - np.abs(a_max / (vInflow + 1e-18)))
            g_max = min(1.0 + buffer, 1.0 + np.abs(a_max / (vInflow + 1e-18)))

            # correct terms that are "outside of the buffer"
            for i in range(len(g)):
                if g[i] < g_min:
                    g[i] = g_min
                    alpha[i] = vInflow * (1 - g_min)
                elif g[i] > g_max:
                    g[i] = g_max
                    alpha[i] = vInflow * (1 - g_max)

            u = g * np.cos(alpha)
            v = g * np.sin(alpha)

        else:
            warnings.warn(
                "InflowParser.inflow_offline(): InflowProfileType {:d} not matched. Returning None".format(
                    InflowProfileType
                )
            )
            return None

        return u, v

    def inflow_budgets(budget_obj):
        """
        Reads budgets from a BudgetIO object and returns the (y-averaged) velocity field at the inlet

        Arguments
        ---------
        budget_obj (BudgetIO) : BudgetIO object with (or without) the following budgets loaded: 'ubar', 'vbar'

        Returns
        -------
        u, v
        """

        try:
            ubar = budget_obj.budget["ubar"]
            vbar = budget_obj.budget["vbar"]
        except KeyError as e:
            if budget_obj.verbose:
                print(
                    "InflowParser.inflow_budgets(): 'ubar' or 'vbar' not loaded. Calling read_budgets()."
                )

            budget_obj.read_budgets(["ubar", "vbar"])
            ubar = budget_obj.budget["ubar"]
            vbar = budget_obj.budget["vbar"]

        u = np.mean(ubar[0, :, :], axis=0)  # average in y
        v = np.mean(vbar[0, :, :], axis=0)

        return u, v
