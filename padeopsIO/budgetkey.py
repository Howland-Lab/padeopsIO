"""
Module containing a bi-directional has thable for the mapping and inverse-mapping
for budget terms in index-tuple form (budget #, term #) defined in PadeOps and common
form as defined in get_key()

Kirby Heck
March 2022
"""


class bidict(dict):
    """
    Bi-directional dictionary, courtesy of:
    https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table

    Used to look up a mapping and inverse mapping (not necessarily unique) between key parings.
    """

    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


def get_key():
    """
    Returns a bidirectional hash table between colloquial string values for budget terms and
    a tuple-look up ordered (budget #, term #) as defined in e.g. budget_time_avg.F90.
    """
    key = {  # BUDGET 0 TERMS: (1st and second order averages, scalars excluded)
        "ubar": (0, 1),
        "vbar": (0, 2),
        "wbar": (0, 3),
        "uu": (0, 4),
        "uv": (0, 5),
        "uw": (0, 6),
        "vv": (0, 7),
        "vw": (0, 8),
        "ww": (0, 9),
        "pbar": (0, 10),
        "tau11": (0, 11),
        "tau12": (0, 12),
        "tau13": (0, 13),
        "tau22": (0, 14),
        "tau23": (0, 15),
        "tau33": (0, 16),
        "pu": (0, 17),
        "pv": (0, 18),
        "pw": (0, 19),
        "uk": (0, 20),
        "vk": (0, 21),
        "wk": (0, 22),
        "ujtau1j": (0, 23),
        "ujtau2j": (0, 24),
        "ujtau3j": (0, 25),
        "Tbar": (0, 26),
        "uT": (0, 27),
        "vT": (0, 28),
        "wT": (0, 29),
        "TT": (0, 30),
        "wb": (0, 31),
        # BUDGET 1 TERMS: (momentum)
        "DuDt": (1, 1),  # x-advection
        "dpdx": (1, 2),  # x-pressure gradient
        "xSGS": (1, 3),  # x-sub grid stresses
        "xAD": (1, 4),  # x-Actuator disk
        "DvDt": (1, 5),
        "dpdy": (1, 6),
        "ySGS": (1, 7),
        "DwDt": (1, 8),
        "dpdz": (1, 9),
        "zSGS": (1, 10),
        "xCor": (1, 11),  # x-coriolis
        "xGeo": (1, 12),  # x-geostrophic pressure grad.
        "yCor": (1, 13),
        "yGeo": (1, 14),
        "yAD": (1, 15),
        "zBuoy": (1, 16),  # z buoyancy term
        # BUDGET 2 TERMS: (MKE)  TODO - improve the naming keys
        "MKE_shear": (2, 1),
        "MKE_adv": (2, 2),
        "MKE_rs_transport": (2, 3),
        "MKE_p_transport": (2, 4),
        "MKE_sgs_transport": (2, 5),
        "MKE_dissipation": (2, 6),
        "MKE_AD": (2, 7),
        "MKE_geostrophic": (2, 8),
        "MKE_coriolis": (2, 9),
        "MKE_buoyancy": (2, 10),
        # BUDGET 3 TERMS: (TKE)
        "TKE_shear_production": (3, 1),
        "TKE_adv": (3, 2),
        "TKE_turb_transport": (3, 3),
        "TKE_p_transport": (3, 4),
        "TKE_sgs_transport": (3, 5),
        "TKE_dissipation": (3, 6),
        "TKE_AD": (3, 7),
        "TKE_buoyancy": (3, 8),
        "TKE_storage": (3, 9),
        # BUDGET 4 TERMS: TODO
        # BUDGET 5 TERMS: Wake deficit
        "uwake": (5, 1),
        "vwake": (5, 2),
        "wwake": (5, 3),
    }

    return bidict(key)


def key_labels(u="U", l="D"):
    """
    Returns a dictionary that assigns a label to each budget key.
    """
    key = {
        # FIELDS
        "u": f"$u/{u}$",
        "v": f"$v/{u}$",
        "w": f"$w/{u}$",
        "T": f"$\\theta$",
        "p": f"$p/{u}^2$",
        "pfrn": f"$p^\\mathrm{{fr}}/{u}^2$",
        "ptrb": f"$p^\\mathrm{{trb}}/{u}^2$",
        "pdns": f"$p^\\mathrm{{dns}}/{u}^2$",
        # BUDGET 0 TERMS: (1st and second order averages, scalars excluded)
        "ubar": "$\\bar{u}/U$",
        "vbar": "$\\bar{v}/U$",
        "wbar": "$\\bar{w}/U$",
        "uu": "$\\overline{u'u'}/U^2$",
        "uv": "$\\overline{u'v'}/U^2$",
        "uw": "$\\overline{u'w'}/U^2$",
        "vv": "$\\overline{v'v'}/U^2$",
        "vw": "$\\overline{v'w'}/U^2$",
        "ww": "$\\overline{w'w'}/U^2$",
        "pbar": "$\\bar{p}/U^2$",
        "tau11": "$\\tau_{11}$",
        "tau12": "$\\tau_{12}$",
        "tau13": "$\\tau_{13}$",
        "tau22": "$\\tau_{22}$",
        "tau23": "$\\tau_{23}$",
        "tau33": "$\\tau_{33}$",
        "pu": "$\\overline{p'u'}$",
        "pv": "$\\overline{p'v'}$",
        "pw": "$\\overline{p'w'}$",
        "uk": "$\\overline{k'u'}$",
        "vk": "$\\overline{k'v'}$",
        "wk": "$\\overline{k'w'}$",
        "ujtau1j": (0, 23),
        "ujtau2j": (0, 24),
        "ujtau3j": (0, 25),
        "Tbar": "$\\bar{\\theta}$",
        "uT": "$\\overline{\\theta'u'}$",
        "vT": "$\\overline{\\theta'v'}$",
        "wT": "$\\overline{\\theta'w'}$",
        "TT": "$\\overline{\\theta'\\theta'}$",
        # BUDGET 1 TERMS: (momentum)
        "DuDt": "$-Du/Dt$",  # x-advection
        "dpdx": "$-\\partial p/\\partial x$",  # x-pressure gradient
        "xSGS": "$x$-SGS",  # x-sub grid stresses
        "xAD": "AD$_x$",  # x-Actuator disk
        "DvDt": "$-Dv/Dt$",
        "dpdy": "$-\\partial p/\\partial y$",
        "ySGS": "$y$-SGS",
        "DwDt": "$Dw/Dt$",
        "dpdz": "$\\partial p/\\partial z$",
        "zSGS": "$z$-SGS",
        "xCor": "$x$-Coriolis",  # x-coriolis
        "xGeo": "$x$-Geostrophic",  # x-geostrophic pressure grad.
        "yCor": "$y$-Coriolis",
        "yGeo": "$y$-Geostrophic",
        "yAD": "AD$_y$",
        # BUDGET 2 TERMS: (MKE)  TODO - improve the naming keys
        "MKE_TKE_loss": (2, 1),
        "MKE_adv": (2, 2),
        "MKE_tau_transport": (2, 3),
        "MKE_p_transport": (2, 4),
        "MKE_SGS_transport": (2, 5),
        "MKE_dissipation": (2, 6),
        "MKE_AD": (2, 7),
        "MKE_geostrophic": (2, 8),
        "MKE_coriolis": (2, 9),
        # BUDGET 3 TERMS: (TKE)
        "TKE_shear_production": (3, 1),
        "TKE_turb_transport": (3, 2),
        "TKE_p_strain": (3, 3),
        "TKE_p_transport": (3, 4),
        "TKE_SGS_transport": (3, 5),
        "TKE_dissipation": (3, 6),
        "TKE_buoyancy": (3, 7),
        "TKE_coriolis": (3, 8),
        "TKE_AD": (3, 9),
        # BUDGET 4 TERMS: TODO
        # BUDGET 5 TERMS: Wake deficit
        "uwake": "$\\Delta u_{wake}$",
        "vwake": "$\\Delta v_{wake}$",
        "wwake": "$\\Delta w_{wake}$",
        # SPATIAL COORDINATES
        "x": f"$x/{l}$",
        "y": f"$y/{l}$",
        "z": f"$z/{l}$",
        "t": f"$t{u}/{l}$",
    }

    return key
