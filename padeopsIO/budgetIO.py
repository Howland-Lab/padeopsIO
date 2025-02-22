"""
BudgetIO is the main linking object between PadeOps simulation data
and the python post-processing framework. Namely, BudgetIO is used
for reading instantaneous fields, time-averaged budgets, turbine 
velocity and power data, and more. 

These data can be read in from PadeOps source files and then saved
as fast-access .npz binaries or exported to matlab as .mat files. 

Kirby Heck
2024 May 23
"""

import numpy as np
import xarray as xr
import re
import warnings
from scipy.io import savemat, loadmat
from scipy.interpolate import RegularGridInterpolator as RGI
from pathlib import Path
from typing import Union, Literal, Any

from . import budgetkey, turbineArray
from .utils.io_utils import structure_to_dict, key_search_r
from .utils.nml_utils import parser
from .utils import tools
from .gridslice import get_xids, GridDataset


class BudgetIO:
    """
    Class for reading and writing outputfiles to PadeOps.
    """

    key = budgetkey.get_key()

    @property
    def x(self):
        return self.budget.coords.get("x", None).to_numpy()  # for now, return numpy array

    @property
    def y(self):
        return self.budget.coords.get("y", None).to_numpy()  # for now, return numpy array

    @property
    def z(self):
        return self.budget.coords.get("z", None).to_numpy()  # for now, return numpy array

    # NOTE: these are deprecated, use x, y, z instead
    @property
    def xLine(self):
        return self.x

    @property
    def yLine(self):
        return self.x

    @property
    def zLine(self):
        return self.x

    def print(self, *args):
        """Prints statements if self.quiet is False"""
        if not self.quiet:
            print(*args)

    def printv(self, *args):
        """Prints verbose messages if self.verbose is True"""
        if self.verbose:
            self.print(*args)

    def warn(self, *args):
        """Prints warning messages if self.show_warnings is True"""
        if self.show_warnings and not self.quiet:
            warnings.warn(*args)

    def __init__(
        self,
        dirname: Union[Path, str],
        verbose: bool = False,
        quiet: bool = False,
        show_warnings: bool = True,
        filename: Union[str, None] = None,
        runid: Union[int, None] = None,
        normalize_origin: Union[tuple[int], str, bool] = False,
        src: Union[Literal["padeops", "npz", "npy", "mat"], None] = None,
        padeops: bool = False,
        npz: bool = False,
        npy: bool = False,
        mat: bool = False,
        strict_runid: bool = False,
    ):
        """
        Creates different instance variables depending on the keyword arguments given.

        Every instance needs a directory name. If this object is reading information
        from output files dumped by PadeOps, then this is the directory where those
        files are stored. This object may also read information from a local subset of
        saved data.

        The BudgetIO class will try to initialize from the type of source files
        requested in kwargs `src`. Alternatively, the user may pass in a boolean
        for kwarg [`padeops`, `mat`, `npz`, `npy`].

        # read from source files in directory "data"
        >>> sim = pio.BudgetIO(r"/path/to/source/data", padeops=True, runid=1)

        The `filename` flag is assumed to be the directory name (self.dirname.name)
        unless it is specified. When reading from .npz files, it is possible that the
        saved filename differs from the directory which the data are stored.

        >>> sim = pio.BudgetIO(r"/path/to/mydata", npz=True)  # looks for file mydata_budgets.npz
        >>> sim = pio.BudgetIO(r"/path/to/moved/data", npz=True, filename="mydata")  # still looks for file mydata_budgets.npz

        Regardless of the method of reading budget files, __init__ will initialize
        the following fields:

        RUN INFORMATION:
            filename, dirname, runid
        DOMAIN VARIABLES:
            Lx, Ly, Lz, nx, ny, nz, dx, dy, dz, xLine, yLine, zLine,
        TURBINE VARIABLES:
            n_turb,
        PHYSICS:
            Re, Ro, Fr
        BUDGET VARIABLES:
            last_tidx, last_n,

        """

        self.quiet = quiet  # suppresses messages

        # print verbose? default False
        if verbose:
            self.verbose = True
            if self.quiet:
                self.quiet = False
                self.print(
                    "__init__(): `quiet` is True but `verbose` is False. Setting `quiet=False`"
                )

            self.printv("Attempting to initialize BudgetIO object at", dirname)
        else:
            self.verbose = False

        self.show_warnings = show_warnings

        if isinstance(dirname, str):
            dirname = Path(dirname)  # using pathlib for everything

        self.dirname = dirname
        self.dir_name = self.dirname  # deprecate dir_name eventually

        # all files associated with this case will begin with <filename>
        self.filename = filename or dirname.name
        self.fname_budgets = self.filename + "_budgets.{:s}"  # standardize this here
        self.fname_meta = self.filename + "_metadata.{:s}"
        # ========== Associate files ==========

        # if we are given the required keywords, try to initialize from PadeOps source files
        self.associate_padeops = False
        self.associate_npz = False
        self.associate_mat = False
        self.associate_nml = False
        self.associate_fields = False
        self.associate_budgets = False
        self.associate_grid = False
        self.associate_turbines = False
        self.normalized_xyz = False

        if padeops or src == "padeops":
            # at this point, we are reading from PadeOps output files
            self.associate_padeops = True

            try:
                self._init_padeops(
                    runid=runid,
                    normalize_origin=normalize_origin,
                    strict_runid=strict_runid,
                )
                self.printv(
                    f"Initialized BudgetIO at {dirname} from PadeOps source files."
                )

            except OSError as err:
                self.warn(
                    "Attempted to read PadeOps output files, but at least one was missing."
                )
                self.print(err)
                raise

        elif mat or src == "mat":  # .mat saved files
            self.associate_mat = True
            self._init_mat()
            self.printv(f"Initialized BudgetIO at {dirname} from .mat files. ")

        elif npz or src == "npz":  # .npz saved files
            self.associate_npz = True
            self._init_npz()
            self.printv(f"Initialized BudgetIO at {dirname} from .npz files. ")

        elif npy or src == "npy":
            self.associate_npy = True
            self._init_npy(normalize_origin=normalize_origin)
            self.printv(f"Initialized BudgetIO at {dirname} from .npy files. ")

        else:
            raise AttributeError("__init__(): No init associated with the source type")

    def _init_padeops(self, runid=None, normalize_origin=False, strict_runid=False):
        """
        Initializes source files to be read from output files in PadeOps.

        Raises OSError if source files cannot be read
        """

        # parse namelist
        try:
            self._read_inputfile(
                runid=runid, strict_runid=strict_runid
            )  # this initializes convenience variables

        except IndexError as err:
            self.warn(
                f"_init_padeops(): {self.filename} could not find input file. Perhaps the directory does not exist? "
            )
            self.print(err)
            raise err

        # default time ID for initialization
        self.tidx = 0  # tidx defualts to zero
        self.time = 0

        # READ TURBINES, only do this if usewindturbines = True
        if (
            self.associate_nml
            and "windturbines" in self.input_nml.keys()
            and self.input_nml["windturbines"]["usewindturbines"]
        ):
            self.printv("_init_padeops(): Initializing wind turbine array object")

            turb_dir = Path(self.input_nml["windturbines"]["turbinfodir"])
            if not turb_dir.exists():
                # hotfix: maybe this folder was copied elsewhere
                turb_dir = self.dirname / "turb"

            num_turbines = self.input_nml["windturbines"]["num_turbines"]
            ADM_type = self.input_nml["windturbines"]["adm_type"]
            try:
                self.turbineArray = turbineArray.TurbineArray(
                    turb_dir,
                    num_turbines=num_turbines,
                    ADM_type=ADM_type,
                    verbose=self.verbose,
                )
                self.associate_turbines = True

                self.printv(
                    f"_init_padeops(): Finished initializing wind turbine array with {num_turbines:d} turbine(s)"
                )

            except FileNotFoundError as e:
                self.warn("Turbine file not found, bypassing associating turbines.")
                self.turbineArray = None
                self.printv(e)
            self.ta = self.turbineArray  # alias this for easier use

        # Throw an error if no RunID is found
        if "runid" not in self.__dict__:
            raise AttributeError(
                "No RunID found. To explicitly pass one in, use kwarg: runid="
            )

        # loads the grid, normalizes if `associate_turbines=True` and `normalize_origin='turb'`
        # (Should be done AFTER loading turbines to normalize origin)
        if not self.associate_grid:
            self._load_grid(normalize_origin=normalize_origin)

        # object is reading from PadeOps output files directly
        self.printv(f"BudgetIO initialized using info files at time: {self.time:.06f}")

        # try to associate fields
        # self.field = {}
        try:
            self.last_tidx = self.unique_tidx(
                return_last=True
            )  # last tidx in the run with fields
            self.associate_fields = True

        except FileNotFoundError as e:
            self.warn(f"_init_padeops(): {self.filename} no field files found.")

        # try to associate budgets
        try:
            self.all_budget_tidx = self.unique_budget_tidx(return_last=False)
            self.associate_budgets = True
        except FileNotFoundError as e:
            self.warn(f"_init_padeops(): {self.filename} no budget files found.")

        if (
            self.associate_fields
        ):  # The following are initialized as the final saved instanteous field and budget:
            self.field_tidx = self.last_tidx

        if self.associate_budgets:
            self.last_n = self.last_budget_n()  # last tidx with an associated budget
            self.budget_tidx = self.unique_budget_tidx(
                return_last=True
            )  # but may be changed by the user
            self.budget_n = self.last_n

    def _read_inputfile(self, runid=None, strict_runid=False):
        """
        Reads the input file (Fortran 90 namelist) associated with the CFD simulation.

        Parameters
        ----------
        runid : int
            RunID number to try and match, given inputfiles self.dirname.
            Default: None
        strict_runid : bool
            Must match the given runid to an input file.

        Returns
        -------
        None
        """

        if strict_runid and runid is None:
            raise ValueError(
                "_read_inputfile(): `strict_runid` is True but `runid` is None."
            )

        # search all files ending in '*.dat'
        inputfile_ls = list(self.dirname.glob("*.dat"))

        if len(inputfile_ls) == 0:
            raise FileNotFoundError(
                "_read_inputfile(): No inputfiles found at {:}".format(
                    self.dirname.resolve()
                )
            )

        self.printv("\tFound the following files:", inputfile_ls)

        # try to search all input files '*.dat' for the proper run and match it
        for inputfile in inputfile_ls:
            input_nml = parser(inputfile)
            self.printv("\t_read_inputfile(): trying inputfile", inputfile)

            try:
                tmp_runid = input_nml["io"]["runid"]
            except KeyError as e:
                self.printv("\t_read_inputfile(): no runid for", inputfile)
                tmp_runid = None  # not all input files have a RunID

            if runid is not None:
                if tmp_runid == runid:
                    self.input_nml = input_nml
                    self._convenience_variables()  # make some variables in the metadata more accessible, also loads grid
                    self.associate_nml = True  # successfully loaded input file

                    self.printv("\t_read_inputfile(): matched RunID with", inputfile)
                    return

            elif self.verbose:
                self.printv(
                    "\t_read_inputfile(): WARNING - no keyword `runid` given to init."
                )

        # if there are still no input files found, we've got a problem
        if strict_runid:
            raise FileNotFoundError(
                f"_read_inputfile(): No match found for `runid` = {runid}."
            )

        self.warn(
            f"_read_inputfile(): No match to given `runid`, reading namelist file from {inputfile_ls[0]}."
        )

        self.input_nml = parser(inputfile_ls[0])
        self._convenience_variables()  # make some variables in the metadata more accessible
        self.associate_nml = True  # successfully loaded input file

    def _convenience_variables(self):
        """
        Aside from reading in the Namelist, which has all of the metadata, also make some
        parameters more accessible.

        Called by _read_inputfile() and by _init_npz()

        Special note: these are all lower case when reading from the dictionary or namelist!
        """

        # RUN VARIABLES:
        try:
            self.runid = self.input_nml["io"]["runid"]
        except KeyError:
            self.print("_convenience_variables(): no runid found")
            self.runid = None

        # TURBINE VARIABLES:
        try:
            self.n_turb = self.input_nml["windturbines"]["num_turbines"]
        except KeyError:
            self.n_turb = 0

        # PHYSICS:
        if self.input_nml["physics"]["isinviscid"]:  # boolean
            self.Re = np.inf
        else:
            self.Re = self.input_nml["physics"]["re"]

        if self.input_nml["physics"]["usecoriolis"]:
            self.Ro = key_search_r(self.input_nml, "ro")
            self.lat = key_search_r(self.input_nml, "latitude")
            self.Ro_f = self.Ro / (2 * np.cos(self.lat * np.pi / 180))
        else:
            self.Ro = np.inf

        if key_search_r(self.input_nml, "isstratified"):
            self.Fr = key_search_r(self.input_nml, "fr")
        else:
            self.Fr = np.inf

        self.galpha = key_search_r(self.input_nml, "g_alpha")

    def _load_grid(
        self, x=None, y=None, z=None, origin=(0, 0, 0), normalize_origin=None
    ):
        """
        Creates dx, dy, dz, and xLine, yLine, zLine variables.

        Expects (self.)Lx, Ly, Lz, nx, ny, nz in kwargs or in self.input_nml
        """

        if self.associate_grid:
            self.printv("_load_grid(): Grid already exists. ")
            return

        if self.associate_padeops:
            # need to parse the inputfile to build the staggered grid
            gridkeys = ["nx", "ny", "nz", "lx", "ly", "lz"]
            gridvars = {key: key_search_r(self.input_nml, key) for key in gridkeys}

            x = np.arange(gridvars["nx"]) * gridvars["lx"] / gridvars["nx"]
            y = np.arange(gridvars["ny"]) * gridvars["ly"] / gridvars["ny"]
            # staggered in z
            z = (0.5 + np.arange(gridvars["nz"])) * gridvars["lz"] / gridvars["nz"]

        # initialize grid variable
        self.field = GridDataset(x=x, y=y, z=z)
        self.budget = GridDataset(x=x, y=y, z=z)
        self.grid = self.field.grid  # Grid3(x=x, y=y, z=z)
        # # copy grid keys into the namespace of `self`
        # for xi in ["x", "y", "z"]:
        #     for key in ["{:s}", "L{:s}", "d{:s}", "n{:s}"]:
        #         setattr(self, key.format(xi), getattr(self.grid, key.format(xi)))

        # self.xLine, self.yLine, self.zLine = (
        #     self.x,
        #     self.y,
        #     self.z,
        # )  # try to phase out xLine, etc.

        self.origin = origin  # default origin location
        if normalize_origin:  # not None or False
            self.normalize_origin(
                normalize_origin
            )  # expects tuple (x, y, z) or string "turb"

        self.associate_grid = True

    def normalize_origin(self, origin=None):
        """
        Normalize the origin to point `origin` (x, y, z)

        `origin` can also be a string "turb" to place the origin
        at the leading turbine.

        Parameters
        ----------
        xyz : None or tuple
            If tuple, moves the origin to (x, y, z)
            If none, resets the origin.
        """

        if origin in ["turb", "turbine"]:
            if self.associate_turbines:
                self.turbineArray.sort(by="xloc")
                origin = self.ta[0].pos  # position of the leading turbine
            else:
                if not self.quiet:
                    print(
                        "Attempted to normalize origin to `turbine`, but no turbines associated"
                    )
                return

        if origin is None:
            self.normalized_xyz = False
            origin = (0, 0, 0)
        else:
            self.normalized_xyz = True

        # move the origin now:
        for ds in [self.field, self.budget]:
            ds.coords["x"] = ds["x"] - (origin[0] - self.origin[0])
            ds.coords["y"] = ds["y"] - (origin[1] - self.origin[1])
            ds.coords["z"] = ds["z"] - (origin[2] - self.origin[2])
        self.origin = origin

    def _init_npz(self, normalize_origin=False):
        """
        Initializes the BudgetIO object by attempting to read .npz files
        saved from a previous BudgetIO object from write_npz().

        Expects target files:
        One filename including "{filename}_budgets.npz"
        One filename including "_metadata.npz"
        """
        # load metadata: expects a file named <filename>_metadata.npz
        filepath = self.dirname / self.fname_meta.format("npz")
        try:
            ret = np.load(filepath, allow_pickle=True)
        except FileNotFoundError as e:
            raise e

        self.input_nml = ret["input_nml"].item()
        self.associate_nml = True

        if "turbineArray" in ret.files:
            init_dict = ret["turbineArray"].item()
            init_ls = [t for t in init_dict["turbines"]]
            self.turbineArray = turbineArray.TurbineArray(init_ls=init_ls)
            self.associate_turbines = True

        origin = (0, 0, 0)
        if "origin" in ret.files:
            origin = ret["origin"]

        # set convenience variables:
        self._convenience_variables()
        self.associate_nml = True

        if not self.associate_grid:
            self._load_grid(
                x=np.squeeze(ret["x"]),
                y=np.squeeze(ret["y"]),
                z=np.squeeze(ret["z"]),
                origin=origin,
                normalize_origin=normalize_origin,
            )

        # check budget files exist
        if not (self.dirname / self.fname_budgets.format("npz")).exists():
            self.warn("No associated budget files found")
        else:
            self.associate_budgets = True
            self.budget_n = None
            self.budget_tidx = None
            self.last_n = None  # all these are missing in npz files 04/24/2023

        self.printv("_init_npz(): BudgetIO initialized using .npz files.")

    def _init_npy(self, **kwargs):
        """
        WARNING: Deprecated feature, use _init_npz() instead.

        Initializes the BudgetIO object by attempting to read .npy metadata files
        saved from a previous BudgetIO object from write_npz().

        Expects target files:
        One filename including "{filename}_budgets.npz"
        One filename including "_metadata.npy"
        """
        print("_init_npy(): Warning - deprecated. Use _init_npz() instead. ")

        # load metadata: expects a file named <filename>_metadata.npy
        filepath = self.dirname / self.fname_meta.format("npy")
        try:
            self.input_nml = np.load(filepath, allow_pickle=True).item()
        except FileNotFoundError as e:
            raise e

        # check budget files
        if not (self.dirname / self.fname_budgets.format("npz")).exists():
            self.warn("No associated budget files found")
        else:
            self.associate_budgets = True
            self.budget_n = None
            self.budget_tidx = None
            self.last_n = None  # all these are missing in npy files 04/24/2023

        # attempt to load turbine file - need this before loading grid
        if (
            "auxiliary" in self.input_nml.keys()
            and "turbineArray" in self.input_nml["auxiliary"]
        ):
            self.turbineArray = turbineArray.TurbineArray(
                init_dict=self.input_nml["auxiliary"]["turbineArray"]
            )
            self.associate_turbines = True

        self._convenience_variables()
        self.associate_nml = True

        if not self.associate_grid:
            self._load_grid(**kwargs)

        self.printv("_init_npz(): BudgetIO initialized using .npz files.")

    def _init_mat(self, normalize_origin=False):
        """
        Initializes the BudgetIO object by attempting to read .mat files
        saved from a previous BudgetIO object from write_mat().

        Expects target files:
        One filename including "{filename}_budgets.mat"
        One filename including "{filename}_metadata.mat"
        """

        # load metadata: expects a file named <filename>_metadata.mat
        filepath = self.dirname / self.fname_meta("mat")
        try:
            ret = loadmat(filepath)
        except FileNotFoundError as e:
            raise e

        self.input_nml = structure_to_dict(ret["input_nml"])
        self.associate_nml = True

        if "turbineArray" in ret.keys():
            init_dict = structure_to_dict(ret["turbineArray"])
            init_ls = [structure_to_dict(t) for t in init_dict["turbines"]]
            self.turbineArray = turbineArray.TurbineArray(init_ls=init_ls)
            self.associate_turbines = True

        origin = (0, 0, 0)
        if "origin" in ret.keys():
            origin = ret["origin"]

        # set convenience variables:
        self._convenience_variables()
        self.associate_nml = True

        if not self.associate_grid:
            self._load_grid(
                x=np.squeeze(ret["x"]),
                y=np.squeeze(ret["y"]),
                z=np.squeeze(ret["z"]),
                origin=origin,
                normalize_origin=normalize_origin,
            )

        # link budgets
        if not (self.dirname / self.fname_budgets.format(".mat")).exists():
            self.warn("No associated budget files found")
        else:
            self.associate_budgets = True
            self.budget_n = None
            self.budget_tidx = None
            self.last_n = None  # all these are missing in .mat files 07/03/2023

        self.printv("_init_mat(): BudgetIO initialized using .mat files.")

    def write_data(
        self,
        write_dir=None,
        budget_terms="current",
        filename=None,
        overwrite=False,
        xlim=None,
        ylim=None,
        zlim=None,
        fmt="npz",
        xy_avg=False,
    ):
        """
        Saves budgets as .npz files. All budgets are put in one file, and
        simulation metadata is in a second file. Budgets follow keys as given
        in `padeopsIO.budgetkey`.

        Parameters
        ----------
        write_dir : str
            Location to write .npz files.
            Default: same directory as self.dirname
        budget_terms : list
            Budget terms to be saved (see ._parse_budget_terms()). Alternatively,
            use "current" to save the budget terms that are currently loaded.
        filename : str
            Sets the filename of written files
        overwrite : bool
            If true, will overwrite existing .npz files.
        xlim, ylim, zlim : slice bounds
            See BudgetIO.slice()
        fmt : str
            Format of output files, either "npz" or "mat". Default is "npz".
        """
        if not self.associate_budgets:
            self.warn("write_data(): No budgets associated, returning.")
            return

        # declare directory to write to, default to the working directory
        write_dir = write_dir or self.dirname

        if budget_terms == "current":
            key_subset = self.budget.keys()  # currently loaded budgets

        else:
            # need to parse budget_terms with the key
            key_subset = self._parse_budget_terms(budget_terms)

        # load budgets (TODO: add fields)
        if xy_avg:
            sl = self.xy_avg(budget_terms=key_subset, xlim=xlim, ylim=ylim, zlim=zlim)
        else:
            sl = self.slice(budget_terms=key_subset, xlim=xlim, ylim=ylim, zlim=zlim)

        # if `filename` is provided, write files with the provided name
        if filename is None:
            filename = self.filename
            fname_budgets = self.fname_budgets
        else:
            fname_budgets = filename + "_budgets.{:s}"

        write_dir = Path(write_dir)
        filepath = write_dir / fname_budgets.format(fmt)

        # don't unintentionally overwrite files...
        write_arrs = False
        if not filepath.exists():
            write_arrs = True
        elif overwrite:
            write_arrs = True
        else:
            self.warn(
                "Existing files found. Failed to write; pass overwrite=True to override."
            )
            return

        save_arrs = {}
        for key in key_subset:
            # crop the domain of the budgets here:
            save_arrs[key] = sl[key]

        # write files
        if write_arrs:
            self.printv("write_data(): attempting to save budgets to", filepath)

            if fmt == "npz":
                np.savez(filepath, **save_arrs)
            elif fmt == "mat":
                savemat(filepath, save_arrs)
            else:
                raise ValueError("File format `fmt` needs to be npz or mat")

            # SAVE METADATA
            self.write_metadata(write_dir, filename, fmt, sl["x"], sl["y"], sl["z"])

            self.print(
                "write_data: Successfully saved the following budgets: ",
                list(key_subset),
                "at " + str(filepath),
            )

    def write_npz(self, *args, **kwargs):
        """
        Saves data to .npz format (legacy code). See `self.write_data`
        """
        self.write_data(*args, fmt="npz", **kwargs)

    def write_mat(self, *args, **kwargs):
        """
        Saves data to .mat format (legacy code). See `self.write_data`
        """
        self.write_data(*args, fmt="mat", **kwargs)

    def write_metadata(self, write_dir, fname, src, x, y, z):
        """
        The saved budgets aren't useful on their own unless we
        also save some information like the mesh used in the
        simulation and some other information like the physical setup.

        That goes here.
        """

        save_vars = ["input_nml"]  # , 'xLine', 'yLine', 'zLine']
        save_dict = {key: self.__dict__[key] for key in save_vars}
        save_dict["x"] = x
        save_dict["y"] = y
        save_dict["z"] = z
        save_dict["origin"] = self.origin

        if self.associate_turbines:
            save_dict["turbineArray"] = self.turbineArray.todict()
            for k in range(self.turbineArray.num_turbines):
                # write turbine information
                for prop in ["power", "uvel", "vvel"]:
                    try:
                        save_dict["t{:d}_{:s}".format(k + 1, prop)] = (
                            self.read_turb_property(
                                tidx="all", prop_str=prop, steady=False, turb=k + 1
                            )
                        )
                    except KeyError as e:
                        pass

        filepath_meta = write_dir / (fname + f"_metadata.{src}")

        if src == "mat":
            savemat(filepath_meta, save_dict)

        elif src == "npz":
            np.savez(filepath_meta, **save_dict)

        self.printv(f"write_metadata(): metadata written to {filepath_meta}")

    def read_fields(self, field_terms=None, tidx=None, time=None):
        """
        Reads fields from PadeOps output files into the self.field dictionary.

        Parameters
        ----------
        field_terms : list
            list of field terms to read, must be be limited to:
            'u', 'v', 'w', 'p', 'T'
        tidx : int, optional
            reads fields from the specified time ID. Default: self.last_tidx
        time : float, optional
            reads fields from the specified time. Default: None

        Returns
        -------
        None
            Read fields are saved in self.field
        """

        if not self.associate_fields:
            raise AttributeError("read_fields(): No fields linked. ")

        dict_match = {
            "u": "uVel",
            "v": "vVel",
            "w": "wVel",
            "p": "prss",
            "T": "potT",
            "pfrn": "pfrn",  # fringe pressure
            "pdns": "pdns",  # DNS pressure... what is this?
            "ptrb": "ptrb",  # turbine pressure... what is this?
        }  # add more?

        # parse terms:
        if field_terms is None:
            terms = dict_match.keys()

        else:
            terms = [t for t in field_terms if t in dict_match.keys()]

        # parse tidx
        if tidx is None:
            if time is not None:
                tidx_all, times = self.get_tidx_pairs()
                _id = np.argmin(np.abs(times - time))
                tidx = tidx_all[_id]
                self.printv(
                    f"read_fields(): `time` = {time} passed in, found nearest time = {times[_id]}"
                )
            else:
                tidx = self.last_tidx

        else:  # find closest tidx
            tidx_all = self.unique_tidx()
            if tidx not in tidx_all:
                # find the nearest that actually exists
                closest_tidx = tidx_all[np.argmin(np.abs(tidx_all - tidx))]

                self.print(
                    f"Requested field tidx={tidx:d} could not be found. Using tidx={closest_tidx:d} instead."
                )
                tidx = closest_tidx

        # update self.time and self.tidx:
        self.tidx = tidx
        self.update_time(tidx)

        # the following is very similar to PadeOpsViz.ReadVelocities()

        for term in terms:
            fname = (
                self.dirname
                / f"Run{self.runid:02d}_{dict_match[term]:s}_t{self.tidx:06d}.out"
            )
            tmp = np.fromfile(fname, dtype=np.dtype(np.float64), count=-1)
            self.field[term] = tmp.reshape(self.grid.nxi, order="F")  # reshape into a 3D array

        self.print(
            f"BudgetIO loaded fields {str(list(terms)):s} at tidx: {self.tidx:d}, time: {self.time:.06f}"
        )

    def update_time(self, tidx):
        """Updates self.time"""
        info_fname = self.dirname / f"Run{self.runid:02d}_info_t{tidx:06d}.out"
        info = np.genfromtxt(info_fname, dtype=None)
        self.time = info[0]
        # return nothing

    def clear_budgets(self):
        """
        Clears any loaded budgets.

        Returns
        -------
        keys (list) : list of cleared budgets.
        """
        if not self.associate_budgets:
            self.printv("clear_budgets(): no budgets to clear. ")
            return

        loaded_keys = self.budget.keys()
        self.budget = GridDataset(coords=self.budget.coords)
        self.budget_n = None
        self.budget_tidx = None  # reset to final TIDX

        self.printv("clear_budgets(): Cleared loaded budgets: {}".format(loaded_keys))

        return loaded_keys

    def read_budgets(
        self,
        budget_terms="default",
        overwrite=False,
        tidx=None,
        time=None,
    ):
        """
        Accompanying method to write_budgets. Reads budgets saved as .npz files

        Parameters
        ----------
        budget_terms : list
            Budget terms (see ._parse_budget_terms() and budgetkey.py)
        overwrite : bool, optional
            If True, re-loads budgets that have already been loaded. Default False;
            checks existing budgets before loading new ones.
        tidx : int, optional
            If given, requests budget dumps at a specific time ID. Default None. This only affects
            reading from PadeOps output files; .npz and .mat are limited to one saved tidx.
        time : float, optional
            If given, requests budget dumps at a specific time. Default None.

        Returns
        -------
        None
            Saves result in self.budget
        """

        if not self.associate_budgets:
            raise AttributeError("read_budgets(): No budgets linked. ")

        # parse budget_terms with the key
        key_subset = self._parse_budget_terms(budget_terms)

        if time is not None:
            tidx_all, times = self.get_tidx_pairs(budget=True)
            _id = np.argmin(np.abs(times - time))
            tidx = tidx_all[_id]
            self.printv(
                f"read_fields(): `time` = {time} passed in, found nearest time = {times[_id]}"
            )

        # Decide: overwrite existing budgets or not?
        if overwrite:
            # clear budgets -- we are explicitly overwriting budgets
            self.clear_budgets()

        elif self.budget.keys() is not None:
            # budgets are already loaded, check which ones
            if (self.budget_tidx == tidx) or (tidx is None):
                # remove items that have already been loaded in -- this omits overwriting these terms
                key_subset = {
                    key: key_subset[key]
                    for key in key_subset
                    if key not in self.budget.keys()
                }

                if self.verbose:  # print which keys were removed
                    remove_keys = [
                        key for key in key_subset if key in self.budget.keys()
                    ]
                    if len(remove_keys) > 0:
                        self.print(
                            "read_budgets(): requested budgets that have already been loaded. \
                               \n  Removed the following: {}. Pass overwrite=True to read budgets anyway.".format(
                                remove_keys
                            )
                        )

            else:
                # clear budgets -- different tidx is currently loaded
                self.clear_budgets()

        if self.associate_padeops:
            self._read_budgets_padeops(key_subset, tidx=tidx)
        elif self.associate_npz:
            self._read_budgets_npz(key_subset)
        elif self.associate_mat:
            self._read_budgets_mat(key_subset)
        else:
            raise AttributeError("read_budgets(): No budgets linked. ")

        if len(key_subset) > 0:
            self.printv("read_budgets: Successfully loaded budgets. ")

    def _read_budgets_padeops(self, key_subset, tidx):
        """
        Uses a method similar to ReadVelocities_Budget() in PadeOpsViz to read and store full-field budget terms.
        """

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

        try:
            self.update_time(tidx)
        except FileNotFoundError:
            self.warn(f"Tried to update time, but no info file found for TIDX {tidx}")
            pass  # probably budget and field dumps not synchronized

        self.printv(f"Loading budgets {list(key_subset.keys())} from {tidx}")

        # Match requested keys with (budget, term) tuples and load Fortran binaries
        for key in key_subset:
            budget, term = BudgetIO.key[key]

            searchstr = f"Run{self.runid:02d}_budget{budget:01d}_term{term:02d}_t{tidx:06d}_*.s3D"
            try:
                u_fname = next(self.dirname.glob(searchstr))
            except StopIteration as e:
                raise FileNotFoundError(f"No files found at {searchstr}")

            self.budget_n = int(
                re.findall(".*_t\d+_n(\d+)", str(u_fname))[0]
            )  # extract n from string
            self.budget_tidx = tidx  # update self.budget_tidx

            tmp = np.fromfile(u_fname, dtype=np.dtype(np.float64), count=-1)
            self.budget[key] = tmp.reshape(self.grid.nxi, order="F")  # reshape into a 3D array

        if self.verbose and len(key_subset) > 0:
            print("BudgetIO loaded the budget fields at TIDX:" + "{:.06f}".format(tidx))

    def _read_budgets_npz(self, key_subset, mmap=None):
        """
        Reads budgets written by .write_npz() and loads them into memory
        """

        # load the npz file and keep the requested budget keys
        for key in key_subset:
            npz = np.load(self.dirname / self.fname_budgets.format("npz"))
            self.budget[key] = npz[key]

        self.printv(
            "BudgetIO loaded the following budgets from .npz: ",
            list(key_subset.keys()),
        )

    def _read_budgets_mat(self, key_subset):
        """
        Reads budgets written by .write_mat()
        """

        for key in key_subset:
            budgets = loadmat(self.dirname / self.fname_budgets.format("mat"))
            self.budget[key] = budgets[key]

        self.printv(
            "BudgetIO loaded the following budgets from .mat: ",
            list(key_subset.keys()),
        )

    def _parse_budget_terms(self, budget_terms):
        """
        Takes a list of budget terms, either keyed in index form
        (budget #, term #) or in common form (e.g. ['ubar', 'vbar'])
        and returns a subset of the `keys` dictionary that matches two
        together.
        `keys` dictionary is always keyed in plain text form.

        budget_terms can also be a string: 'all', or 'default'.

        'default' tries to load the following:
            Budget 0 terms: ubar, vbar, wbar, all Reynolds stresses, and p_bar
            Budget 1 terms: all momentum terms
        'all' checks what budgets exist and tries to load them all.

        For more information on the bi-directional keys, see budget_key.py

        Arguments
        ---------
        budget_terms : list of strings or string, see above
        """

        # add string shortcuts here... # TODO move shortcuts to budgetkey.py?
        if budget_terms is None:
            return dict()

        elif budget_terms == "current":
            budget_terms = list(self.budget.keys())

        elif budget_terms == "all":
            budget_terms = self.existing_terms()

        elif isinstance(budget_terms, str) and budget_terms in self.key:
            budget_terms = [budget_terms]  # cast to a list

        elif "budget" in budget_terms and any(chr.isdigit() for chr in budget_terms):
            budgetnum = re.findall("[0-5]", budget_terms)
            if len(budgetnum) < 1 or len(budgetnum) > 1:
                raise AttributeError(
                    'read_budgets(): budget_terms incorrectly specified. \n \
                                     String should contain a single number from 0-5, e.g. "budget0".'
                )
            else:
                budget_terms = [
                    term for term in self.key if self.key[term][0] == int(budgetnum[0])
                ]

        elif type(budget_terms) == str:
            self.warn(
                "keyword argument budget_terms must be either 'default', 'all', 'RANS' or a list."
            )
            return {}  # empty dictionary

        # parse through terms: they are either 1) valid, 2) missing (but valid keys), or 3) invalid (not in BudgetIO.key)
        existing_keys = self.existing_terms()
        # corresponding associated tuples (#, #)
        existing_tup = [self.key[key] for key in existing_keys]

        valid_keys = [t for t in budget_terms if t in existing_keys]
        missing_keys = [
            t for t in budget_terms if t not in existing_keys and t in self.key
        ]
        invalid_terms = [
            t for t in budget_terms if t not in self.key and t not in self.key.inverse
        ]

        valid_tup = [
            tup for tup in budget_terms if tup in existing_tup
        ]  # existing tuples
        missing_tup = [
            tup
            for tup in budget_terms
            if tup not in existing_tup and tup in self.key.inverse
        ]

        # now combine existing valid keys and valid tuples, removing any duplicates
        valid_terms = set(
            valid_keys + [self.key.inverse[tup][0] for tup in valid_tup]
        )  # combine and remove duplicates
        missing_terms = set(
            missing_keys + [self.key.inverse[tup][0] for tup in missing_tup]
        )

        # generate the key
        key_subset = {key: self.key[key] for key in valid_terms}

        # warn the user if some requested terms did not exist
        if len(key_subset) == 0:
            self.warn("_parse_budget_terms(): No keys being returned; none matched.")

        if len(missing_terms) > 0:
            self.warn(
                "_parse_budget_terms(): Several terms were requested but the following could not be found: \
                {}".format(
                    missing_terms
                )
            )

        if len(invalid_terms) > 0:
            self.warn(
                "_parse_budget_terms(): The following budget terms were requested but the following do not exist: \
                {}".format(
                    invalid_terms
                )
            )

        return key_subset

    def slice(
        self,
        budget_terms=None,
        field=None,
        field_terms=None,
        sl=None,
        keys=None,
        tidx=None,
        time=None,
        xlim=None,
        ylim=None,
        zlim=None,
        overwrite=False,
        **sel_kwargs,
    ):
        """
        Returns a slice of the requested budget term(s) as a dictionary.

        Parameters
        ----------
        budget_terms : list or string
            budget term or terms to slice from.
            If None, expects a value for `field` or `sl`
        field : array-like or dict of arraylike
            fields similar to self.field[] or self.budget[]
        field_terms: list
            read fields from read_fields().
        sl : GridDataset, optional.
            GridDataset or xarray Dataset from self.slice()
        keys : list
            fields in slice `sl`. Keys to slice into from the input slice `sl`
        tidx : int
            time ID to read budgets from, see read_budgets(). Default None
        time : float
            time to read budgets from, see read_budgets(). Default None
        xlim, ylim, zlim : tuple
            in physical domain coordinates, the slice limits.
            If an integer is given, then the dimension of the
            slice will be reduced by one. If None is given (default),
            then the entire domain extent is sliced.
        overwrite : bool
            Overwrites loaded budgets, see read_budgets(). Default False
        sel_kwargs : optional
            Additional keyword arguments to Dataset.sel()

        Returns
        -------
        GridDataset
            xarray of sliced field variables
        """

        if sl is not None:
            self.warn("Recommended usage: use sl.slice() instead")
            return sl.slice(xlim=xlim, ylim=ylim, zlim=zlim, keys=keys)

        # parse what field arrays to slice into
        if field_terms is not None:
            # read fields
            self.read_fields(field_terms=field_terms, tidx=tidx, time=time)
            preslice = self.field
            field_terms = [field_terms] if isinstance(field_terms, str) else field_terms
            keys = [term for term in field_terms if term in self.field.keys()]

        elif budget_terms is not None:
            # read budgets
            self.read_budgets(
                budget_terms=budget_terms, tidx=tidx, time=time, overwrite=overwrite
            )
            preslice = self.budget
            budget_terms = (
                [budget_terms] if isinstance(budget_terms, str) else budget_terms
            )
            keys = [term for term in budget_terms if term in self.budget.keys()]

        elif field is not None:
            raise NotImplementedError("Deprecated v0.2.0")

        else:
            self.printv(
                "BudgetIO.slice(): Both budget_terms() and field_terms() were None."
            )
            return None

        if len(keys) == 0:
            return None

        return preslice.slice(xlim=xlim, ylim=ylim, zlim=zlim, keys=keys, **sel_kwargs)

    def islice(
        self,
        budget_terms,
        x=None,
        y=None,
        z=None,
        xlim=None,
        ylim=None,
        zlim=None,
        tidx=None,
        make_meshgrid=True,
        overwrite=False,
    ):
        """
        Like slice, but interpolates using RegularGridInterpolator.
        And slower.

        Parameters
        ----------
        self : BudgetIO object
        budget_terms : list
            budget terms sent to _parse_budget_terms()

        """

        if all([xi is None for xi in [x, y, z]]):
            xid, yid, zid = self.get_xids(
                x=xlim, y=ylim, z=zlim, return_none=True, return_slice=True
            )
            x = self.xLine[xid]
            y = self.yLine[yid]
            z = self.zLine[zid]
        else:
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            z = np.atleast_1d(z)

        # read the requested budgets
        keys = self._parse_budget_terms(budget_terms)
        self.read_budgets(budget_terms=keys, tidx=tidx, overwrite=overwrite)

        # this is the dictionary that will be returned  #TODO: make this into a slice object
        ret = dict()

        # do the interpolation
        for key in keys:
            interp_obj = RGI(
                (self.xLine, self.yLine, self.zLine), np.array(self.budget[key])
            )  # hotfix

            if make_meshgrid:
                xiG = np.meshgrid(x, y, z, indexing="ij")
                xi = np.array([xG.ravel() for xG in xiG]).T
            else:
                xi = np.array([x, y, z]).T

            ret[key] = np.squeeze(interp_obj(xi).reshape(xiG[0].shape))

        # finish the slice... add extents and axes
        ret["x"] = x
        ret["y"] = y
        ret["z"] = z

        ext = []
        for term in ["x", "y", "z"]:
            if (
                ret[term].ndim > 0
            ):  # if this is actually a slice (not a number), then add it to the extents
                # if len(ret[term]) > 1:
                ext += [np.min(ret[term]), np.max(ret[term])]
        ret["extent"] = np.array(ext)
        ret["keys"] = keys

        return ret

    def get_xids(self, **kwargs):
        """
        Translates x, y, and z limits in the physical domain to
        indices based on self.xLine, self.yLine, and self.zLine

        Parameters
        ---------
        x, y, z : float or iterable (tuple, list, etc.)
            Physical locations to return the nearest index
        return_none : bool
            If True, populates output tuple with None if input is None.
            Default False.
        return_slice : bool
            If True, returns a tuple of slices instead a tuple of lists.
            Default False.

        Returns
        -------
        xid, yid, zid : list or tuple of lists
            Indices for the requested x, y, z, args in the order: x, y, z.
            If, for example, y and z are requested, then the
            returned tuple will have (yid, zid) lists.
            If only one value (float or int) is passed in for e.g. x,
            then an integer will be passed back in xid.
        """

        if not self.associate_grid:
            raise (AttributeError("No grid associated. "))

        # use object grid coordinates
        if "x_ax" not in kwargs or kwargs["x_ax"] is None:
            kwargs["x_ax"] = self.x
        if "y_ax" not in kwargs or kwargs["y_ax"] is None:
            kwargs["y_ax"] = self.y
        if "z_ax" not in kwargs or kwargs["z_ax"] is None:
            kwargs["z_ax"] = self.z

        return get_xids(**kwargs)

    def xy_avg(self, budget_terms=None, field_terms=None, zlim=None, **slice_kwargs):
        """
        xy-averages requested budget terms

        Parameters
        budget_terms : list
            Budget terms, see _parse_budget_terms()
        zlim : array-like
            z-limits for slicing
        slice_kwargs : Any
            Additional keyword arguments, see BudgetIO.slice()
        """
        # by default, take the whole x,y domain
        _slice_kwargs = dict(xlim=None, ylim=None, zlim=zlim)
        _slice_kwargs.update(slice_kwargs)

        to_merge = list(
            filter(
                None,
                [
                    self.slice(budget_terms=budget_terms, **_slice_kwargs),
                    self.slice(field_terms=field_terms, **_slice_kwargs),
                ],
            )
        )

        axes = [axis for axis in ["x", "y"] if axis in self.field.dims]
        if len(to_merge) == 0:
            return None
        elif len(to_merge) == 1:
            return to_merge[0].mean(axes)
        else:
            return xr.merge(to_merge).mean(axes)

    def unique_tidx(self, return_last=False, search_str="Run{:02d}.*_t(\d+).*.out"):
        """
        Pulls all the unique tidx values from a directory.

        Parameters
        ----------
        return_last : bool
            If True, returns only the largest value of TIDX. Default False.

        Returns
        -------
        t_list : array
            List of unique time IDs (TIDX)
        return_last : bool, optional
            if True, returns only the last (largest) entry. Default False
        search_str : regex, optional
            Regular expression for the search string and capture groups
        """

        if not self.associate_padeops:
            return None  # TODO - is this lost information? is it useful information?

        # retrieves filenames and parses unique integers, returns an array of unique integers
        filenames = self.dirname.glob("*")
        runid = self.runid

        # searches for the formatting *_t(\d+)* in all filenames
        t_list = [
            int(re.findall(search_str.format(runid), str(name))[0])
            for name in filenames
            if re.findall(search_str.format(runid), str(name))
        ]

        if len(t_list) == 0:
            raise FileNotFoundError("unique_tidx(): No files found")

        t_list.sort()

        if return_last:
            return t_list[-1]
        else:
            return np.unique(t_list)

    def unique_budget_tidx(self, return_last=True):
        """
        Pulls all the unique tidx values from a directory.

        Parameters
        ----------
        return_last : bool
            If False, returns only the largest TIDX associated with budgets. Else,
            returns an entire list of unique tidx associated with budgets. Default True

        Returns
        -------
        t_list : array
            List of unique budget time IDs (TIDX)
        return_last : bool, optional
            if True, reutnrs only the last (largest) entry. Default True
        """

        # TODO: fix for .npz
        if not self.associate_padeops:
            return None

        return self.unique_tidx(
            return_last=return_last, search_str="Run{:02d}.*budget.*_t(\d+).*"
        )

    def unique_times(self, return_last=False):
        """
        Reads the .out file of each unique time and returns an array of
        [physical] times corresponding to the time IDs from unique_tidx().

        Parameters
        ----------
        return_last : bool
            If True, returns only the largest time. Default False.

        Returns
        -------
        times : array
            list of times associated with each time ID in unique_tidx()
        """

        # TODO: fix for .npz
        if not self.associate_padeops:
            return None

        times = []

        if return_last:  # save time by only reading the final TIDX
            tidx = self.unique_tidx(return_last=return_last)
            fname = self.dirname / f"Run{self.runid:02d}_info_t{tidx:06d}.out"
            t = np.genfromtxt(fname, dtype=None)[0]
            return t

        for tidx in self.unique_tidx():
            fname = self.dirname / f"Run{self.runid:02d}_info_t{tidx:06d}.out"
            t = np.genfromtxt(fname, dtype=None)[0]
            times.append(t)

        return np.array(times)

    def match_budget_n(self, tidx=None):
        """Match budget n to budget tid"""

        if tidx is None:
            tidx = self.unique_budget_tidx(return_last=False)

        filenames = self.dirname.glob("*")
        search_str = "Run{:02d}.*budget.*_t{:06d}_n(\d+).*"

        # the following is not efficient, but sufficient for now
        n_list = []
        for tid in tidx:
            _n = [
                int(re.findall(search_str.format(self.runid, tid), str(name))[0])
                for name in filenames
                if re.findall(search_str.format(self.runid, tid), str(name))
            ]
            n_list.append(_n[0])

        return np.array(n_list)

    def last_budget_n(self, return_last=True):
        """
        Pulls all unique n from budget terms in a directory and returns the largest value.

        Parameters
        ----------
        return_last : bool
            If true, only returns the last (largest) element.

        Returns
        -------
        np.array
        """
        if not self.associate_padeops:
            return None

        return self.unique_tidx(
            return_last=return_last, search_str="Run{:02d}.*_n(\d+).*"
        )

    def existing_budgets(self):
        """
        Checks file names for which budgets were output.
        """
        filenames = self.dirname.glob("*")

        if self.associate_padeops:
            runid = self.runid
            # capturing *_budget(\d+)* in filenames
            budget_list = [
                int(re.findall("Run{:02d}.*_budget(\d+).*".format(runid), str(name))[0])
                for name in filenames
                if re.findall("Run{:02d}.*_budget(\d+).*".format(runid), str(name))
            ]
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

            budget_list = [BudgetIO.key[t][0] for t in t_list]

        if len(budget_list) == 0:
            self.warn("existing_budgets(): No associated budget files found. ")

        return list(np.unique(budget_list))

    def existing_terms(self, budget=None):
        """
        Checks file names for a particular budget and returns a list of all the existing terms.

        Arguments
        ---------
        budget : int
            optional, default None. If provided, searches a particular budget for existing terms.
            Otherwise, will search for all existing terms. `budget` can also be a list of integers.
                Budget 0: mean statistics
                Budget 1: momentum
                Budget 2: MKE
                Budget 3: TKE
                Budget 4: Reynolds Stresses

        Returns
        -------
        t_list (list) : list of tuples of budgets found

        """

        t_list = []

        # if no budget is given, look through all saved budgets
        if budget is None:
            budget_list = self.existing_budgets()

        else:
            # convert to list if integer is given
            if type(budget) != list:
                budget_list = [budget]
            else:
                budget_list = budget

        # find budgets by name matching with PadeOps output conventions
        if self.associate_padeops:
            tup_list = []
            # loop through budgets
            for b in budget_list:
                search_str = f"Run{self.runid:02d}_budget{b:01d}_term(\d+).*"
                terms = self.unique_tidx(search_str=search_str)
                tup_list += [((b, term)) for term in terms]  # these are all tuples

            # convert tuples to keys
            t_list = [BudgetIO.key.inverse[key][0] for key in tup_list]
        # find budgets matching .npz convention in write_npz()
        else:
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

            tup_list = [BudgetIO.key[t] for t in all_terms]  # list of associated tuples
            t_list = []  # this is the list to be built and returned

            for b in budget_list:
                t_list += [tup for tup in tup_list if tup[0] == b]

        # else:
        if len(t_list) == 0:
            self.warn("existing_terms(): No terms found for budget " + str(budget))

        return t_list

    def Read_x_slice(self, xid, field_terms=["u"], tidx_list=[]):
        """
        Reads slices of dumped quantities at a time ID or time IDs.

        Arguments
        ---------
        xid : int
            integer of xid dumped by initialize.F90. NOTE: Fortran indexing starts at 1.
        label_list : list
            list of terms to read in.
            Available is typically: "u", "v", "w", and "P" (case-sensitive)
        tidx_list : list
            list of time IDs.

        Returns
        -------
        sl : dict
            formatted dictionary similar to BudgetIO.slice()
        """

        sl = {}
        if type(field_terms) == str:
            field_terms = [field_terms]

        for tidx in tidx_list:
            for lab in field_terms:
                fname = (
                    self.dirname
                    / f"Run{self.runid:02d}_t{tidx:06d}_{'x':s}{xid:05d}.pl{lab:s}"
                )

                key_name = "{:s}_{:d}".format(lab, tidx)
                sl[key_name] = np.fromfile(
                    fname, dtype=np.dtype(np.float64), count=-1
                ).reshape((self.grid.ny, self.grid.nz), order="F")

        sl["x"] = self.x[[xid - 1]]
        sl["y"] = self.y
        sl["z"] = self.z

        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ["x", "y", "z"]:
            if (
                len(sl[term]) > 1
            ):  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(sl[term]), np.max(sl[term])]

        sl["extent"] = ext

        return sl

    def Read_y_slice(self, yid, field_terms=["u"], tidx_list=[]):
        """
        Reads slices of dumped quantities at a time ID or time IDs.

        Arguments
        ---------
        yid : int
            integer of yid dumped by initialize.F90
        label_list : list
            list of terms to read in.
            Available is typically: "u", "v", "w", and "P" (case-sensitive)
        tidx_list : list
            list of time IDs.

        Returns
        -------
        sl (dict) : formatted dictionary similar to BudgetIO.slice()
        """

        sl = {}
        if type(field_terms) == str:
            field_terms = [field_terms]

        for tidx in tidx_list:
            for lab in field_terms:
                fname = (
                    self.dirname
                    / f"Run{self.runid:02d}_t{tidx:06d}_{'y':s}{yid:05d}.pl{lab:s}"
                )

                key_name = "{:s}_{:d}".format(lab, tidx)
                sl[key_name] = np.fromfile(
                    fname, dtype=np.dtype(np.float64), count=-1
                ).reshape((self.grid.nx, self.grid.nz), order="F")

        sl["x"] = self.x
        sl["y"] = self.y[[yid - 1]]
        sl["z"] = self.z

        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ["x", "y", "z"]:
            if (
                len(sl[term]) > 1
            ):  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(sl[term]), np.max(sl[term])]

        sl["extent"] = ext  # TODO: make into Slice() object

        return sl

    def Read_z_slice(self, zid, field_terms=["u"], tidx_list=[]):
        """
        Reads slices of dumped quantities at a time ID or time IDs.

        Arguments
        ---------
        zid : int
            integer of zid dumped by initialize.F90
        label_list : list
            list of terms to read in.
            Available is typically: "u", "v", "w", and "P" (case-sensitive)
        tidx_list : list
            list of time IDs.

        Returns
        -------
        sl (dict) : formatted dictionary similar to BudgetIO.slice()
        """

        sl = {}
        if type(field_terms) == str:
            field_terms = [field_terms]

        for tidx in tidx_list:
            for lab in field_terms:
                fname = (
                    self.dirname
                    / f"Run{self.runid:02d}_t{tidx:06d}_{'z':s}{zid:05d}.pl{lab:s}"
                )

                key_name = "{:s}_{:d}".format(lab, tidx)
                sl[key_name] = np.fromfile(
                    fname, dtype=np.dtype(np.float64), count=-1
                ).reshape((self.grid.nx, self.grid.ny), order="F")

        sl["x"] = self.x
        sl["y"] = self.y
        sl["z"] = self.z[[zid - 1]]

        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ["x", "y", "z"]:
            if (
                len(sl[term]) > 1
            ):  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(sl[term]), np.max(sl[term])]

        sl["extent"] = ext

        return sl

    def _read_turb_file(self, prop, tid=None, turb=1):
        """
        Reads the turbine power from the output files

        Arguments
        ---------
        prop (str) : property string name, either 'power', 'uvel', or 'vvel'
        tidx (int) : time ID to read turbine power from. Default: calls self.unique_tidx()
        turb (int) : Turbine number. Default 1
        """
        if prop == "power":
            fstr = "Run{:02d}_t{:06d}_turbP{:02}.pow"
        elif prop == "uvel":
            fstr = "Run{:02d}_t{:06d}_turbU{:02}.vel"
        elif prop == "vvel":
            fstr = "Run{:02d}_t{:06d}_turbV{:02}.vel"
        else:
            raise ValueError(
                "_read_turb_prop(): `prop` property must be 'power', 'uvel', or 'vvel'"
            )

        if tid is None:
            try:
                tid = self.last_tidx
            except ValueError as e:  # TODO - Fix this!!
                tid = self.unique_tidx(return_last=True)

        fname = self.dirname / fstr.format(self.runid, tid, turb)
        self.printv("\tReading", fname)

        ret = np.genfromtxt(fname, dtype=float)  # read fortran ASCII output file

        # for some reason, np.genfromtxt makes a size 0 array for length-1 text files.
        # Hotfix: multiply by 1.
        ret = ret * 1

        return ret  # this is an array

    def read_turb_property(
        self, tidx, prop_str, turb=1, steady=None, dup_threshold=1e-12
    ):
        """
        Helper function to read turbine power, uvel, vvel. Calls self._read_turb_file()
        for every time ID in tidx.

        steady : bool, optional
            Averages results if True. If False, returns an array
            containing the contents of `*.pow`. Default None (True)
        """

        if not self.associate_padeops:  # read from saved files
            if self.associate_mat:
                fname = self.dirname / self.fname_meta.format("mat")
                tmp = loadmat(fname)

            elif self.associate_npz:
                fname = self.dirname / self.fname_meta.format("npz")
                tmp = np.load(fname)
            else:
                raise AttributeError("read_turb_property(): How did you get here? ")

            try:
                return np.squeeze(tmp[f"t{turb}_{prop_str}"])
            except KeyError as e:
                raise e

        # else: read from padeops:
        prop_time = []  # power array to return

        if tidx is None:
            tidx = [self.last_tidx]  # just try the last TIDX by default
        elif isinstance(tidx, str) and tidx == "all":
            tidx = self.unique_tidx(search_str="Run{:02d}.*_t(\d+).*.pow")

        if not hasattr(tidx, "__iter__"):
            tidx = np.atleast_1d(tidx)

        if steady == None:
            if len(tidx) > 1:
                steady = False  # assume if calling for more than 1 time ID, that steady is FALSE by default
            else:
                steady = True

        for tid in tidx:  # loop through time IDs and call helper function
            prop = self._read_turb_file(prop_str, tid=tid, turb=turb)
            if (
                type(prop) == np.float64
            ):  # if returned is not an array, cast to an array
                prop = np.array([prop])
            prop_time.append(prop)

        prop_time = np.concatenate(prop_time)  # make into an array

        # Upon starting budgets, there appear to be "close duplicates" which are not taken out by `np.unique`
        ids_remove = np.where(abs(np.diff(prop_time)) < dup_threshold)
        ret = np.delete(prop_time, ids_remove)

        if steady:
            return np.mean(ret)
        else:
            return ret

    def read_turb_power(self, tidx=None, **kwargs):
        """
        Reads the turbine power files output by LES in Actuator Disk type 2 and type 5.

        Parameters
        ----------
        tidx : list-like
            list or array of time IDs to load data. Default: self.last_tidx.
            If tidx = 'all', then this calls self.unique_tidx()
        **kwargs : dict
            see self._read_turb_file()
        """
        return self.read_turb_property(tidx, "power", **kwargs)

    def read_turb_uvel(self, tidx=None, **kwargs):
        """
        Reads turbine u-velocity.

        See self.read_turb_power() and self._read_turb_file()
        """
        return self.read_turb_property(tidx, "uvel", **kwargs)

    def read_turb_vvel(self, tidx=None, **kwargs):
        """
        Reads turbine v-velocity

        See self.read_turb_power() and self._read_turb_file()
        """
        return self.read_turb_property(tidx, "vvel", **kwargs)

    def get_logfiles(self, path=None, search_str="*.o[0-9]*", id=-1):
        """
        Searches for all logfiles formatted "*.o[0-9]" (Stampede3 format)
        and returns the entire list if `id` is None, otherwise returns
        the specific `id` requested.

        Parameters
        ----------
        path : path-like, optional
            Directory to search for logfiles. Default is self.dir_name
        search_str : str, optional
            Pattern to attempt to match. Default is "*.o[0-9]*"
        id : int, optional
            If multiple logfiles exist, selects this index of the list.
            Default is -1
        """
        path = path or self.dir_name
        return tools.get_logfiles(path, search_str=search_str, id=id)

    def get_ustar(self, logfile=None, crop_budget=True, average=True):
        """
        Gleans ustar from the logfile.

        Parameters
        ----------
        logfile : path-like, optional
            Path to logfile. If None, searches for all files ending in '.o[0-9]*'.
            Default is None.
        crop_budget : bool, optional
            Crops time axis to budgets. Defaults to True.
        average : bool, optional
            Time averages. Defaults to True.
        """
        return tools.get_ustar(
            self, search_str=logfile, crop_budget=crop_budget, average=average
        )

    def get_uhub(self, z_hub=0, use_fields=False, **slice_kwargs):
        """Compute the hub height velocity"""
        return tools.get_uhub(self, z_hub=z_hub, use_fields=use_fields, **slice_kwargs)

    def get_phihub(
        self, z_hub=0, return_degrees=False, use_fields=False, **slice_kwargs
    ):
        """Interpolate hub height wind direction (radians)."""
        return tools.get_phihub(
            self,
            z_hub=z_hub,
            return_degrees=return_degrees,
            use_fields=use_fields,
            **slice_kwargs,
        )

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
        return tools.get_timekey(self, budget=budget)

    def get_tidx_pairs(self, budget=False):
        """
        Returns a dictionary matching time keys [TIDX in PadeOps] to non-dimensional times.

        Arguments
        ----------
        self : BudgetIO object
        budget : bool
            If true, matches budget times from BudgetIO.unique_budget_tidx(). Default false.

        Returns
        -------
        (array, array)
            matching (TIDX, time) tuple
        """
        return tools.get_tidx_pairs(self, budget=budget)

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
        return tools.get_time_ax(
            self, return_tidx=return_tidx, missing_init_ok=missing_init_ok
        )

    def get_dt(self):
        """Computes a mean time step dt for a simulation"""
        return tools.get_dt(self)


if __name__ == "__main__":
    """
    TODO - add unit tests to class
    """
    print("padeopsIO: No unit tests included yet. ")
