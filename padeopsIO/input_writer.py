"""
Helper functions for writing input files

Anticipated usage: 

```
from padeopsIO.input_writer import write_neutral

inputs = dict(...)  # list of user-defined inputs to populate template
write_neutral(inputs, n_hrs=4)
```

Kirby Heck
2025 January 19
"""

from pathlib import Path
import numpy as np
import jinja2
import itertools
import re
import warnings

BASE = Path(__file__).parent
TEMPLATE_NEUTRAL = BASE / "templates" / "input_spinup.j2"
TEMPLATE_UPSAMPLE = BASE / "templates" / "input_upsample.j2"
TEMPLATE_ROTATE = BASE / "templates" / "input_rotate.j2"
TEMPLATE_CONCURRENT = BASE / "templates" / "input_main.j2"
TEMPLATE_PRIMARY = BASE / "templates" / "input_primary.j2"
TEMPLATE_PRECURSOR = BASE / "templates" / "input_precursor.j2"
TEMPLATE_LAMINAR = BASE / "templates" / "input_laminar.j2"
TEMPLATE_HIT = BASE / "templates" / "input_hit.j2"
TEMPLATE_INTERACT = BASE / "templates" / "input_interact.j2"
TEMPLATE_DEFICIT = BASE / "templates" / "input_interact_deficit.j2"
TEMPLATE_SUBMIT = BASE / "templates" / "submit.j2"
TEMPLATE_TURBINE = BASE / "templates" / "ActuatorDisk_0001_input.j2"

CORES_PER_NODE = 48  # SKX nodes on Stampede3
DOF_PER_CORE = 32**3 * 2  # ~approx recommendation for PadeOps


def safe_mkdir(inputs, quiet=False, dst=None, key="dirname"):
    """Create output directory without overwriting files"""
    if dst is None:
        output_dir = Path(inputs[key])
    else:
        output_dir = dst

    output_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print("Writing files to", output_dir.resolve())

    return output_dir


def check_keys(inputs, keys):
    """
    Checks to make sure `inputs.keys()` contains all of the keys
    in `keys`
    """
    for key in keys:
        if key not in inputs.keys():
            raise ValueError(f'Missing key in `inputs`: "{key}".')

    return


def find_last_restart(inputs, return_frameangle=True):
    """
    Finds the final restart file and gleans the TID
    and frame angle, if requested
    """
    basedir = Path(inputs["restart_dir"])
    RID = inputs["restart_rid"]
    restart_files = basedir.glob(f"RESTART_Run{RID:02d}_info*")

    # find the last restart file; largest TID
    filename = None
    tid = -1
    for file in restart_files:
        new_tid = int(
            re.findall(r"info.(\d+)$", str(file))[0]
        )  # glean the TID from the string
        if new_tid > tid:
            tid = new_tid
            filename = file

    if tid == 0:
        # this did not find any files
        warnings.warn(
            "find_last_restart(): no restart files found, defaulting to TID 0"
        )

    if not return_frameangle:
        return tid
    else:
        print(filename)
        data = np.genfromtxt(filename, dtype=None)
        if len(data) < 2:
            frameangle = 0
        else:
            frameangle = -data[1]

        return tid, frameangle


def get_nnodes(inputs):
    """
    Computes the ideal number of nodes for PadeOps

    Rounds to an even number
    """
    nx = inputs["nx"]
    ny = inputs["ny"]
    nz = inputs["nz"]

    # compute n_nodes; round to nearest even number
    n_nodes = 2 * np.round(nx * ny * nz / DOF_PER_CORE / CORES_PER_NODE / 2)
    return n_nodes


def get_iterator(return_names=True, **kwargs):
    """
    Return an iterator of keyword argument values using itertools.product()
    """

    # get just the argument values so we can turn them into an iterator:
    args = list(kwargs.values())
    for k, arg in enumerate(args):
        if not hasattr(arg, "__iter__"):
            args[k] = [arg]  # make sure we can iterate over singleton elements

    iter_vars = itertools.product(*args)

    if return_names:
        # we also need to name each directory in the sweep
        names = get_iter_names(**kwargs)

        return iter_vars, names

    return iter_vars  # return just the variables if return_names is False


def get_iter_names(**kwargs):
    """
    Enumerate along len(kwargs) axes.
    """
    num_args = [
        np.arange(0, len(val))
        for key, val in kwargs.items()
        if hasattr(val, "__iter__")
    ]
    prod_args = itertools.product(*num_args)

    # build the directory "name" string:
    fname = ""
    for key, val in kwargs.items():
        if hasattr(val, "__iter__") and len(kwargs[key]) > 1:
            fname += (
                key + "_{:02d}_"
            )  # create string to be formatted: fname.format(*name)

    names = [fname[:-1].format(*ids) for ids in prod_args]  # write filenames
    return names


# def iter_to_df(**kwargs):
#     cols = list(kwargs.keys())
#     product, names = get_iterator(return_names=True, **kwargs)
#     df = pl.DataFrame(product, schema=cols)
#     df = df.with_columns(pl.Series("name", names))

#     return df.select(["name"] + cols)  # rearrange so `name` is the first column


def write_laminar(
    inputs,
    dst=None,
    quiet=False,
    inputfile_name="input_laminar.dat",
    n_hrs=2,  # allocated time, in hours
    node_min=1,
):
    """
    Write input file for spinup simulation

    Parameters
    ----------
    inputs : dict
        Dictionary of LES inputs
    dst : Path, optional
        Destination of written files. If none, defaults to inputs['dirname']
    quiet : bool, optional
        Silences print statements. Default False
    inputfile_name : str, optional
        String to title new input file
    """

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)

    # load spinup template and write template:
    with open(TEMPLATE_LAMINAR, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_name, "w") as f:
        f.write(out)

    # write turbines
    write_turbine(inputs, quiet=quiet, dst=dst)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs,
                inputfile_name,
                problem_name="AD_coriolis_shear",
                n_hrs=n_hrs,
                node_min=node_min,
            )
        )

    if not quiet:
        print("\tDone writing laminar inflow files")


def write_interact(
    inputs,
    dst=None,
    quiet=False,
    inputfile_names=["input_ad.dat", "input_hit.dat", "input_interact.dat"],
    n_hrs=2,  # allocated time, in hours
    node_min=1,
):
    """
    Write input file for spinup simulation

    Parameters
    ----------
    inputs : dict
        Dictionary of LES inputs
    dst : Path, optional
        Destination of written files. If none, defaults to inputs['dirname']
    quiet : bool, optional
        Silences print statements. Default False
    inputfile_name : str, optional
        String to title new input file
    """

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)

    # load laminar template and write:
    with open(TEMPLATE_LAMINAR, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_names[0], "w") as f:
        f.write(out)

    # load HIT template and write:
    with open(TEMPLATE_HIT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_names[1], "w") as f:
        f.write(out)

    # load CONCURRENT template and write:
    with open(TEMPLATE_INTERACT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_names[2], "w") as f:
        f.write(out)

    # write turbines
    write_turbine(inputs, quiet=quiet, dst=dst)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs,
                inputfile_names[-1],
                problem_name="HIT_shear",
                n_hrs=n_hrs,
                node_min=node_min,
            )
        )

    if not quiet:
        print("\tDone writing HIT turbulent inflow files")


def write_deficit(
    inputs,
    dst=None,
    quiet=False,
    inputfile_names=[
        "input_ad.dat",
        "input_empty.dat",
        "input_hit.dat",
        "input_interact.dat",
    ],
    n_hrs=4,  # allocated time, in hours
    node_min=1,
):
    """
    Write HIT deficit problem input files

    Parameters
    ----------
    inputs : dict
        Dictionary of LES inputs
    dst : Path, optional
        Destination of written files. If none, defaults to inputs['dirname']
    quiet : bool, optional
        Silences print statements. Default False
    inputfile_names : list of str, optional
        String to title new input file
    """

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)
    if not inputs["do_deficit_budgets"]:
        print("WARNING: running HIT_AD_deficit with deficit budgets OFF")

    # load PRIMARY template and write:
    with open(TEMPLATE_LAMINAR, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)
    out = template.render(inputs)
    with open(OUTPUT / inputfile_names[0], "w") as f:
        f.write(out)

    # load PRECURSOR template and write:
    with open(TEMPLATE_LAMINAR, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)
    # turn off wind turbines
    _precursor = inputs.copy()
    _precursor.update(usewindturbines=False, runid=2)
    out = template.render(_precursor)
    with open(OUTPUT / inputfile_names[1], "w") as f:
        f.write(out)

    # load HIT template and write:
    with open(TEMPLATE_HIT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)
    out = template.render(inputs)
    with open(OUTPUT / inputfile_names[2], "w") as f:
        f.write(out)

    # load interact template and write:
    with open(TEMPLATE_DEFICIT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)
    out = template.render(inputs)
    with open(OUTPUT / inputfile_names[3], "w") as f:
        f.write(out)

    # write turbines
    write_turbine(inputs, quiet=quiet, dst=dst)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs,
                inputfile_names[-1],
                problem_name="HIT_AD_deficit",
                n_hrs=n_hrs,
                node_min=node_min,
            )
        )

    if not quiet:
        print("\tDone writing HIT turbulent inflow files")


def write_hit(
    inputs,
    dst=None,
    quiet=False,
    inputfile_name="input_hit.dat",
    n_hrs=2,  # allocated time, in hours
    node_min=1,
):
    """
    Write input file for HIT simulation.

    Parameters
    ----------
    inputs : dict
        Dictionary of LES inputs
    dst : Path, optional
        Destination of written files. If none, defaults to inputs['dirname']
    quiet : bool, optional
        Silences print statements. Default False
    inputfile_name : str, optional
        String to title new input file
    """

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)

    # load spinup template and write template:
    with open(TEMPLATE_HIT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_name, "w") as f:
        f.write(out)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs,
                inputfile_name,
                problem_name="HIT_Periodic_moving",
                n_hrs=n_hrs,
                node_min=node_min,
            )
        )

    if not quiet:
        print("\tDone writing HIT input files")


def write_neutral(
    inputs,
    dst=None,
    quiet=False,
    inputfile_name="input_neutral.dat",
    n_hrs=24,  # allocated time, in hours
):
    """
    Write input file for spinup simulation

    Parameters
    ----------
    inputs : dict
        Dictionary of inputs, including:
            nx, ny, nz
            dirname,
            tstop,
            Lx, Ly (optional)
    dst : Path, optional
        Destination of written files. If none, defaults to inputs['dirname']
    quiet : bool, optional
        Silences print statements. Default False
    inputfile_name : str, optional
        String to title new input file
    """

    # check existing keys
    keys = [
        "nx",
        "ny",
        "nz",
        "dirname",
        "Ro",
        "Fr",
        "lat",
        "z0",
        "Lx",
        "Ly",
        "Lz",
        "do_budgets",
        "time_budget_start",
    ]
    check_keys(inputs, keys)

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)

    # load spinup template and write template:
    with open(TEMPLATE_NEUTRAL, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_name, "w") as f:
        f.write(out)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs, inputfile_name, problem_name="neutral_pbl", n_hrs=n_hrs
            )
        )

    if not quiet:
        print("\tDone writing spinups files")


def write_upsample(
    inputs,
    dst=None,
    quiet=False,
    inputfile_name="input_upsample.dat",
):
    """
    Writes upsampled fields inputs

    nx, ny in `inputs` are the FINAL number of points in x and y, after upsampling
    """

    # check existing keys
    keys = ["nx", "ny", "nz", "dirname", "restart_rid"]
    check_keys(inputs, keys)

    # update inputs
    update_upsample_inputs(inputs)

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)

    # load spinup template and write template:
    with open(TEMPLATE_UPSAMPLE, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_name, "w") as f:
        f.write(out)

    if not quiet:
        print("\tDone writing upsampling files")


def update_upsample_inputs(inputs):
    """Glean the restart TID from the restart RID"""
    TID = find_last_restart(inputs, return_frameangle=False)
    inputs["restart_tid"] = TID
    return  # updates dictionary, returns nothing


def write_rotate(
    inputs,
    dst=None,
    quiet=False,
    inputfile_name="input_rotate.dat",
    n_hrs=6,
):
    """
    Writes inputs for the rotation phase
    """
    # check existing keys
    keys = [
        "nx",
        "ny",
        "nz",
        "dirname",
        "restart_dir",
        "Ro",
        "Fr",
        "lat",
        "z0",
        "z_ref",
        "Lx",
        "Ly",
        "Lz",
    ]
    check_keys(inputs, keys)

    # adds frame angle line to RESTART file
    prep_rotation(inputs, quiet=quiet)

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)

    # load spinup template and write template:
    with open(TEMPLATE_ROTATE, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_name, "w") as f:
        f.write(out)

    # make submit.sh file
    with open(OUTPUT / "submit_rotation.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs, inputfile_name, problem_name="neutral_pbl", n_hrs=n_hrs
            )
        )

    if not quiet:
        print("\tDone writing rotation files")


def prep_rotation(inputs, rid=2, tid=0, frameangle=0.0, quiet=False):
    """Appends frame angle line to rotation restart files"""
    fname = Path(inputs["restart_dir"]) / f"RESTART_Run{rid:02d}_info.{tid:06d}"

    try:
        data = np.genfromtxt(fname, dtype=None)
        if len(np.atleast_1d(data)) > 1:
            return  # frame angle line already exists
    except FileNotFoundError:
        raise

    with open(fname, "a") as src:
        src.write(f"{frameangle:11.1f}")

    if not quiet:
        print("Added frame angle to retart files for rotation phase")


def write_concurrent(
    inputs,
    dst=None,
    quiet=False,
    n_hrs=24,
):
    """
    Write concurrent files, main function. Calls helper function
    _write_concurrent() after gleaning the restart TID and frame angle
    from restart files.
    """

    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst)
    # finish populating restart_tid, frame_angle fields
    tid, frameangle = find_last_restart(inputs, return_frameangle=True)
    inputs.update(frameangle=frameangle, restart_tid=tid)

    for _template, name in zip(
        [TEMPLATE_PRIMARY, TEMPLATE_PRECURSOR, TEMPLATE_CONCURRENT],
        ["primary", "precursor", "main"],
    ):
        # load spinup template and write template:
        with open(_template, "r") as f:
            template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

        # render output
        out = template.render(inputs)
        with open(OUTPUT / f"input_{name}.dat", "w") as f:
            f.write(out)

        if not quiet:
            print(f"\tDone writing {name} files")

    # make submit.sh file
    with open(OUTPUT / "submit_concurrent.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs,
                "input_main.dat",
                problem_name="neutral_pbl_concurrent",
                n_hrs=n_hrs,
            )
        )


def write_turbine(inputs, quiet=False, dst=None, turbnum=1):
    """Writes a turbine file by copying ActuatorDisk_0001_input.j2"""
    # make output directory
    OUTPUT = safe_mkdir(inputs, quiet=quiet, dst=dst, key="turbine_dir")

    with open(TEMPLATE_TURBINE, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / f"ActuatorDisk_{turbnum:04d}_input.inp", "w") as f:
        f.write(out)

    if not quiet:
        print(f"\tDone writing ActuatorDisk_{turbnum:04d}_input.inp file")


def sbatch_prep_args(
    inputs,
    name="input.dat",
    problem_name="AD_coriolis_shear",
    node_cap=128,
    node_min=1,
    n_hrs=4,
):
    """
    Returns a dictionary to populate fields in submit.j2 template

    Parameters
    ----------
    inputs : dict
        Dictionary, including the following fields:
            nx, ny, dirname
    name : str
        Name of input file to run
    problem_name : str, optional
        Name of problem file. Defaults to 'AD_coriolis_shear'
    node_cap : int, optional
        Maximum number of nodes. Defaults to 128
    n_hrs : int, optional
        Number of wall clock hours. Defaults to 4

    Returns
    -------
    dict : ret
        Dictionary to populate the submit file template
    """

    n_hrs = int(n_hrs)
    n_nodes = get_nnodes(inputs)

    # fill out submit file
    if problem_name == "neutral_pbl_concurrent":
        ret = dict(
            n_nodes=n_nodes * 2,
            n_hrs=n_hrs,
            problem_dir="turbines",
            problem_name=problem_name,
        )
    elif problem_name in ["neutral_pbl", "HIT_Periodic_moving"]:
        ret = dict(
            n_nodes=n_nodes,
            n_hrs=n_hrs,
            problem_dir="incompressible",
            problem_name=problem_name,
        )
    elif problem_name == "AD_coriolis_shear":
        ret = dict(
            n_nodes=n_nodes,
            n_hrs=n_hrs,
            problem_dir="turbines",
            problem_name=problem_name,
        )
    elif problem_name == "HIT_shear":
        ret = dict(
            n_nodes=n_nodes * 1.25,
            n_hrs=n_hrs,
            problem_dir="turbines",
            problem_name=problem_name,
        )
    elif problem_name == "HIT_AD_deficit":
        ret = dict(
            n_nodes=n_nodes * 2.25,
            n_hrs=n_hrs,
            problem_dir="turbines",
            problem_name=problem_name,
        )
    else:
        raise NotImplementedError
    ret["n_nodes"] = int(np.max([np.min([ret["n_nodes"], node_cap]), node_min]))
    ret["inputfile_name"] = name
    ret["dirname"] = inputs["dirname"]
    return ret


def sbatch_write_file(
    inputs,
    name="input.dat",
    problem_name="neutral_pbl",
    node_cap=128,
    node_min=1,
    n_hrs=24,
):
    """Returns a filled out submit.sh template"""
    # laod template
    with open(TEMPLATE_SUBMIT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # fill out submit file data
    submit_data = sbatch_prep_args(
        inputs,
        name,
        problem_name=problem_name,
        node_cap=node_cap,
        node_min=node_min,
        n_hrs=n_hrs,
    )
    return template.render(submit_data)


if __name__ == "__main__":
    # test writing files to tmp
    pass  # TODO
