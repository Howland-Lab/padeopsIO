"""
Namelist utilities for reading and writing Fortran90 namelists. 

Note: All namelists are read in lowercase and written to lowercase! This is 
done because Python is case-sensitive but Fortran is not. 
# TODO: Fix this to retain case in the future

Note 2: Comments are not saved when reading namelists. 

Kirby Heck
2023 Oct 10
"""

import re


def read(*args, **kwargs):
    """
    Alias for parser. See parser()
    """
    return parser(*args, **kwargs)


def write(*args, **kwargs):
    """
    Aslias for writer. See writer()
    """
    return writer(*args, **kwargs)


def parser(filename, to_lowercase=True):
    """
    Parses a namelist into nested dictionaries. Each key to the returned dictionary
    is the name of a namelist in `filename`, and the value of that key is itself a
    dictionary of namelist variables and associated values.

    Parameters
    ----------
    filename : path
        Path to a namelist text file
    to_lowercase : bool
        Casts keys to lowercase if True. Default True.

    Returns
    -------
    Namelists (nested dictionary)
    """

    Namelists = {}
    Namelist = {}  # active namelist
    active = None
    with open(filename, "r") as f:
        lines = f.readlines()

        # loop through lines
        for line in lines:
            res = re.search(
                r"^\s*&(\w+)", line
            )  # namelists start with "&" followed by a string
            if res is not None:  # starting new namelist
                active = res.group(1)  # save string in active list
                if to_lowercase:
                    active = active.lower()
                continue

            # within a namelist now; search `variable` = `value`
            res = re.search(r"(\S+)\s+=\s+(\S+)", line)
            if res is not None:
                # found variable/value pairing
                key = res.group(1)
                if to_lowercase:
                    key = key.lower()

                value = cast_str_to_X(res.group(2))

                Namelist[key] = value
                continue

            res = re.search(r"/", line)
            if res is not None:
                # Found end of Namelist
                if active is not None:
                    Namelists[active] = Namelist.copy()
                    Namelist = {}
                    active = None  # terminate namelist, start a new one

    return Namelists


def writer(filepath, Namelists):
    """
    Writes a namelist formatted in the same way that is output by nml_parser()

    Parameters
    ----------
    filepath : path
        Path where the namelist will be written
    Namelists : Nested dict from parser()

    Returns
    -------
    None
    """
    with open(filepath, "w") as f:  # begin writing a file
        for key in Namelists.keys():
            # write header
            f.write("&" + key + "\n")
            # loop thru variables
            for var in Namelists[key].keys():
                tmp = Namelists[key][var]
                value = cast_to_str(tmp)
                # write variable/value combinations
                f.write("{:" "<24} = {:s}\n".format(var, value))
            f.write("/\n")


def cast_str_to_X(value):
    """
    Tries to cast the input value to an integer, then a float.
    If all fail, then returns the original string.

    Parameters
    ----------
    value : str
        Value read in by the parser as a string

    Returns
    -------
    value : str or int or float
        Cast value
    """

    # try integer casting:
    try:
        tmp = int(value)
        return tmp
    except ValueError:
        pass

    # integer casting failed, try float
    try:
        tmp1 = re.sub(
            "[dD]", "e", value
        )  # Fortran uses d, D as well as e, E for exponential notation
        tmp2 = float(tmp1)
        return tmp2
    except ValueError:
        pass

    if value.lower() == ".true.":
        return True
    elif value.lower() == ".false.":
        return False

    char_delimiter = value[0]  # probably, this is a quotation mark "
    if value[-1] == char_delimiter:
        return value[1:-1]  # returns original value, sans quotes

    return value  # hopefully we don't get here


def cast_to_str(value):
    """
    Takes a value of unknown type (value) and casts to a fortran namelist-readable string.

    Parameters
    ----------
    value : str or float or int
        Variable value to write into the namelist

    Returns
    -------
    ret : str
        String representation of the value.
        For example:
            int(5)      ->  '5'
            'abc'       ->  '"abc"'
            float(5)    ->  '5.00000000e+00'  (single precision)
            bool(True)  ->  '.true.'
    """

    # note: this is NOT a try/except statement because we want floats (that are also integers)
    # to be parsed as floats, not integers.
    if type(value) == int:
        return str(value)

    if type(value) == bool:
        return f".{str(value).lower()}."

    try:
        tmp = float(value)
        return "{:.08e}".format(tmp)
    except ValueError:
        return f'"{value}"'  # return the original string, in double quotes
