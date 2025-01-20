# Additional IO functions (some work-in-progress)
# USAGE: from io_utils import *

import re
import numpy as np


def key_search_r(nested_dict, key):
    """
    Copied from budgetkey.py

    Performs a recursive search for the dictionary key `key` in any of the dictionaries contained
    inside `nested_dict`. Returns the value of nested_dict[key] at the first match.

    Parameters
    ----------
    nested_dict (dict-like) : dictionary [possibly of dictionaries]
    key (str) : dictionary key to match

    Returns
    -------
    nested_dict[key] if successful, None otherwise.
    """

    try:
        for k in nested_dict.keys():
            if k == key:
                return nested_dict[k]
            else:
                res = key_search_r(nested_dict[k], key)
                if res is not None:
                    return res

    except AttributeError as e:
        return


def query_logfile(
    filename,
    search_terms=["TIDX"],
    fsearch=r"({:s}).*\s+([-+]?(\d+(\.\d*)?|\.\d+)([dDeE][-+]?\d+)?)",
    maxlen=None,
    crop_equal=True,
):
    """
    Queries the PadeOps output log file for text lines printed out by temporalhook.F90.

    By default, the search looks for TERM followed by any arbitrary characters, then at least 1
    character of white space followed by a number of format %e (exponential). Casts the resulting
    string into a float.

    Parameters
    ----------
    filename (path) : string or path to an output log file.
    search_terms (list) : list of search terms. Default: length 1 list ['TIDX'].
        Search terms are case sensitive.
    fsearch (string) : string format for regex target.
    maxlen (int) : maximum length of return lists. Default: None
    crop_equal (bool) : crops all variables to the same length if True. Default: True

    Returns
    -------
    ret (dict) : dictionary of {search_term: [list of matched values]}
    """

    ret = {key: [] for key in search_terms}

    search_str = ""
    # build the regex match
    for term in search_terms:
        search_str += term + "|"

    search = fsearch.format(search_str[:-1])  # do not include last pipe in search_str

    with open(filename, "r") as f:
        lines = f.readlines()

        for k, line in enumerate(lines):
            match = re.search(search, line)
            if match is not None:
                key = match.groups()[0]  # this was the matched keyword

                # before appending, check to make sure we haven't exceeded maxlen
                if maxlen is not None and (len(ret[key]) >= maxlen):
                    break

                # append the associated matched value, cast into a float
                ret[key].append(float(match.groups()[1]))

    # convert lists to array:
    if crop_equal:
        # crop all array elements to the last N items
        N = min(len(ret[key]) for key in ret.keys() if len(ret[key]) > 0)
        to_return = dict()
        for (
            key
        ) in (
            ret.keys()
        ):  # {key: array(ret[key][-N:]) for key in ret.keys() if len(ret[key]) > 0}
            to_return[key] = (
                np.array(ret[key][-N:]) if len(ret[key]) > 0 else np.full(N, np.nan)
            )
        return to_return

    else:
        return {key: np.array(ret[key]) for key in ret.keys()}


def structure_to_dict(arr):
    """
    Function to convert a numpy structured array to a nested dictionary.

    Numpy structured arrays are the form in which scipy.io.savemat saves .mat files
    and also how scipy.io.loadmat loads .mat files.

    See also:
    https://docs.scipy.org/doc/numpy-1.14.0/user/basics.rec.html
    """
    keys = arr.dtype.names
    if keys is None:
        raise TypeError(
            "structure_to_dict(): `ndarray` argument is not a structured datatype"
        )
    ret = {}

    for key in keys:
        val = arr[key][0][0]

        if type(val) == np.ndarray and val.dtype.names is not None:
            # recursive call
            val = structure_to_dict(val)
            ret[key] = val  # store the dictionary
        else:
            if len(val.flat) == 1:
                ret[key] = val.flat[0]  # store the key/value pairing
            else:
                ret[key] = [item for item in val.flat]  # store the key/value pairing

    return ret  # return (nested) dict
