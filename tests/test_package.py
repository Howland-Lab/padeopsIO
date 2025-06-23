"""
Check to make sure the package imports are correct
"""


def test_full_import():
    import padeopsIO as pio


def test_io_imports():
    from padeopsIO.utils.io_utils import (
        key_search_r,
        query_logfile,
        structure_to_dict,
        deserialize_cross_platform_paths,
        get_unique_ids,
    )
    from padeopsIO import BudgetIO, DeficitIO


def test_budget_imports(): 
    from padeopsIO.budgetkey import get_key
    from padeopsIO.budget import Budget
    from padeopsIO.budget_addons import (
        NewBudget, 
        RANSBudget, 
        LESMomentum, 
        BudgetMKE, 
        BudgetVorticity
    )


def test_grid_imports():
    from padeopsIO.gridslice import GridDataset


def test_turbine_imports():
    from padeopsIO.turbineArray import Turbine, TurbineArray
