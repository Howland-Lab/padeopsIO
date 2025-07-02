import padeopsIO as pio
import numpy as np
from pathlib import Path


data_path = Path(__file__).parent


def test_npz():
    # Test reading a .npz file
    sim = pio.BudgetIO(data_path, filename="example", npz=True, verbose=False)

    # Add some basic assertions to verify the data was loaded correctly
    assert sim is not None, "BudgetIO should return a valid object"

    # try to load data
    assert type(sim.grid.shape) == tuple
    assert type(sim.input_nml) == dict
    assert sim.associate_budgets
    assert sim.associate_grid
    assert sim.associate_turbines
    assert sim.filename == "example"
    assert len(sim.existing_terms()) > 0
    assert len(sim.ta.turbines) == 2  # two turbines
    assert type(sim.read_turb_power(turb=1)) == np.ndarray


if __name__ == "__main__":
    test_npz()
    print("Test completed successfully.")
