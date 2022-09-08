import numpy as np


def _validate_array_and_make_bool(mask):
    """
    Function to validate that the mask used is boolean and that it has data
    """

    # convert to bool and remove 1D axis
    mask = mask.astype(bool).squeeze()

    # check that it's 2d
    n_dims = len(mask.shape)

    if n_dims != 2:
        raise ValueError(f"Shape Error: mask should be 2D, {n_dims} dimensions found")

    if np.sum(mask) == 0:
        # check that mask has data
        raise ValueError("Error: no data found in mask")

    return mask
