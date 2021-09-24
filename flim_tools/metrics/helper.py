import numpy as np

def _validate_array_and_make_bool(mask):
    """
    Function to validate that the mask used is boolean and that it has data
    """

    # convert to bool and remove 1D axis 
    mask = mask.astype(bool).squeeze()
    
    # check that it's 2d
    n_dims = len(mask.shape) 
    assert  n_dims == 2, f"Shape Error: mask should be 2D, {n_dims} dimensions found"
    
    # check that mask has data
    assert np.sum(mask) != 0, "Error: no data found in mask"
    
    return mask