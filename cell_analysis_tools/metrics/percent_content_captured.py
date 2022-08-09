import numpy as np
from .helper import _validate_array_and_make_bool


def percent_content_captured(mask_a, mask_b):
    """
    Computes the content of mask_a captured in mask_b and returns it as a percent. (A intersect B)/A
    # From the paper `Mitohacker <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7642274/>`

    Parameters
    ----------
    mask_a : ndarray
        boolean mask.
    mask_b : ndarray
        boolean mask.

    Returns
    -------
    percent: float
        percent of mask_a content captured by mask_b.

    """
    mask_a = _validate_array_and_make_bool(mask_a)
    mask_b = _validate_array_and_make_bool(mask_b)

    return np.sum(np.logical_and(mask_a, mask_b)) / np.sum(mask_a)
