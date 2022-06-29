import numpy as np

from .helper import _validate_array_and_make_bool


# https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
def dice(im1, im2, empty_score=1.0):
    """
    Computes the dice coefficient/F1 score, a measure of average similarity.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    empty-score : int
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    float
        Dice coefficient as a float on range [0,1]. \n
        Maximum similarity = 1 \n
        No similarity = 0 \n
        Both are empty (sum eq to zero) = empty_score
        
    Note
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = _validate_array_and_make_bool(im1)
    im2 = _validate_array_and_make_bool(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:  # no true values in either image
        print("warning: no true values in either array, returning empty_score=1")
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum
