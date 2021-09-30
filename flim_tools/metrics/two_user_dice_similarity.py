import numpy as np
from .helper import _validate_array_and_make_bool


def two_user_dice_similarity(mask_gt1, mask_gt2, mask_pred):
    """
    Variation on average_relative_performance but areas are added in the
    denominator

    Parameters
    ----------
    mask_gt1 : ndarray
        ground truth mask 1.
    mask_gt2 : ndarray
        ground truth mask 2.
    mask_predicted : ndarray
        predicted mask.

    Returns
    -------
    dice score: float
        two user average performance score, dice like.

    """
    # from above but add mask areas as above
    # ((Mito Catcher vs. User 1 + Mito Catcher vs. User 2)/(User 1 vs. User 2))

    return (
        np.sum(np.logical_and(mask_pred, mask_gt1))
        + np.sum(np.logical_and(mask_pred, mask_gt2))
    ) / (
        np.sum(mask_gt1) + np.sum(mask_gt2)
    )  # sum areas
