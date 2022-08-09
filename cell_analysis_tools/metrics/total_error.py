import numpy as np

from .helper import _validate_array_and_make_bool


def total_error(mask_gt, mask_pred, weight_fn=1, weight_fp=1):
    """
    Computes the percent of misclassified pixels on an image. 
    In the case where one is more important than the 
    other, a weighted average may be used: c0FP + c1FN
    
    'https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou <https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou>'_

    
    Parameters
    ----------
    mask_gt : bool ndarray
        groudn truth mask.
    mask_predicted : bool ndarray
        DESCRIPTION.
    weight_fn : int, optional
        scalar weight applied to false negatives. The default is 1.
    weight_fp : int, optional
        scalar weight applied to false positives. The default is 1.

    Returns
    -------
    percent: float
        total error of the predicted mask given a ground truth mask.

    """

    # convert to bool
    mask_gt = _validate_array_and_make_bool(mask_gt)
    mask_pred = _validate_array_and_make_bool(mask_pred)

    # subtraction operation requires arrays to be int
    # > 0 then convers then back to bool
    fp = (mask_pred.astype(int) - mask_gt.astype(int)) > 0  # bool
    fn = (mask_gt.astype(int) - mask_pred.astype(int)) > 0  # bool

    n_rows, n_cols = mask_gt.shape
    # return (np.sum(fn)*weight_fn + np.sum(fp)*weight_fp) / (n_rows * n_cols )
    return (np.sum(fn) * weight_fn + np.sum(fp) * weight_fp) / (
        (n_rows * n_cols) * ((weight_fp + weight_fn) / 2)
    )
