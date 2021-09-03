import numpy as np
from scipy.spatial.distance import directed_hausdorff

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
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = _validate_array_and_make_bool(im1)
    im2 = _validate_array_and_make_bool(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0: # no true values in either image
        print("warning: no true values in either array, returning empty_score=1")
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum   


def jaccard(mask_gt, mask_pred):
    """
    Calculates the IoU/jaccard index for a pair of masks. 
    
    Logic is implemented to require images of the same size. 

    Parameters
    ----------
    mask_pred: np.ndarray
        Ground truth numpy ndarray image.
    mask_gt: np.ndarray
        Current image numpy ndarray.

    Returns
    -------
    jaccard_index: float
        Calculated Jaccard index/distance.
    """

    if mask_gt.shape != mask_pred.shape:
        raise ValueError(f"Shape mismatch: the shape of the ground truth mask {mask_gt.shape} does not match shape of predicted mask {mask_pred.shape}")

    if not mask_gt.any() and not mask_pred.any():
        raise ValueError('Ground truth mask and predicted mask cannot both be entirely zeros')

    mask_gt = _validate_array_and_make_bool(mask_gt)
    mask_pred = _validate_array_and_make_bool(mask_pred)

    return np.logical_and(mask_gt, mask_pred).sum() / np.logical_or(mask_gt, mask_pred).sum()


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


def total_error(mask_gt, mask_pred, weight_fn=1, weight_fp=1):
    """
    Computes the percent of misclassified pixels on an image. 
    In the case where one is more important than the 
    other, a weighted average may be used: c0FP + c1FN
    https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou

    
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
    fp = (mask_pred.astype(int) - mask_gt.astype(int)) > 0 # bool 
    fn = (mask_gt.astype(int) - mask_pred.astype(int)) > 0 # bool

    n_rows, n_cols = mask_gt.shape
    # return (np.sum(fn)*weight_fn + np.sum(fp)*weight_fp) / (n_rows * n_cols )
    return (np.sum(fn)*weight_fn + np.sum(fp)*weight_fp) / ((n_rows * n_cols )*((weight_fp+weight_fn)/2))


def average_relative_performance(mask_gt1, mask_gt2, mask_pred): 
    """
    Returns average relative performance of a predicted mask against 
    two ground truth masks from two different users
    
    Taken from mito hacker paper `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7642274/`

    ((Mito Catcher vs. User 1 + Mito Catcher vs. User 2)/(User 1 vs. User 2))
    
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
    float
        average performance score.

    """
    mask_gt1 = _validate_array_and_make_bool(mask_gt1)
    mask_gt2 = _validate_array_and_make_bool(mask_gt2)
    mask_predicted = _validate_array_and_make_bool(mask_pred)

    return (np.sum(np.logical_and(mask_predicted, mask_gt1)) +
            np.sum(np.logical_and(mask_predicted, mask_gt2))) \
            / (2 * np.sum((np.logical_and(mask_gt1, mask_gt2))))



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



def hausdorff_distance(mask_pred, mask_gt):
    """
    Calculates the Hausdorff distance for a given image, provided a ground truth. Both arrays must have the same number of columns. 
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
    Parameters
    ----------
    mask_pred : np.ndarray
        predicted segmentation mask
    mask_gt : np.ndarray
        ground truth mask

    Returns
    -------
        ddouble
            The directed Hausdorff distance between arrays u and v,
        index_1int
            index of point contributing to Hausdorff pair in u
        index_2int
            index of point contributing to Hausdorff pair in v

    """
    
    mask_pred = _validate_array_and_make_bool(mask_pred)
    mask_gt = _validate_array_and_make_bool(mask_gt)
    
    return directed_hausdorff(mask_pred, mask_gt)


if __name__ == "__main__":
    
    
    import matplotlib.pylab as plt
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 300
    
    size = 512
    mask_a = np.ones((size, size))
    mask_b = np.ones((size, size))
    
    mask_b[size//2:,...] = 0

    fix, ax = plt.subplots(1,2)
    ax[0].title.set_text("ones \n mask_a")
    ax[0].imshow(mask_a) 
    ax[1].title.set_text("top half are ones \n mask_b")
    ax[1].imshow(mask_b)
    plt.show()
    
    dice_coeff = dice(mask_a, mask_b)
    assert dice_coeff == (2/3), "incorrect dice score"
    
    jaccard_idx = jaccard(mask_a, mask_b)
    assert jaccard_idx == 0.5, "incorrect jaccard index"

    percent_error = total_error(mask_a, mask_b)
    assert percent_error == 0.5, "incorrect total error for masks"
    
    
    #### test of other metrics 
    w1=100
    w2=1
    mask_gt = np.zeros((10,10))
    mask_gt[:,:5] = 1
    
    mask_predicted = np.zeros((10,10))
    mask_predicted[:,:] = 1

    # total error 
    te = total_error(mask_gt, mask_predicted, weight_fn=1, weight_fp=1)
    assert te == 0.5, "incorrect total score"
    
    # average performance
    ## TODO check this, it's off
    mask_gt1 = np.zeros((10,10))
    mask_gt1[:,:5] = 1
    
    mask_gt2 = np.zeros((10,10))
    mask_gt2[:5,:] = 1
    
    mask_predicted = np.zeros((10,10))
    mask_predicted[:,:] = 1
    ap = average_relative_performance(mask_gt1, mask_gt2, mask_predicted)
    print(ap)
    

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
    
    return (np.sum(np.logical_and(mask_pred, mask_gt1)) +
            np.sum(np.logical_and(mask_pred, mask_gt2))) \
            / (np.sum(mask_gt1) + np.sum(mask_gt2)) # sum areas