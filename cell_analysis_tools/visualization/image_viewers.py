import matplotlib as mpl
import matplotlib.pylab as plt

mpl.rcParams["figure.dpi"] == 300
import numpy as np
from skimage.color import label2rgb

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.metrics import dice, total_error


def image_show(image):
    """

    Parameters
    ----------
    image : ndarray
        image to show.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(image)  # , cmap='gray'
    ax.axis("off")
    plt.show()
    return fig, ax


def compare_images(im1, title1, im2, title2, suptitle=None, figsize=(10, 5), save_path=None) -> None:
    """
     
        Parameters
        ----------
        im1 : np.ndarray
            image 1.
        title1 : str
            title for image 1.
        im2 : np.ndarray
            image 2.
        title2 : str
            title for image 2.
        figsize : TYPE, optional
            size of figure. The default is (10, 5).
    
        Returns
        -------
        None.
    
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle)
    ax[0].title.set_text(title1)
    ax[0].imshow(im1)
    ax[0].set_axis_off()

    ax[1].title.set_text(title2)
    ax[1].imshow(im2)
    ax[1].set_axis_off()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def compare_orig_mask_gt_pred(
    im: np.ndarray, mask_gt: np.ndarray, mask_pred: np.ndarray, title: str = ""
) -> None:
    """
    Simple function for comparing oringal image and ground truth image


    Parameters
    ----------
    im : np.ndarray
        original image.
    mask_gt : np.ndarray
        ground truth image.
    mask_pred : np.ndarray
        predicted mask.
    title : str, optional
        title of the plot, usually the origina filename. The default is "".

    Returns
    -------
    None
        function only just plots data.

    """
    alpha = 0.5
    im_overlay = label2rgb(
        mask_pred, normalize(im), bg_label=0, alpha=alpha, image_alpha=1, kind="overlay"
    )

    fig, ax = plt.subplots(2, 3, figsize=(10, 7))

    plt.suptitle(title)
    ax[0, 0].title.set_text(f"original")
    ax[0, 0].set_axis_off()
    ax[0, 0].imshow(im)

    # overlayed
    dice_coeff = dice(mask_pred, mask_gt)
    ax[0, 1].title.set_text(f"overlayed mask_pred")
    ax[0, 1].set_axis_off()
    ax[0, 1].imshow(im_overlay)

    # mask gt
    ax[1, 0].title.set_text(f"mask_gt")
    ax[1, 0].set_axis_off()
    ax[1, 0].imshow(mask_gt)

    # mask pred
    ax[1, 1].title.set_text(f"mask_pred \n dice: {dice_coeff:.4f}")
    ax[1, 1].set_axis_off()
    ax[1, 1].imshow(mask_pred)

    ## XOR
    mask_xor = np.logical_xor(mask_gt, mask_pred)

    error_total = total_error(mask_gt, mask_pred)
    ax[0, 2].title.set_text(f"mask_xor\n total error: {(error_total*100):.3f}")
    ax[0, 2].set_axis_off()
    ax[0, 2].imshow(mask_xor)

    ax[1, 2].set_axis_off()
    plt.show()


if __name__ == "__main__":

    import numpy as np

    im = np.random.rand(40, 40)
    compare_orig_mask_gt_pred(im, im, im)

    # print("TODO add test code")

    # im_orig = np.random.rand(512,512)
    # im_gt = np.round(im_orig)

    # compare_orig_mask_gt_pred(im_orig, im_gt, im_orig,"comparing originals")
