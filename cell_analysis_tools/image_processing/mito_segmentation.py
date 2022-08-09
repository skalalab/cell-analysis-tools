from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt
import tifffile

mpl.rcParams["figure.dpi"] = 300
import re

import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from skimage import morphology
from sklearn.model_selection import ParameterGrid

from cell_analysis_tools.image_processing import bin_im, kmeans_threshold, normalize
from cell_analysis_tools.metrics import dice

if __name__ == "__main__":

    # load image
    path_images = Path(
        r"Z:\0-Projects and Experiments\TQ - AAG - mitochondria-actin segmentation\210205_GoldStandard_NADH_images".replace(
            "\\", "/"
        )
    )
    list_path_images = list(path_images.glob("*.tif"))
    path_image = list_path_images[0]
    im = tifffile.imread(path_image)
    # plt.imshow(im)
    # plt.show()

    # load mask
    path_masks = Path(
        r"Z:\0-Projects and Experiments\TQ - AAG - mitochondria-actin segmentation\210111_GoldStandard_Stain_Masks".replace(
            "\\", "/"
        )
    )
    path_masks_aag = path_masks / "AAG-masks"
    path_masks_tq = path_masks / "TQ-masks"
    list_path_masks_aag = list(path_masks_aag.glob("*.tif"))
    list_path_masks_tq = list(path_masks_tq.glob("*.tif"))

    list_path_masks_aag_str = [str(p) for p in list_path_masks_aag]
    list_path_masks_tq_str = [str(p) for p in list_path_masks_tq]

    # string to find
    handle_nadh = path_image.stem

    if handle_nadh.split("_", 1)[0] == "2020":
        handle_tmre_prefix = handle_nadh.rsplit("_", 3)[0]
        handle_tmre_suffix = handle_nadh.split("_", 5)[-1]
        handle_tmre = handle_tmre_prefix + "_TMRE_" + handle_tmre_suffix

    # mask
    path_mask_aag = list(
        filter(re.compile(f"{handle_tmre}").search, list_path_masks_aag_str)
    )[0]
    path_mask_tq = list(
        filter(re.compile(f"{handle_tmre}").search, list_path_masks_tq_str)
    )[0]

    mask_aag = tifffile.imread(path_mask_aag)
    mask_tq = tifffile.imread(path_mask_tq)

    # normalize
    im = normalize(im)

    list_bin_sizes = np.arange(0, 10, step=1)
    list_disk_radius = np.arange(0, 10, step=1)

    param_grid = {
        "bin_size": list_bin_sizes,
        "selem_radius": list_disk_radius,
        # "kmeans_"
    }
    parameters = list(ParameterGrid(param_grid))

    # dice score placeholder --> bin_sizes in rows, disk radius in cols
    array_dice_scores_aag = np.zeros((len(list_bin_sizes), len(list_disk_radius)))
    array_dice_scores_tq = np.zeros((len(list_bin_sizes), len(list_disk_radius)))

    #%%

    for pos, params in enumerate(parameters, start=1):
        print(f"iteration {pos}/{len(parameters)}")
        bin_size = params["bin_size"]
        selem_radius = params["selem_radius"]

        # tophat with bin size zero doesn't make sense
        # if bin_size == 0:
        #     continue

        # bin_size=2
        kernel_size = bin_size * 2 + 1
        im_binned = bin_im(im, bin_size)
        # plt.imshow(im_binned)
        # plt.show()

        # tophat for bg equalization and edge sharpening
        # selem_radius = 11
        selem = morphology.disk(selem_radius)
        top_hat = morphology.white_tophat(im_binned, selem)
        # plt.imshow(top_hat)
        # plt.show()

        # manual top hat
        im_erosion = grey_erosion(im_binned, footprint=selem)
        # plt.imshow(im_erosion)
        # plt.show()

        im_dilation = grey_dilation(im_erosion, footprint=selem)
        # plt.imshow(im_dilation)
        # plt.show()

        im_subtraction = im_binned - im_dilation
        # plt.imshow(im_subtraction)
        # plt.show()

        assert not (
            top_hat - im_subtraction
        ).any(), "top_hat does not match manual top hat"

        if im_subtraction.sum() == 0:
            print(
                f"top_hat returned all zeros: bin_size:{bin_size} radius: {selem_radius}"
            )
            continue

        # cluster
        k = 2
        n_brightest_clusters = 1
        im_kmeans = kmeans_threshold(
            top_hat, k=k, n_brightest_clusters=n_brightest_clusters
        )

        # plt.title(f"bin: {kernel_size}x{kernel_size}  disk_radii:{selem_radius}")
        # plt.imshow(im_kmeans)
        # plt.show()

        debug = False
        if debug:
            # original    aag_mask   tq_mask
            # mask_pred   dice plot
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))

            fig.suptitle(
                f"bin: {kernel_size}x{kernel_size}  disk_radius:{selem_radius}"
            )
            ax[0, 0].set_title(f"original")
            ax[0, 0].imshow(im)
            ax[0, 0].set_axis_off()

            ax[0, 1].set_title(f"mask_aag")
            ax[0, 1].imshow(mask_aag)
            ax[0, 1].set_axis_off()

            ax[0, 2].set_title(f"mask_tq")
            ax[0, 2].imshow(mask_tq)
            ax[0, 2].set_axis_off()

            ax[1, 0].set_title(f"gray_erosion")
            ax[1, 0].imshow(im_erosion)
            ax[1, 0].set_axis_off()

            ax[1, 1].set_title(f"gray_dilation")
            ax[1, 1].imshow(im_dilation)
            ax[1, 1].set_axis_off()

            ax[1, 2].set_title(f"top_hat \n (im - opening(im))")
            ax[1, 2].imshow(im_subtraction)
            ax[1, 2].set_axis_off()
            plt.show()

        array_dice_scores_aag[bin_size, selem_radius] = dice(im_subtraction, mask_aag)
        array_dice_scores_tq[bin_size, selem_radius] = dice(im_subtraction, mask_tq)

    #%%
