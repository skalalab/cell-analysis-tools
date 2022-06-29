import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage import data
from skimage.color import label2rgb
from skimage.feature import canny
from skimage.filters import (
    threshold_minimum,
    threshold_multiotsu,
    threshold_otsu,
    threshold_yen,
)
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, closing, square
from skimage.segmentation import clear_border


def thresh_demo(image, show_plots=True, re=False):
    # Here, we assign each of the threshold results as described above
    thresh_otsu = threshold_otsu(image)
    thresh_yen = threshold_yen(image)
    thresh_min = threshold_minimum(image)

    # We have a built in method for binary closing
    # - in this case, we want to keep any pixel where it exceeds the thresholded image
    bw_otsu = binary_closing(image > thresh_otsu, square(3))
    bw_yen = binary_closing(image > thresh_yen, square(3))
    bw_min = binary_closing(image > thresh_min, square(3))

    if re:
        return bw_otsu, bw_yen, bw_min

    if show_plots:
        # Plot the final images here
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        fig.set_facecolor((1, 1, 1, 1))
        fig.suptitle("Binary closed images", fontsize=16)
        ax1.imshow(bw_otsu, plt.cm.Greys_r)
        ax1.set_title("Otsu", fontsize=12)
        ax1.set_axis_off()
        ax2.imshow(bw_yen, plt.cm.Greys_r)
        ax2.set_title("Yen", fontsize=12)
        ax2.set_axis_off()
        ax3.imshow(bw_min, plt.cm.Greys_r)
        ax3.set_title("Minimum", fontsize=12)
        ax3.set_axis_off()
        plt.tight_layout()
        plt.show()


def remove_artificats_demo(image, show_plots=True, re=False):
    bw_otsu, bw_yen, bw_min = thresh_demo(image, show_plots=False, re=True)

    # remove artifacts connected to image border
    cleared_otsu = clear_border(bw_otsu)
    cleared_yen = clear_border(bw_yen)
    cleared_min = clear_border(bw_min)
    cleared_max = np.maximum(
        np.array(cleared_otsu).astype(int), np.array(cleared_min).astype(int)
    )

    if show_plots:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        fig.set_facecolor((1, 1, 1, 1))
        fig.suptitle("Cleared Borders", fontsize=16)
        ax1.imshow(cleared_otsu, plt.cm.Greys_r)
        ax1.set_axis_off()
        ax1.set_title("Otsu", fontsize=12)
        ax2.imshow(cleared_yen, plt.cm.Greys_r)
        ax2.set_axis_off()
        ax2.set_title("Yen", fontsize=12)
        ax3.imshow(cleared_min, plt.cm.Greys_r)
        ax3.set_axis_off()
        ax3.set_title("Minimum", fontsize=12)
        ax4.imshow(cleared_max, plt.cm.Greys_r)
        ax4.set_axis_off()
        ax4.set_title("Maximum Intensity", fontsize=12)
        plt.tight_layout()
        plt.show()
    if re:
        return cleared_otsu, cleared_yen, cleared_min, cleared_max


def label_and_assign_rois(image, show_plots=True, re=False):
    cleared_otsu, cleared_yen, cleared_min, cleared_max = remove_artificats_demo(
        image, show_plots=False, re=True
    )

    # label image regions
    label_image_otsu = label(cleared_otsu)
    label_image_yen = label(cleared_yen)
    label_image_min = label(cleared_min)
    label_image_max = label(cleared_max)

    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay_otsu = label2rgb(label_image_otsu, image=image, bg_label=0)
    image_label_overlay_yen = label2rgb(label_image_yen, image=image, bg_label=0)
    image_label_overlay_min = label2rgb(label_image_min, image=image, bg_label=0)
    image_label_overlay_max = label2rgb(label_image_max, image=image, bg_label=0)

    if re:
        return label_image_otsu, label_image_yen, label_image_min, label_image_max

    if show_plots:
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        fig.set_facecolor((1, 1, 1, 1))
        fig.suptitle("Assigned region props and cell profiler style labels")
        ax[0].imshow(image_label_overlay_otsu)
        ax[1].imshow(image_label_overlay_yen)
        ax[2].imshow(image_label_overlay_min)
        ax[3].imshow(image_label_overlay_max)

        labeled_thresholding_methods = [
            label_image_otsu,
            label_image_yen,
            label_image_min,
            label_image_max,
        ]

        for index, labeled_thresholding_image in enumerate(
            labeled_thresholding_methods
        ):
            for region in regionprops(labeled_thresholding_image):
                # take regions with large enough areas
                if region.area >= 100:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle(
                        (minc, minr),
                        maxc - minc,
                        maxr - minr,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                    ax[index].add_patch(rect)

        ax[0].set_axis_off()
        ax[0].set_title("Otsu")
        ax[1].set_axis_off()
        ax[1].set_title("Yen")
        ax[2].set_axis_off()
        ax[2].set_title("Minimum")
        ax[3].set_axis_off()
        ax[3].set_title("Maximum Intensity")
        plt.tight_layout()
        plt.show()


def full_pipeline(image, show_plots=True, re=False):
    (
        label_image_otsu,
        label_image_yen,
        label_image_min,
        label_image_max,
    ) = label_and_assign_rois(image, show_plots=False, re=True)
    print
    edges = canny(
        image=label_image_max, sigma=2.0, low_threshold=0.01, high_threshold=0.1,
    )
    binary_filled_edges = ndi.binary_fill_holes(binary_closing(edges))
    labeled_image = ndi.label(binary_filled_edges)[0]

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        binary_filled_edges.astype(np.uint8), connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 150

    # your answer image
    img2 = np.zeros((output.shape))

    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    img2_labeled = ndi.label(img2)[0]

    if show_plots:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        fig.set_facecolor((1, 1, 1, 1))
        fig.suptitle("Full pipeline demo")
        ax1.set_title("Original Image")
        ax1.imshow(image, cmap=plt.cm.Greys_r)
        ax2.set_title("Detected Edges (Canny Filter)")
        ax2.imshow(edges, cmap=plt.cm.Greys_r)
        ax3.set_title("Binary closing applied, edges filled")
        ax3.imshow(binary_filled_edges, cmap=plt.cm.Greys_r)
        ax4.set_title("Cellprofiler style region labeling")
        ax4.imshow(labeled_image, cmap=plt.cm.Greys_r)
        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        fig.set_facecolor((1, 1, 1, 1))
        fig.suptitle("Small item filtering")
        ax1.imshow(labeled_image, cmap=plt.cm.Greys_r)
        ax1.set_axis_off()
        ax1.set_title("Non filtered image")
        ax2.imshow(img2_labeled, cmap=plt.cm.Greys_r)
        ax2.set_axis_off()
        ax2.set_title("Attempt at small item filtering")
        plt.tight_layout()
        plt.show()

    if re:
        return img2_labeled, output
