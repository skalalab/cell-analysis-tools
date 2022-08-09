import numpy
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from read_roi import read_roi_zip
from skimage.draw import polygon2mask


# given a list of ROIs returns a mask
def create_mask_from_rois(rois_vert):
    width = 256
    full_image_mask = np.zeros((width, width), dtype=np.uint8)
    rois_in_image = []
    rois_vertices = []
    for roi in list(rois_vert.keys()):
        rois_vertices.append(roi)
        col_coords = rois_vert[roi]["x"]
        row_coords = rois_vert[roi]["y"]
        polygon = [
            (row_coords[i], col_coords[i]) for i in range(0, len(row_coords))
        ]  # create list of values
        # img = Image.new('L', (width, width), 0)
        # ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        # single_roi = numpy.array(img)
        image_shape = (width, width)
        single_roi_mask = polygon2mask(image_shape, polygon)
        full_image_mask = (
            full_image_mask + single_roi_mask
        )  # add roi to whol image mask
        rois_in_image.append(single_roi_mask)
        #    plt.imshow(mask)
    binary_mask = full_image_mask > 0
    # image_show(mask)
    # image_show(binary_mask)
    return binary_mask, rois_in_image


def split_mask_into_rois(image_masks):
    mask_sets = []
    for mask_image_idx, mask in enumerate(image_masks):  # iterate through each mask
        rois = []
        # iterate through all mask rois, zero is bg mask
        # image_show(mask)
        for roi_value in np.unique(mask):
            if roi_value != 0:  # skip bg mask with value of zero
                rois.append(mask == roi_value)
        print(len(rois))
        mask_sets.append(rois)
    return mask_sets


def threshold_masks(images, masks, rois_pixel_count):
    refined_roi_masks = []
    thresholds = []

    # mask photon images
    masked_images = []
    for pos, image in enumerate(images):
        masked_images.append(image * masks[pos])

    # iterate through all the images
    for img_idx, masked_image in enumerate(masked_images):
        # threshold from 0 to max num pixels until we get ~ same ammount
        for i in np.arange(np.max(masked_image)):
            thresholded_image = masked_image > i  # boolean mask
            if thresholded_image.sum() <= rois_pixel_count[img_idx]:  # count pixels
                print(f"threshold: {i} ")
                refined_roi_masks.append(thresholded_image)
                thresholds.append(i)
                break

    return thresholds, refined_roi_masks


def refined_roi_sets(roi_sets, masks):
    new_roi_sets = []
    for pos, roi_set in enumerate(roi_sets):  # iterate through roi sets
        mask = masks[pos]
        image_rois = []  # store each images ROIs
        for roi in roi_set:
            image_rois.append(mask * roi)
        new_roi_sets.append(image_rois)
    return new_roi_sets
