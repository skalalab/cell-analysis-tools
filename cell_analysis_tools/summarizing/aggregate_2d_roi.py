import re
from pathlib import Path

import numpy
import numpy as np
import numpy.ma as ma
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from flim_tools.image_processing import normalize
from flim_tools.io import load_sdt

#%%


def aggregate_2d_roi(mask, im, debug=False):

    pass


#%%  CREATE DICTIONARY OF FILE PATHS


def _validate_file_found(results):

    if len(results) == 0:
        print("no image found")
        return False

    if len(results) > 1:
        print("too many matches found")
        return False
    return True


# path to dataset and to output
path_dataset = Path("C:/Users/Nabiki/Desktop/data/T_cells-paper/Data/042018 - Donor 6")
path_output = Path("C:/Users/Nabiki/Desktop/")

# stores dictionary of paths to set of files for image (.sdt, t1.tiff, t2.tiff etc())
dataset = []

# paths to dish folders
list_paths_dishes = list(path_dataset.glob("Dish*"))

# iterate through dishes --> odd filenames are nadh
for path_dish in list_paths_dishes[
    :5
]:  # list_paths_dishes[0:1] # oly select these dishes with valid info
    pass

    # GET LIST PATHS TO SDT FILES
    path_sdt_files = list(path_dish.glob("*.sdt"))  # [13579]{{1}}\
    list_sdt_paths = [str(path) for path in path_sdt_files]  # convert to string
    pattern = r".*[13579]{1}\.sdt"  # find only odd files
    list_path_sdt = list(filter(re.compile(pattern).match, list_sdt_paths))
    list_path_sdt.sort()
    list_path_sdt = [Path(path) for path in list_path_sdt]  # convert to pathlib objects

    # LIST PATHS TO MASKS
    """""" """""" """whole cell ROI's""" """""" """"""
    regex = "Cell_mask.*\.tif"
    list_path_whole_cell_masks = list(path_dish.glob("Cell_mask*.tif"))
    list_path_whole_cell_masks.sort()

    """""" """""" """cytoplasm cell ROI's""" """""" """"""
    regex = "Mask_image.*\.tif"
    list_path_cyto_masks = list(path_dish.glob("Mask_image*.tif"))
    list_path_cyto_masks.sort()

    # num sdt files doesn't match the masks
    if (
        not len(list_path_sdt)
        == len(list_path_whole_cell_masks)
        == len(list_path_cyto_masks)
    ):
        print("number of SDT files doesn't match the masks")
        print(
            f" sdt: {len(list_path_sdt)} \n whole_cell_masks: {len(list_path_whole_cell_masks)} \n list_path_cyto_mask: {len(list_path_cyto_masks)}"
        )
        print(
            f"\nFix this and then come back to this dish:  {path_dish.parent.stem} / {path_dish.stem} "
        )
        continue  # skip this dish folder
    else:
        print(
            f"num sdt files, cell masks and cytoplasm masks the same: {len(list_path_sdt)}"
        )

    ##### TODO
    list_path_all_files = list(path_dish.glob("*"))
    list_path_all_files = [str(p) for p in list_path_all_files]  # convert to strings
    for pos, path_sdt in enumerate(list_path_sdt):
        pass
        prefix = path_sdt.stem  # get file identifer

        print(f"processing: {path_sdt}")
        ### load SPCM files by matching prefix
        pattern = f".*{prefix}.*Ch2.*photons.tiff"
        path_photons = list(
            filter(re.compile(pattern).match, list_path_all_files)
        )  # returns list of matching files, should be 1
        if not _validate_file_found(path_photons):
            print("photons")
            continue
        else:
            path_photons = Path(path_photons[0])  # one item list, get string

        pattern = f".*{prefix}.*Ch2.*a1\[%\].tiff"
        path_a1 = list(filter(re.compile(pattern).match, list_path_all_files))
        if not _validate_file_found(path_a1):
            print("a1")
            continue
        else:
            path_a1 = Path(path_a1[0])  # one item list, get string

        pattern = f".*{prefix}.*Ch2.*a2\[%\].tiff"
        path_a2 = list(filter(re.compile(pattern).match, list_path_all_files))
        if not _validate_file_found(path_a2):
            print("a2")
            continue
        else:
            path_a2 = Path(path_a2[0])  # one item list, get string

        pattern = f".*{prefix}.*Ch2.*t1.tiff"
        path_t1 = list(filter(re.compile(pattern).match, list_path_all_files))
        if not _validate_file_found(path_t1):
            print("t1")
            continue
        else:
            path_t1 = Path(path_t1[0])  # one item list, get string

        pattern = f".*{prefix}.*Ch2.*t2.tiff"
        path_t2 = list(filter(re.compile(pattern).match, list_path_all_files))
        if not _validate_file_found(path_t2):
            print("t2")
            continue
        else:
            path_t2 = Path(path_t2[0])  # one item list, get string

        pattern = f".*{prefix}.*Ch2.*chi.tiff"
        path_chisq = list(filter(re.compile(pattern).match, list_path_all_files))
        if not _validate_file_found(path_chisq):
            print("chisq")
            continue
        else:
            path_chisq = Path(path_chisq[0])  # one item list, get string

        # sanity check - sdts match masks
        # load sdt
        im_sdt = load_sdt(list_path_sdt[pos])
        im_sdt = np.reshape(im_sdt, (2, 256, 256, 256))
        im_sdt = np.sum(im_sdt[1, ...], axis=2)
        im_sdt = normalize(im_sdt)

        # image dividers
        height = im_sdt.shape[1]
        div = np.ones((height, 5))

        # load masks
        path_m_cyto = list_path_cyto_masks[pos]
        path_m_cell = list_path_whole_cell_masks[pos]
        m_cyto = tifffile.imread(str(path_m_cyto))
        m_cell = tifffile.imread(str(path_m_cell))
        plt.title(
            f"{path_sdt.parent.stem} - {prefix} \n  sdt  | mask_whole_cell  | mask_cytoplasm"
        )
        plt.imshow(np.c_[im_sdt, div, normalize(m_cell), div, normalize(m_cyto)])
        plt.show()

        # ALL FILES FOUND, ADD TO LIST
        set_dict = {
            "sdt": path_sdt,
            "mask_whole_cell": path_m_cell,  # store paths to cell/cyto masks
            "mask_cyto": path_m_cyto,
            "photons": path_photons,
            "a1": path_a1,
            "a2": path_a2,
            "t1": path_t1,
            "t2": path_t2,
            "chisq": path_chisq,
        }

        dataset.append(set_dict)
#%% # iterate and aggregate

# lists to store decay data
list_decays_header = []
list_decays = []

# lists to store spcm features data
df_features_header = []
list_im_features_data = []

# ITERATE THROUGH IMAGES
for idx, im_set in enumerate(dataset):  # look at first image, last in set is not good
    pass
    print(f"processing image: {idx+1}/{len(dataset)}  |  {im_set['sdt'].name}")

    # LOAD MASK TO USE SELECT THE MASK TO USE
    # mask = tifffile.imread(im_set['mask_whole_cell'])
    mask = tifffile.imread(im_set["mask_cyto"])

    # CALCULATE CYTO VALUES
    list_roi_values = np.unique(mask)
    # plt.imshow(mask==0)
    # plt.show()

    list_roi_values = np.delete(
        list_roi_values, np.where(list_roi_values == 0)
    )  # exclude bg

    # ITERATE THROUGH ROI's, create header along the way
    for roi_idx, roi_value in enumerate(list_roi_values):
        print(f"processing roi {roi_idx+1}/{len(list_roi_values)}")
        pass

        # CREATE MASK OF ROI
        m_roi = (mask == roi_value).astype(int)

        roi_label = f"{path_sdt.stem}_roi_{roi_value}"

        #################
        # AGGREGATE DECAY
        # extract decay
        path_sdt = im_set["sdt"]
        im_sdt = load_sdt(path_sdt)
        im_sdt = np.reshape(im_sdt, (2, 256, 256, 256))  # adjsut these as needed
        im_sdt = im_sdt[1, ...]
        masked_sdt_data = im_sdt * m_roi[..., np.newaxis]
        decay = masked_sdt_data.sum(axis=(0, 1))  # sum over x and y, leave t alone
        # plt.plot(decay)
        # plt.show()
        list_decays_header.append(roi_label)
        list_decays.append(decay)

        #################
        # EXTRACT FEATURES - iterate through spcm files(features list)
        list_roi_features = []
        list_roi_features.append(roi_label)  # add label
        m_roi = 1 - m_roi  # invert mask for use in masked array

        # Make list of just spcm files(features)
        list_features = list(im_set.keys())
        list_features.remove("sdt")
        list_features.remove("mask_whole_cell")
        list_features.remove("mask_cyto")

        list_roi_features_header = []  # make list of header terms for df
        for feature in list_features:
            pass

            # feature
            file = im_set[feature]
            data = tifffile.imread(file)
            m_data = ma.masked_array(data, mask=m_roi)

            header = f"{file.stem}"

            # mean
            list_roi_features_header.append(feature + "_mean")
            list_roi_features.append(m_data.mean(axis=(0, 1)))

            # stdev
            list_roi_features_header.append(feature + "_stdev")
            list_roi_features.append(m_data.std(axis=(0, 1)))

        #################
        # ADD tm values
        a1 = ma.masked_array(tifffile.imread(im_set["a1"]), mask=m_roi)
        t1 = ma.masked_array(tifffile.imread(im_set["t1"]), mask=m_roi)
        a2 = ma.masked_array(tifffile.imread(im_set["a2"]), mask=m_roi)
        t2 = ma.masked_array(tifffile.imread(im_set["t2"]), mask=m_roi)

        list_roi_features_header.append("tm_mean")
        list_roi_features.append(np.mean(a1 * t1 + a2 * t2))

        list_roi_features_header.append("tm_stdev")
        list_roi_features.append(np.std(a1 * t1 + a2 * t2))

        #################
        # INCLUDE OTHER ROI PROPERTIES
        l = 1 - m_roi  # invert mask again
        r = regionprops(l)  # get region props for this mask, should be array of 1

        # append calculated features by regionprops
        list_regionprop_header = [
            "area",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
        ]
        list_roi_features.append(r[0].area)
        list_roi_features.append(r[0].major_axis_length)
        list_roi_features.append(r[0].minor_axis_length)
        list_roi_features.append(r[0].perimeter)

        # add roi features to list
        list_im_features_data.append(list_roi_features)

        ###### POPULATE DF HEADER IF LIST IS EMPTY
        if not df_features_header:
            df_features_header = (
                ["image_roi_id"] + list_roi_features_header + list_regionprop_header
            )
#%% ASSEMBLE AND EXPORT DF

### EXPORT DECAYS
df_decays = pd.DataFrame()
for roi_id, decay in list(zip(list_decays_header, list_decays)):
    pass
    df_decays[roi_id] = decay

#### EXPORT FEATURES -- create dataframe
df_features = pd.DataFrame(list_im_features_data, columns=df_features_header)

#%% SAVE CSVs
df_decays.to_csv(path_output / f"{path_dataset.stem}_decays.csv")
df_features.to_csv(path_output / f"{path_dataset.stem}_features.csv")
