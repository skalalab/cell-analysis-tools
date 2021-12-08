# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:58:18 2021

@author: Nabiki
"""

from skimage.measure import regionprops
from skimage.morphology import label
import numpy.ma as ma
import numpy as np
from flim_tools.io import read_asc
import tifffile
from pprint import pprint

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

#%%
def regionprops_omi(
                    image_id, # base_name of image
                    label_image,
                    im_nadh_intensity,
                    im_nadh_a1, 
                    im_nadh_a2, 
                    im_nadh_t1, 
                    im_nadh_t2,
                    im_fad_intensity,
                    im_fad_a1,
                    im_fad_a2,
                    im_fad_t1,
                    im_fad_t2,
                    ):
    """
    Takes in labels image as well as nadh and fad images to reuturn
    mean and stdev of each parameter per roi

    Parameters
    ----------
    label_image : TYPE
        DESCRIPTION.
    im_nadh_intensity : TYPE
        DESCRIPTION.
    im_nadh_a1 : TYPE
        DESCRIPTION.
    im_nadh_a2 : TYPE
        DESCRIPTION.
    im_nadh_t1 : TYPE
        DESCRIPTION.
    im_nadh_t2 : TYPE
        DESCRIPTION.
    im_fad_intensity : TYPE, optional
        DESCRIPTION. The default is None.
    im_fad_a1 : TYPE, optional
        DESCRIPTION. The default is None.
    im_fad_a2 : TYPE, optional
        DESCRIPTION. The default is None.
    im_fad_t1 : TYPE, optional
        DESCRIPTION. The default is None.
    im_fad_t2 : TYPE, optional
        DESCRIPTION. The default is None.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    pass
    # 1. nadh_intensity
    # 2. nadh_a1
    # 3. nadh_a2
    # 4. nadh_t1
    # 5. nadh_t2
    # 6. nadh_tau_mean
    # 7. fad_intensity
    # 8. fad_a1
    # 9. fad_a2
    # 10. fad_t1
    # 11. fad_t2
    # 12. fad_tau_mean
    # 13. redox_ratio

    # convert a1/a2 to percent
    im_nadh_tau_mean = (im_nadh_a1/100 * im_nadh_t1) + (im_nadh_a2/100 * im_nadh_t2)
    im_fad_tau_mean = (im_fad_a1/100 * im_fad_t1) + (im_fad_a2/100 * im_fad_t2)
    im_redox_ratio = im_nadh_intensity / im_fad_intensity
    
    
    def stdev(roi, intensity):
        inverted_roi = np.invert(roi.astype(bool))
        masked_image = ma.masked_array(intensity, mask=inverted_roi)
        return np.std(masked_image)
    
    # count # of components and exlude bg
    n_rois = 0
    for value in np.unique(label_image)[1:]: #exclude bg
        im_temp = label(label_image==value)
        # plt.title(len(np.unique(im_temp)))
        # plt.imshow(im_temp)
        # plt.show()
        n_rois += len(np.unique(im_temp)) -1 # exclude bg
        # print(n_rois)
        
    # make sure rois are uniquely labeled, exclude bg
    #assert len(np.unique(label_image)[1:]) == n_rois, "mask does not have unique labels"
            
    nadh_intensity = regionprops(label_image, im_nadh_intensity, extra_properties=[stdev])
    nadh_a1 = regionprops(label_image, im_nadh_a1, extra_properties=[stdev])
    nadh_a2 = regionprops(label_image, im_nadh_a2, extra_properties=[stdev])
    nadh_t1 = regionprops(label_image, im_nadh_t1, extra_properties=[stdev])
    nadh_t2 = regionprops(label_image, im_nadh_t2, extra_properties=[stdev])
    nadh_tau_mean = regionprops(label_image, im_nadh_tau_mean, extra_properties=[stdev])
    fad_intensity = regionprops(label_image, im_fad_intensity, extra_properties=[stdev])
    fad_a1 = regionprops(label_image, im_fad_a1, extra_properties=[stdev])
    fad_a2 = regionprops(label_image, im_fad_a2, extra_properties=[stdev])
    fad_t1 = regionprops(label_image, im_fad_t1, extra_properties=[stdev])
    fad_t2 = regionprops(label_image, im_fad_t2, extra_properties=[stdev])
    fad_tau_mean = regionprops(label_image, im_fad_tau_mean, extra_properties=[stdev])
    redox_ratio = regionprops(label_image, im_redox_ratio, extra_properties=[stdev])

    dict_regionprops = {
        "nadh_intensity" : nadh_intensity,
        "nadh_a1" : nadh_a1,
        "nadh_a2" : nadh_a2,
        "nadh_t1" : nadh_t1,
        "nadh_t2" : nadh_t2,
        "nadh_tau_mean" : nadh_tau_mean,
        "fad_intensity" : fad_intensity,
        "fad_a1" : fad_a1,
        "fad_a2" : fad_a2,
        "fad_t1" : fad_t1,
        "fad_t2" : fad_t2,
        "fad_tau_mean" : fad_tau_mean,
         "redox_ratio" : redox_ratio,
        }
    
    dict_omi = {}
      
    for rp_key in dict_regionprops.keys():#iterate through regionprops
        for region in dict_regionprops[rp_key]: # iterate through rois in regionprops
            pass
            dict_key_name = f"{image_id}_{region.label}"
            if not dict_key_name in dict_omi.keys(): # add key if needed
                pass
                dict_omi[dict_key_name] = {} # new dict for label if needed
                dict_omi[dict_key_name][f"mask_label"] = int(region.label)
            
            dict_omi[dict_key_name][f"{rp_key}_mean"] = region.mean_intensity
            dict_omi[dict_key_name][f"{rp_key}_stdev"] = region.stdev


    #dictionary of omi features could be df if we wanted to
    return dict_omi
#%%

if __name__ == "__main__":
    
    from pathlib import Path
    import pandas as pd
    path_dictionaries = Path(r"Z:\0-Projects and Experiments\RD - redox_ratio_development\Data Combined + QC Complete\0-dictionaries")
    
    list_csv_files = list(path_dictionaries.glob("*"))

    csv_dict =  pd.read_csv(list_csv_files[0])   
    csv_dict.index.name = "base_name"
    
    test_dict = csv_dict.iloc[0]
    
    def load_image(path):
        # detects extension and loads image accordingly
        # tif/tiff vs asc
        pass
        if path.suffix == ".asc":
            return read_asc(path)
        if path.suffix in [".tiff", ".tif"]:
            return tifffile.imread(path)
#%%

    omi_props = regionprops_omi(
        label_image = load_image(Path(test_dict.mask_cell)),
        im_nadh_intensity = load_image(Path(test_dict.nadh_photons)),
        im_nadh_a1 = load_image(Path(test_dict.nadh_a1)), 
        im_nadh_a2 = load_image(Path(test_dict.nadh_a2)), 
        im_nadh_t1 = load_image(Path(test_dict.nadh_t1)), 
        im_nadh_t2 = load_image(Path(test_dict.nadh_t2)),
        im_fad_intensity = load_image(Path(test_dict.fad_photons)),
        im_fad_a1 = load_image(Path(test_dict.fad_a1)),
        im_fad_a2 = load_image(Path(test_dict.fad_a2)),
        im_fad_t1 = load_image(Path(test_dict.fad_t1)),
        im_fad_t2 = load_image(Path(test_dict.fad_t2)),
        )    
    
    pprint(omi_props)
    df = pd.DataFrame(omi_props).transpose()
