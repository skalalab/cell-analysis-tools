from pathlib import Path

from tempfile import TemporaryDirectory
import shutil
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import tifffile

from natsort import natsorted
from cell_analysis_tools.visualization import compare_images

from sdt_read.read_bruker_sdt import read_sdt150
from sdt_read.read_wiscscan_sdt import read_sdt_wiscscan
import os

from helper import _write_sdt
import numpy as np
import re

from tqdm import tqdm

import subprocess
#%%

debug = False

path_sdts = Path("./sdts")
path_masks = Path("./masks")
path_output = Path("./sdts_summed")
# generate a list of folders to process

list_path_sdts = [sdt for sdt in path_sdts.rglob("*.sdt") if sdt.is_file()]

#SDT files
list_masks_files = [str(p) for p in natsorted(path_masks.glob("*.tiff"))]

# iterate throught the SDT files
for idx, path_sdt in tqdm(enumerate(list_path_sdts[:])):
    pass
    
    # Find mask
    base_name = path_sdt.stem
    path_mask = list(filter(re.compile(f".*{base_name}.*").search, list_masks_files))[0]

    if not Path(path_mask).exists():
        print(f"Mask not found for : {path_sdt.name}")
        continue
        
    # load mask
    labels = tifffile.imread(path_mask)
   
    ############## Jenu's code to load SDT's
    if os.path.getsize(path_sdt) > (2**25): # (file size is equal to 33555190, but ~32 MB is a good marker)
    	im = read_sdt_wiscscan(path_sdt)
    else:
    	im = read_sdt150(path_sdt)
    ##############
    
    if debug:
        compare_images('sdt', im.sum(axis=2), "mask", labels)
    
    # placeholder array
    sdt_decay_summed = np.zeros_like(im) 

    # iterate through labels
    list_labels = [l for l in np.unique(labels) if l != 0]
    for label in list_labels:
        pass
        mask_label = labels == label 
        
        decay_roi = im * mask_label[...,np.newaxis] # mask decays
        
        decay_summed = decay_roi.sum(axis=(0,1)) 
        
        if debug:

            fig, ax = plt.subplots(1,2, figsize=(10,5))
            fig.suptitle(f"{path_sdt.name} | label: {label}")
            ax[0].imshow(decay_roi.sum(axis=2))
            ax[0].set_aspect('equal')
            ax[1].plot(decay_summed)
            plt.show()
            # ax[1].set_aspect('equal')

            # plt.title(f"{path_sdt.name} | label: {label}")
            # plt.imshow(decay_roi.sum(axis=2))
            # plt.show()
            
            # plt.title(f"label: {label}")
            # plt.plot(decay_summed)
            # plt.show()
        
        sdt_decay_summed[mask_label] = decay_summed[np.newaxis, np.newaxis,:]
        
        # test 512x512x256
        # temp_array = np.zeros((512,512,256))
        # temp_array[:256,256:,...] = im
        # temp_array[:256,:256,...] = im
        # temp_array[256:,256:,...] = sdt_decay_summed
        # temp_array[256:,:256,...] = sdt_decay_summed
        # _write_sdt(path_output / f"{path_sdt.stem}_summed_512_BH.sdt", 
        #             temp_array, 
        #             resolution=512, 
        #             manufacturer="BH")
        #####
    
    # create temporary directory to process and compress files
    with TemporaryDirectory() as tempdir:
        
        tempdir = Path(tempdir)
        print(tempdir / f"{path_sdt.stem}_summed.sdt")
        width, _, _ = im.shape
        path_file = tempdir / f"{path_sdt.stem}_summed.sdt" 
        _write_sdt(path_file, sdt_decay_summed, resolution=width)
        
        #### COMPRESS FILES
        args = ["SDTZip.exe", "-z", str(path_file)]
        subprocess.run(args)
        
        path_compressed_file = path_file.parent / path_file.name.replace(".", ".compressed.")
        while not path_compressed_file.exists():
            pass # wait while file is being created
        
        
        # rename compressed file to uncompressed file
        shutil.copy2(path_compressed_file, path_output / path_file.name)
    
            
     
        