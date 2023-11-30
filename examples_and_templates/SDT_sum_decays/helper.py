# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:19:56 2022

@author: econtrerasguzman
"""

from pathlib import Path
import sys
from PIL import Image

import numpy as np

def _write_sdt(path_output, im, manufacturer="BH", resolution=256):
    

    # Requires the "sdtheader.dat" built header information
    
    ### Example1 : random dataset
    #binary_data=(np.random.randint(100,size=[256*256*256])).astype(np.uint16)
    
    ### Example2 : any data set with 256x256x256 - uint16
    # with open('badger.dat','rb') as fid:
    #     binary_data=np.fromstring(fid.read(),np.uint16)    
    
    #phantom_data= binary_data.ravel().astype(np.uint16)

    path_header = Path(f"./headers/header_{resolution}_{manufacturer}.dat")
    
    with open(path_header,'rb') as fid:
        header_ = fid.read() # prebuilt header_file for all 256x256 files
    
    # convert image
    if not isinstance(im, np.ndarray):
        binary_data = im.to_numpy().astype(np.uint16)
    else:
        binary_data = im.astype(np.uint16)
        
    # combine header and data
    phantom_data = header_ + binary_data.tobytes()
    
    # with open('phantom_data.sdt','wb') as fid:
    #     fid.write(phantom_data)
    
    with open(path_output,'wb') as fid:
        fid.write(phantom_data)   
