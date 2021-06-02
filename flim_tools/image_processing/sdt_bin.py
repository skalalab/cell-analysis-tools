# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:38:37 2021

@author: Nabiki
"""
import numpy as np
from pathlib import Path
from flim_tools.io import load_sdt_file

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

def bin_sdt(im_sdt, bin_size, stride=1, pad_value=0, debug=True):
    """
    bins an SDT 3d array based on the kernel size.

    Parameters
    ----------
    im_sdt : ndarray
        3d ndarray containing FLIM data (x,y,t).
    bin_size : int
        number of pixels to look before and after center pixel.
        kernel = bin_size + 1 + bin_size
    stride : int, optional
        Number of pixels to move when raster scanning the image. The default is 1.
    pad_value : int, optional
        value to pad the image with. The default is 0.
    debug : bool, optional
        Show debugging output. The default is True.

    Returns
    -------
    im_binned : TYPE
        DESCRIPTION.

    """
    
    if stride != 1:
        print("strides larger than 1 not yet implemented")
    
    kernel_length =  bin_size*2 + 1
    
    rows_before = bin_size
    rows_after = bin_size
    cols_before = bin_size
    cols_after = bin_size
    
    n_rows, n_cols, _ = im_sdt.shape
    
    
    padded_sdt = np.pad(im_sdt, # this takes in width(cols) and height(rows)
                    ((rows_before,rows_after),
                     (cols_before,cols_after),
                     (0,0) # dont pad 3rd dim
                     ), 'constant', constant_values=pad_value 
                    )
    
    im_binned = np.zeros(im_sdt.shape)
    
    for row_idx in np.arange(n_rows):
        for col_idx in np.arange(n_cols):
            
            # run this and see if this works!
            # starts at 0(row_idx) : kernel_length + 0
            # then 1 (row_idx) : kernel_length + 1 ...etc
            # up to length n-2
            conv_decay = padded_sdt[row_idx : kernel_length + row_idx,
                                                 col_idx : kernel_length + col_idx,
                                                 :]
            im_binned[row_idx,col_idx,:] = conv_decay.sum(axis=(0,1))
    
    if debug:
        plt.title(f"kernel: {kernel_length}x{kernel_length}")
        plt.imshow(im_binned.sum(axis=2))
        plt.show()
        
    return im_binned

if __name__ == "__main__":
    
       
    HERE = Path(__file__).resolve().parent
    data = HERE.parent / r"example_data\t_cell\Tcells-001.sdt".replace('\\','/')
    
    # load SDT
    im_sdt = load_sdt_file(data)
    for ch in np.arange(im_sdt.shape[0]):
        plt.title(f"channel: {ch}")
        plt.imshow(im_sdt[ch,...].sum(axis=2))
        plt.show()

    # select channel with data
    im_sdt = im_sdt[1,...]
    
    # run with debug on
    bin_sdt(im_sdt, bin_size=3,debug=True)
    
    