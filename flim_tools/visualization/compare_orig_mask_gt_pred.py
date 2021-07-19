# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:50:49 2021

@author: Nabiki
"""
import matplotlib as mpl
mpl.rcParams["figure.dpi"] == 300
import matplotlib.pylab as plt

from flim_tools.image_processing import normalize
from flim_tools.metrics import dice, total_error
from skimage.color import label2rgb
import numpy as np


def compare_orig_mask_gt_pred(im, mask_gt, mask_pred, title="" ):
    alpha = 0.5
    im_overlay = label2rgb(mask_pred, normalize(im), bg_label=0, alpha=alpha, image_alpha=1, kind="overlay")
    
    fig, ax = plt.subplots(2,3, figsize=(10,7))
    
    plt.suptitle(title)
    ax[0,0].title.set_text(f"original")
    ax[0,0].set_axis_off()
    ax[0,0].imshow(im)
    
    # overlayed 
    dice_coeff = dice(mask_pred, mask_gt) 
    ax[0,1].title.set_text(f"overlayed mask_pred")
    ax[0,1].set_axis_off()
    ax[0,1].imshow(im_overlay)
    
    # mask gt 
    ax[1,0].title.set_text(f"mask_gt")
    ax[1,0].set_axis_off()
    ax[1,0].imshow(mask_gt)
    
    # mask pred
    ax[1,1].title.set_text(f"mask_pred \n dice: {dice_coeff:.4f}")
    ax[1,1].set_axis_off()
    ax[1,1].imshow(mask_pred)
    
    ## XOR
    mask_xor = np.logical_xor(mask_gt,mask_pred)
    
    error_total = total_error(mask_gt, mask_pred)
    ax[0,2].title.set_text(f"mask_xor\n total error: {(error_total*100):.3f}")
    ax[0,2].set_axis_off()
    ax[0,2].imshow(mask_xor)
    
    ax[1,2].set_axis_off()
    plt.show()
    

if __name__ == "__main__":

    
    from flim_tools.visualization import compare_orig_mask_gt_pred
    import numpy as np
    
    im = np.random.rand(40,40)
    compare_orig_mask_gt_pred(im, im, im)

    # print("TODO add test code")
    
    # im_orig = np.random.rand(512,512)
    # im_gt = np.round(im_orig)
    
    # compare_orig_mask_gt_pred(im_orig, im_gt, im_orig,"comparing originals")
    
    
    
    
    
    
