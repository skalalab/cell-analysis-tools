#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:05:27 2022

@author: nabiki
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


from cell_analysis_tools.flim import (bin_image
                                      )

class TestFLIM:
    
    default_rng = np.random.default_rng(seed=1)
    
    # generate temporary image
    im = default_rng.random((256,256,256))
    plt.imshow(im.sum(axis=2))
    plt.show()
    
    def test_bin_image(self):
        pass
        im_binned = bin_image(self.im, 2)
        plt.imshow(im_binned.sum(axis=2))
        plt.show()
        

if __name__ == "__main__":
    flim = TestFLIM()
    
    flim.test_bin_image()
        

