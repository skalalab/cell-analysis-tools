# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:26:31 2021

@author: Nabiki
"""
import matplotlib.pylab as plt

def image_show(image, nrows=1, ncols=1): # , cmap='gray'
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(image) # , cmap='gray'
    ax.axis('off')
    plt.show()
    return fig, ax

def compare_images(im1, title1, im2, title2, figsize=(10,5)):
    
    fig, ax = plt.subplots(1,2, figsize=figsize)
    ax[0].title.set_text(title1)
    ax[0].imshow(im1)
    ax[0].set_axis_off()
    
    ax[1].title.set_text(title2)
    ax[1].imshow(im2)
    ax[1].set_axis_off()
    plt.show()