# Python fft filter

from time import time
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifft, ifftshift
from numpy import ones

from skimage.morphology import disk
from pathlib import Path
from PIL import Image 
import matplotlib.pylab as plt

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300


def remove_horizontal_vertical_edges(im, disk_size=20, debug=False):
    """ This function removes horizontal or vertical edges in the image. Designed for removing grid patterns in
    PDMS scaffolds during time lapse imaging 

    Parameters
    ----------
    im : ndarray
        intensit image with grid pattern
    disk_size : int
        size of filter used to filter low frequency components of 2d image
    debug : bool, optional
        output intermediate images , by default False

    Returns
    -------
    ndarray
        filtered image with grid pattern removed
    """

    tic = time()

    imsize = min(im.shape)
    im = im[:imsize, :imsize] # make square
    # % Fourier transform of the image (complex array)
    # fftim = fftshift(fft2(fftshift(im)));
    fftim = fftshift(fft2(fftshift(im)))

    # % create a mask to filter out certain spatial frequencies
    # filter = circ(imsize,20); % pass low freqs near the origin
    selem_disk = disk(disk_size)
    pad_amount = (im.shape[0] - selem_disk.shape[0]) // 2
    filt1 = np.pad(selem_disk,(pad_amount,pad_amount), mode="constant", constant_values=0)

    ##
    # spokewidth = 5; % width of spokes, bigger catches more angles. Could do a pinwheel or other more sophisticated shapes.
    # filter2 = ones(size(im)); filter2(imsize/2-spokewidth:imsize/2+spokewidth,:)=0;filter2(:,imsize/2-spokewidth:imsize/2+spokewidth)=0;
    # filter = (filter|filter2); 

    spokewidth = 5 #; % width of spokes, bigger catches more angles. Could do a pinwheel or other more sophisticated shapes.
    filt2 = ones(im.shape)
    filt2[imsize//2-spokewidth : imsize//2 + spokewidth, :] = 0
    filt2[: , imsize//2-spokewidth : imsize//2+spokewidth] = 0
    filt2 = filt2.astype(np.uint8)
    filt = [filt1|filt2]

    #% apply the filter in Fourier space and FFT back to image space
    # filteredim = abs(fftshift(ifft2(ifftshift(fftim.*filter))));
    filteredim = abs(fftshift(ifft2(ifftshift(fftim*filt))))

    if debug:
        toc = time()
        print(f"elapsed time: {toc-tic}")

        ## display
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(im, cmap="gray")
        ax[0].set_title("original")
        ax[1].imshow(filteredim.squeeze(), cmap="gray")
        ax[1].set_title("Fourier filtered")
    
    return filteredim

if __name__ == "__main__":

    im = Image.open(r"C:\Users\Nabiki\Desktop\Screenshot 2021-09-15 080238.png".replace("\\",'/'))
    im = np.asarray(im)
    
    # plt.imshow(im)
    im = im.sum(axis=2)
    im_filtered = remove_horizontal_vertical_edges(im)
