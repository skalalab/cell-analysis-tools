from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np


mpl.rcParams["figure.dpi"] = 300


import tifffile
from scipy.ndimage import label


def bin_2d(im, bin_size, stride=1, pad_value=0, debug=False):
    """
    Bins a 2d array by a square kernel length (bin_size*2 + 1).

    Parameters
    ----------
    im : ndarray
        3d ndarray containing FLIM data (x,y,t).
    bin_size : int
        number of pixels to look before and after center pixel.
        kernel = bin_size + 1 + bin_size
    stride : int, optional
        Number of pixels to move when raster scanning the image. The default is 1.
    pad_value : int, optional
        value to pad the image with. The default is 0.
    debug : bool, optional
        Show debugging output. The default is False.

    Returns
    -------
    im_binned : ndarray
        binned 3d array of decays

    """

    if stride != 1:
        print("strides larger than 1 not yet implemented")

    kernel_length = bin_size * 2 + 1  # bin left center pixel and right sides

    rows_before = bin_size
    rows_after = bin_size
    cols_before = bin_size
    cols_after = bin_size

    n_rows, n_cols = im.shape

    padded_im = np.pad(
        im,  # this takes in width(cols) and height(rows)
        ((rows_before, rows_after), (cols_before, cols_after)),
        "constant",
        constant_values=pad_value,
    )

    im_binned = np.zeros(im.shape)

    # iterate through original image dimensions
    for row_idx in np.arange(n_rows):
        for col_idx in np.arange(n_cols):

            # starts at 0(row_idx) : kernel_length + 0
            # then 1 (row_idx) : kernel_length + 1 ...etc
            # up to length n-bin_size
            # get window on padded image
            conv_decay = padded_im[
                row_idx : kernel_length + row_idx, col_idx : kernel_length + col_idx
            ]
            # fill in original shape image
            im_binned[row_idx, col_idx] = conv_decay.sum(axis=(0, 1))

    if debug:
        plt.title(f"kernel: {kernel_length}x{kernel_length}")
        plt.imshow(im_binned)
        plt.show()

    return im_binned


if __name__ == "__main__":

    ### bin_2d TEST
    HERE = Path(__file__).resolve().parent

    # load SDT
    im = np.random.rand(512, 512)
    plt.title("original")
    plt.imshow(im)
    plt.show()

    # run with debug on
    bin_2d(im, bin_size=2, debug=True)
