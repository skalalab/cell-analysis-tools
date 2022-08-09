from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

from cell_analysis_tools.io import load_sdt_file

mpl.rcParams["figure.dpi"] = 300


def sum_pool_3d(im, bin_size, stride=1, pad_value=0, debug=False):
    """
    Sums the pixel intensity by a given kernel, reducing output image dimensions.

    Parameters
    ----------
    im : ndarray
        3d array containig FLIM data (x,y,t).
    bin_size : int
        number of pixels to select before and after, bin of 3 is a 7x7 kernel
    stride : int, optional
        number of pixels to advance when doing raster calculations. The default is 1.
    pad_value : int, optional
        Value to pad the edges with if kernel needs it. The default is 0.
    debug : bool , optional
        Show debugging output. The default is False.

    Returns
    -------
    im_sum_pool : ndarray
        3d ndarray that holds a summed version of the image resized image.

    """
    if stride != 1:
        print("strides larger than 1 not yet implemented")

    kernel_length = bin_size * 2 + 1  # 7 kernel length
    # convolve with surrounding pixels

    ## Make sure image is right size or pad
    im_rows, im_cols, im_timebins = im.shape  # collapse and get 2d shape

    # COMPUTE PADDING PIXELS
    remainder_n_row_pixels = (im_rows / stride) % kernel_length
    remainder_n_col_pixels = (im_cols / stride) % kernel_length

    if remainder_n_row_pixels != 0:
        need_n_row_pixels = kernel_length - remainder_n_row_pixels
    else:
        need_n_row_pixels = 0

    if remainder_n_col_pixels != 0:
        need_n_col_pixels = kernel_length - remainder_n_col_pixels
    else:
        need_n_col_pixels = 0

    # pad image at the end of rows and cols

    rows_before = 0
    rows_after = int(need_n_row_pixels)
    cols_before = 0
    cols_after = int(need_n_col_pixels)

    padded_sdt = np.pad(
        im,  # this takes in width(cols) and height(rows)
        (
            (rows_before, rows_after),
            (cols_before, cols_after),
            (0, 0),  # dont pad 3rd dim
        ),
        "constant",
        constant_values=pad_value,
    )

    new_im_rows, new_im_cols, _ = padded_sdt.shape

    # new image placeholder
    im_sum_pool = np.zeros(
        (
            int(new_im_rows / kernel_length),
            int(new_im_cols / kernel_length),
            im_timebins,
        )
    )
    new_n_rows, new_n_cols, _new_n_timebins = im_sum_pool.shape

    # bin image
    for row in np.arange(new_n_rows):
        for col in np.arange(new_n_cols):
            pass
            sdt_block = padded_sdt[
                row * kernel_length : row * kernel_length + kernel_length,
                col * kernel_length : col * kernel_length + kernel_length,
                :,
            ]
            im_sum_pool[row, col, :] = np.sum(sdt_block, axis=(0, 1))

    if debug:
        plt.title(f"kernel_shape: {kernel_length}x{kernel_length}")
        plt.imshow(im_sum_pool.sum(axis=2))
        plt.show()

    return im_sum_pool


if __name__ == "__main__":

    ### sum_pool TEST
    HERE = Path(__file__).resolve().parent
    data = HERE.parent / r"example_data\t_cell\Tcells-001.sdt".replace("\\", "/")

    # load SDT
    im = load_sdt_file(data)
    for ch in np.arange(im.shape[0]):
        plt.title(f"channel: {ch}")
        plt.imshow(im[ch, ...].sum(axis=2))
        plt.show()

    # select channel with data
    im = im[1, ...]

    # run with debug on
    sum_pool_3d(im, bin_size=2, debug=True)
