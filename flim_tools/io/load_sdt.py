import zipfile

import numpy
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from read_roi import read_roi_zip
from skimage.draw import polygon2mask


def load_sdt_file(file_path):
    """ load a new sdt file
    Parameters: 
        file_path (string): path to the sdt file
    
    Returns:
        image = 3d-array representation of image
    """

    with zipfile.ZipFile(file_path) as myzip:
        z1 = myzip.infolist()[
            0
        ]  # "data_block" or sdt bruker uses "data_block001" for multi-sdt"
        with myzip.open(z1.filename) as myfile:
            data = myfile.read()
            data = np.frombuffer(data, np.uint16)

    # format == [channel, x, y, num_timebins]

    # todo
    """  figure out how to extract this info from sdt file"""
    img_1_256_256_256 = 16777216
    img_1_512_512_256 = 67108864
    img_2_512_512_256 = 134217728  # array size
    img_3_512_512_256 = 201326592
    img_2_256_256_256 = 2 * 256 * 256 * 256
    if len(data) == img_2_512_512_256:
        c, x, y, z = (2, 512, 512, 256)
    if len(data) == img_1_256_256_256:
        c, x, y, z = (1, 256, 256, 256)
    if len(data) == img_1_512_512_256:
        c, x, y, z = (1, 512, 512, 256)
    if len(data) == img_3_512_512_256:
        c, x, y, z = (3, 512, 512, 256)
    if len(data) == img_2_256_256_256:
        c, x, y, z = (2, 256, 256, 256)

        # if c == 1:
        #     ''' order is [XYT] '''
        #     numpy_image = np.reshape(data, (x, y, z))
        #     # if debug: tif.imshow(np.sum(image,axis=2))
        #     return numpy_image

        # else:
        """ order is [CXYT] """
    numpy_image = np.reshape(data, (c, x, y, z))
    return np.float32(numpy_image)


def load_sdt_data(filepath):
    """
    Loads the data of an SDT file, reshaping the output is necessary as this
    outputs a 1d array

    e.g.
    An image of length 16777216 should be reshaped to (1, 256,256,256)
    >>> image = np.reshape(data, (1, 256, 256, 256))

    ### 2ch
    An image of length 33554432 should be reshaped to (2 ch, 256,256,256)
    >>> image = np.reshape(data, (2, 256, 256, 256))

    Parameters
    ----------
    filepath : string, pathlib path
        path to the input image

    Returns
    -------
    data : TYPE
        1D array containing all channel and pixel data

    """
    with zipfile.ZipFile(filepath) as myzip:
        z1 = myzip.infolist()[
            0
        ]  # "data_block" or sdt bruker uses "data_block001" for multi-sdt"
        with myzip.open(z1.filename) as myfile:
            data = myfile.read()
            data = np.frombuffer(data, np.uint16)

    return data
