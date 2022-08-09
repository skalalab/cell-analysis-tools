import codecs
import re
from pathlib import Path

import numpy
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from read_roi import read_roi_zip
from skimage.draw import polygon2mask


def read_asc(path):
    """
    Reads in an asc file into an array

    Parameters
    ----------
    path : pathlib.Path
        path to the file.

    Returns
    -------
    array : ndarray
        DESCRIPTION.

    """
    with codecs.open(path, encoding="utf-8-sig") as file:
        # for each line for each value, convert to float
        array = np.array([[float(x) for x in line.split()] for line in file])

    return array


if __name__ == "__main__":

    ### test loading ASC files
    test_files = Path("./test_files")
    test_files.exists()
    list_path_test_files = list(test_files.glob("*"))
    list_test_files_str = [str(p) for p in list_path_test_files]

    list_asc_files_str = list(filter(re.compile(r".*\.asc").match, list_test_files_str))
    list_asc_files = [Path(p) for p in list_asc_files_str]

    for path_file in list_asc_files:
        image = read_asc(path_file)
        plt.title(path_file.name)
        plt.imshow(image)
        plt.show()
