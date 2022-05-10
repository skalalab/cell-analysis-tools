from typing import Union
from .read_asc import read_asc
import numpy as np
import pathlib
from pathlib import Path
import tifffile

def load_image(path : Union[str, pathlib.PurePath])-> np.ndarray:
    """
    Detects the extension and loads image accordingly
    if its a tif/tiff or an asc

    Parameters
    ----------
    path : pathlib path or str
        path to the image.

    Returns
    -------
    np.ndarray
        array containig image.

    """
    if not isinstance(path, pathlib.PurePath):
        path = Path(path)
    pass
    if path.suffix == ".asc":
        return read_asc(path)
    if path.suffix in [".tiff", ".tif"]:
        return tifffile.imread(path)