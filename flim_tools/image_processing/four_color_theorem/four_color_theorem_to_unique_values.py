from four_color_theorem.four_color_theorem import four_color_theorem
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import tifffile
import numpy as np
from skimage.morphology import dilation, disk
from scipy.ndimage import label
from pprint import pprint
from collections import OrderedDict


mask = tifffile.imread("mask.tiff")

plt.imshow(mask)

mask_fc , _ = four_color_theorem(mask)


