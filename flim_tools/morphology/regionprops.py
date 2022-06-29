import numpy as np
from numpy.ma import masked_array
from skimage.measure import label
from skimage.measure import regionprops as _regionprops
from skimage.measure import regionprops_table as _regionprops_table
from skimage.morphology import label

from .fractal_dimension.fractal_dim_gray import fractal_dimension_gray
from .intensity_sum import intensity_sum
from .roi_distance import radius_max, radius_mean, radius_median


def regionprops(label_image, intensity_image=None):
    """ Extended regionprops function adding our own props    
    
    To see a complete docstring for this function see regionprops skimage.

    Notes
    -----
    Additional properties that can be accessed as attributes or keys:

    **radius_max** : float
        The maximum distance of any pixel in the segmented region to the closest background pixel.
    **radius_mean** : float
        Mean value the distances of all pixels in the segmented region to their closest background pixel.
    **radius_median** : float
        Median value the distances of all pixels in the segmented region to their closest background pixel.
    **intensity_sum**
        The sum of the pixel intensities within the segmented region.
    **fractal_dimension** : float
        Fractal dimension, differential box counting method implementation

    """
    return _regionprops(
        label_image,
        intensity_image=intensity_image,
        extra_properties=(
            radius_mean,
            radius_median,
            radius_max,
            intensity_sum,
            fractal_dimension_gray,
        ),
    )


def regionprops_table(label_image, intensity_image, properties=None):

    return _regionprops_table(
        label_image,
        intensity_image=intensity_image,
        properties=properties,
        extra_properties=(
            radius_mean,
            radius_median,
            radius_max,
            intensity_sum,
            fractal_dimension_gray,
        ),
    )


if __name__ == "__main__":

    import matplotlib as mpl
    import matplotlib.pylab as plt
    from skimage.draw import ellipse

    # import numpy as np

    mpl.rcParams["figure.dpi"] = 300

    idx_rows, idx_cols = ellipse(20, 20, 5, 7)
    shape_ellipse = np.zeros((40, 40))
    shape_ellipse[idx_rows, idx_cols] = 1

    import numpy as np

    rng = np.random.default_rng(seed=0)

    intensity = rng.random(shape_ellipse.shape) * shape_ellipse
    intensity = intensity * 255

    plt.imshow(shape_ellipse)
    plt.show()
    shape_ellipse = shape_ellipse.astype(int)

    labels = label(shape_ellipse)
    props = regionprops_table(
        labels,
        intensity,
        properties=[
            "area",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "orientation",
            "solidity",
            "extent",
            "perimeter",
            "radius_max",
            "radius_mean",
            "radius_median",
            "fractal_dimension_gray",
        ],
    )
