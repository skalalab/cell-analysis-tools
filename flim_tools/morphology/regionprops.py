import numpy as np
from skimage.measure import regionprops as _regionprops
from skimage.measure import regionprops_table as _regionprops_table
from skimage.measure import label
from numpy.ma import masked_array
# from ._fractal_dimension import fractal_dimension
from ._fractal import dbc as fractal_dimension
from ._roi_distance import radius_max, radius_mean, radius_median
from .intensity_sum import intensity_sum


def regionprops(label_image, intensity_image=None):
    """ Extended regionprops function adding our own props    
    
    To see a complete docstring for this function see `skimage.measure.regionprops https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops`_.

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
    return _regionprops(label_image, intensity_image=intensity_image, extra_properties=(
        fractal_dimension,
        radius_mean,
        radius_median,
        radius_max,
        intensity_sum,
        )
    )

def regionprops_table(label_image, intensity_image, properties=None ):


        return _regionprops_table(label_image, intensity_image=intensity_image, 
                                    properties=properties, extra_properties=(
        # fractal_dimension,
        radius_mean,
        radius_median,
        radius_max,
        intensity_sum,
        )
    ) 


# Fractal dimension
# mean 
    #**fractal_dimension_mean** : float
    #         the mean fractal dimension was computed to measure the level of mitochondrial complexity at the whole-cell level
    #**fractal_dimension_std** : float
    #         The mean fractal dimension was computed to measure the level of mitochondrial complexity at the whole-cell level 
    # ** Fractal Dimension Lacunarity     
    # The squared value of the standard deviation of fractal dimension divided by the mean fractal dimension. Lower value represents a dense pattern while higher values represent more open patterns.
    # Mitochondria Morphological Class -- The punctate, swollen, and networked morphologies.





if __name__ == "__main__":

    from skimage.draw import ellipse
    import matplotlib.pylab as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["figure.dpi"] = 300

    idx_rows, idx_cols = ellipse(20, 20, 5, 7)
    shape_ellipse = np.zeros((40, 40))
    shape_ellipse[idx_rows, idx_cols] = 1

    plt.imshow(shape_ellipse)
    plt.show()
