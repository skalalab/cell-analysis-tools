

def fractal_dimension():
    """Note, this is an implementation of the differential box counting
    method for determining fractal dimension

    """
    raise NotImplementedError


def fractal_dimension_mean(roi):
    """The mean value of the fractal dimensions of each pixel in a segmented region. Higher values represent more complex patterns.
    """
    raise NotImplementedError 

def fractal_dimension_std(roi):
    """The standard deviation of the fractal dimensions of each pixel in a segmented region.
    """
    raise NotImplementedError

def fractal_dimension_lacunarity(roi):
    """The squared value of the standard deviation of fractal dimension divided by the mean fractal dimension. Lower value represents a dense pattern while higher values represent more open patterns.

    """
    raise NotImplementedError
