def magnitude(roi):
    """The magnitude of a vector, which can be computed at either a pixel or mitochondrion level.

    """
    raise NotImplementedError


def angle(roi):
    """The angle of a vector, which can be computed at either a pixel or mitochondrion level.

    """
    raise NotImplementedError


def dx(roi):
    """The x component of a motion vector, which can be computed at either a pixel or mitochondrion level.

    """
    raise NotImplementedError


def dy(roi):
    """The y component of a motion vector, which can be computed at either a pixel or mitochondrion level.

    """
    raise NotImplementedError


def median_std_intensity(roi):
    """The median of standard deviation of intensity values in a segmented region. These values are obtained by computing the standard deviation of the intensity values of a pixel across all frames of a video.

    """
    raise NotImplementedError
