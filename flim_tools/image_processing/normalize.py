def normalize(im):
    """
    Normalizes an image to 0 and 1 by subtrating min value and 
    dividing by new max value.
    
    Parameters
    ----------
    im : array-like
        intensity image.

    Returns
    -------
    array-like
        array normalized to 0 and 1.

    """
    return (im - im.min()) / (im.max() - im.min())
