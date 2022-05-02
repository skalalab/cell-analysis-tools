import matplotlib.pylab as plt


def image_show(image):
    """

    Parameters
    ----------
    image : ndarray
        image to show.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(image)  # , cmap='gray'
    ax.axis("off")
    plt.show()
    return fig, ax


def compare_images(im1, title1, im2, title2, figsize=(10, 5))->None:
    """
     
        Parameters
        ----------
        im1 : np.ndarray
            image 1.
        title1 : str
            title for image 1.
        im2 : np.ndarray
            image 2.
        title2 : str
            title for image 2.
        figsize : TYPE, optional
            size of figure. The default is (10, 5).
    
        Returns
        -------
        None.
    
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].title.set_text(title1)
    ax[0].imshow(im1)
    ax[0].set_axis_off()

    ax[1].title.set_text(title2)
    ax[1].imshow(im2)
    ax[1].set_axis_off()
    plt.show()
