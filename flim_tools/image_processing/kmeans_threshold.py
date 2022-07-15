import matplotlib.pylab as plt
import numpy as np
from sklearn.cluster import KMeans

from .normalize import normalize


def kmeans_threshold(im, k, n_brightest_clusters, show_image=False):
    """
    Given an image, this function will apply k-means clustering and return the mask that 
    includes the n brightest clusters
    Parameters
    ----------
    im : array-like
        intensity image.
    k : int, 
        number of clusters for k means algorithm
    n_brightest_clusters : int, 
        number of brightest clusers to keep. Must be < k.
        
    show_image : bool, optional
        When debugging,this will display the original image next to the k means thresholded image. The default is False.

    Returns
    -------
    mask : int
        binary mask

    """
    # k = 3
    # n_brightest_clusters = 1 # this should be < k

    if n_brightest_clusters >= k:
        print("n_brightest_clusters must be < k")
        return

    # normalize to 0 and 1
    # im = (im - im.min()) / (im.max() - im.min())  # subtract baseline / new_max value)
    im = normalize(im)

    rows, cols = im.shape
    X = im.reshape((rows * cols, 1))  # reshape to (n_samples, n_features)

    # apply k means to image
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    #### keep brightest clusters by zeroing out dimmestclusters
    num_blanked_dimmer_clusters = (
        k - n_brightest_clusters
    )  # calculate n_dimm clusters based on clusters to keep
    # get list of clusters, first n are dimmest/smallest
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    n_dimmest_clusters = np.argpartition(
        kmeans.cluster_centers_[:, 0], num_blanked_dimmer_clusters
    )
    indices_smallest = n_dimmest_clusters[
        :num_blanked_dimmer_clusters
    ]  # get a hold of n dimmest clusters

    # make dimmest clusters == 0
    newLUT = kmeans.cluster_centers_.copy()
    for x in indices_smallest:
        newLUT[x, :] = 0

    # replace labels with cluster values
    clustered_im = newLUT[kmeans.labels_]
    ####

    # Reshape back the image from 2D to 3D image
    clustered_imaged = clustered_im.reshape(rows, cols)
    mask = (clustered_imaged > 0).astype(int)  # make a binary mask

    if show_image:
        plt.title(f"num clusters (k): {k} , clusters kept: {n_brightest_clusters}")
        divider = np.ones((rows, 5))
        plt.imshow(np.c_[im, divider, mask])
        plt.show()

    return mask
