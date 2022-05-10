from flim_tools.image_processing import (
                                        normalize,
                                        bin_2d,
                                        bin_3d,
                                        sum_pool_3d,
                                        normalize,
                                        kmeans_threshold,
                                        rgb2labels,
                                        rgb2gray,
                                        remove_horizontal_vertical_edges,
                                        fill_and_label_rois
                                        )
import numpy as np
import matplotlib.pylab as plt
from pathlib import Path
import tifffile

class TestImageProcessing():
    
    #generate random fixed image
    size = 128
    rng = np.random.default_rng(seed = 0)
    im_2d = rng.random((size,size))
    im_3d = rng.random((size,size,size))
    
    HERE = Path(__file__).absolute().resolve().parent
    # load test images  
    path_resources = HERE / "resources"
    path_bin_2d = path_resources  / "bin_2d.tiff"
    im_bin_2d = tifffile.imread(path_bin_2d)

    path_bin_3d = path_resources / "bin_3d.tiff"
    im_bin_3d = tifffile.imread(path_bin_3d)
    
    
    path_normalize = path_resources / "normalize.tiff"
    im_normalize = tifffile.imread(path_normalize)
    
    path_kmeans_threshold = path_resources / "kmeans_threshold.tiff"
    im_kmeans_threshold = tifffile.imread(path_kmeans_threshold)
    
    # GENERATE TEST IMAGES
    path_resources = Path(r"C:\Users\econtrerasguzman\Desktop\development\flim_tools\pytest\unit\resources")
    
    im_gt_kmeans = kmeans_threshold(im_2d, k=6, n_brightest_clusters=1)
    tifffile.imwrite(path_resources / "kmeans_threshold.tiff" , im_gt_kmeans )
    plt.imshow(im_gt_kmeans)
    
    im_gt_bin_2d = bin_2d(im_2d, bin_size=3)
    plt.imshow(im_gt_bin_2d)
    tifffile.imwrite(path_resources / "bin_2d.tiff" , im_gt_bin_2d )

    im_gt_bin_3d = bin_3d(im_3d, bin_size=3)
    plt.imshow(im_gt_bin_3d.sum(axis=2))
    tifffile.imwrite(path_resources / "bin_3d.tiff" , im_gt_bin_3d )
    
    im_gt_normalize = normalize(im_2d)
    plt.imshow(im_gt_normalize)
    tifffile.imwrite(path_resources / "normalize.tiff" , im_gt_normalize)
    
 
    def test_bin_2d(self):
        
        results_bin_2d = bin_2d(self.im_2d, bin_size=3)
        
        assert results_bin_2d.all() == self.im_bin_2d.all()
    
    
    def test_bin_3d(self):
        
        results_bin_3d = bin_3d(self.im_3d, bin_size=3)
        
        assert results_bin_3d.all() == self.im_bin_3d.all()
        
    def test_normalize(self):
        assert self.im_normalize.all() == normalize(self.im_2d).all()

    def test_kmeans_threshold(self):
        assert self.im_kmeans_threshold.all() == kmeans_threshold(self.im_2d, k=6, n_brightest_clusters=1).all()
      
      # test_sum_pool_3d,
      # test_rgb2labels,
      # test_rgb2gray,
      # test_remove_horizontal_vertical_edges,
      # test_fill_and_label_rois
     
     
