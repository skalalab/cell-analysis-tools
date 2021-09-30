# import numpy as np
# from scipy.spatial.distance import directed_hausdorff
# from .helper import _validate_array_and_make_bool
# from flim_tools.metrics import dice


# if __name__ == "__main__":


#     import matplotlib.pylab as plt
#     import matplotlib as mpl
#     mpl.rcParams["figure.dpi"] = 300

#     size = 512
#     mask_a = np.ones((size, size))
#     mask_b = np.ones((size, size))

#     mask_b[size//2:,...] = 0

#     fix, ax = plt.subplots(1,2)
#     ax[0].title.set_text("ones \n mask_a")
#     ax[0].imshow(mask_a)
#     ax[1].title.set_text("top half are ones \n mask_b")
#     ax[1].imshow(mask_b)
#     plt.show()

#     dice_coeff = dice(mask_a, mask_b)
#     assert dice_coeff == (2/3), "incorrect dice score"

#     jaccard_idx = jaccard(mask_a, mask_b)
#     assert jaccard_idx == 0.5, "incorrect jaccard index"

#     percent_error = total_error(mask_a, mask_b)
#     assert percent_error == 0.5, "incorrect total error for masks"


#     #### test of other metrics
#     w1=100
#     w2=1
#     mask_gt = np.zeros((10,10))
#     mask_gt[:,:5] = 1

#     mask_predicted = np.zeros((10,10))
#     mask_predicted[:,:] = 1

#     # total error
#     te = total_error(mask_gt, mask_predicted, weight_fn=1, weight_fp=1)
#     assert te == 0.5, "incorrect total score"

#     # average performance
#     ## TODO check this, it's off
#     mask_gt1 = np.zeros((10,10))
#     mask_gt1[:,:5] = 1

#     mask_gt2 = np.zeros((10,10))
#     mask_gt2[:5,:] = 1

#     mask_predicted = np.zeros((10,10))
#     mask_predicted[:,:] = 1
#     ap = average_relative_performance(mask_gt1, mask_gt2, mask_predicted)
#     print(ap)
