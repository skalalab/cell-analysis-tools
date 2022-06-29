import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from flim_tools.metrics import dice, jaccard, total_error

mpl.rcParams["figure.dpi"] = 300


class TestMetrics:

    # generate masks to use for testing
    size = 512
    mask_a = np.ones((size, size))
    mask_b = np.ones((size, size))

    mask_b[size // 2 :, ...] = 0

    ### test of other metrics
    w1 = 100
    w2 = 1
    mask_gt = np.zeros((10, 10))
    mask_gt[:, :5] = 1

    mask_predicted = np.zeros((10, 10))
    mask_predicted[:, :] = 1

    # show masks
    # fix, ax = plt.subplots(1,2)
    # ax[0].title.set_text("ones \n mask_a")
    # ax[0].imshow(mask_a)
    # ax[1].title.set_text("top half are ones \n mask_b")
    # ax[1].imshow(mask_b)
    # plt.show()

    def test_dice(self):

        dice_coeff = dice(self.mask_a, self.mask_b)
        assert dice_coeff == (2 / 3), "incorrect dice score"

    def test_jaccard(self):
        jaccard_idx = jaccard(self.mask_a, self.mask_b)
        assert jaccard_idx == 0.5, "incorrect jaccard index"

    def test_percent_error(self):
        percent_error = total_error(self.mask_a, self.mask_b)
        assert percent_error == 0.5, "incorrect total error for masks"

    def test_total_error(self):
        tot_error = total_error(
            self.mask_gt, self.mask_predicted, weight_fn=1, weight_fp=1
        )
        assert tot_error == 0.5, "incorrect total score"
