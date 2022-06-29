#%%
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import norm

mpl.rcParams["figure.dpi"] = 300

from sklearn.mixture import GaussianMixture


#%%
def h_index(list_distributions) -> float:
    """
    Computes the h-index given a mixture of ditributions.

    Parameters
    ----------
    list_distributions : list
        List of np.arrays containing estimated subdistributions

    Returns
    -------
    float
        calculated H-index
        
    Notes
    -----    
    When comparing H-index between datasets be sure to set the n_components
    paramter of the GaussianMixtureModel to the same value.
       
    References
    ----------
    http://www.microscopist.co.uk/wp-content/uploads/2021/06/FLIM-review.pdf

    """
    median_overall = np.median(np.concatenate(list_distributions))

    # compute running sum
    running_sum = 0
    for distribution in list_distributions:
        pass
        p = len(distribution) / np.sum(
            list(map(len, list_distributions))
        )  # proportion of subpopulation
        d = np.abs(
            np.mean(distribution) - median_overall
        )  # distance between median of subpop and overall med
        running_sum += d * p * np.log(p)
    return -running_sum


#%%


def h_index_single_weighted(list_distributions) -> float:
    """
    Computes the single weighted h-index given a mixture of ditributions.

    Parameters
    ----------
    list_distributions : list
        List of np.arrays containing estimated subdistributions

    Returns
    -------
    float
        calculated H-index
        
    Notes
    -----
    When comparing H-index between datasets be sure to set the n_components
    paramter of the GaussianMixtureModel to the same value.

    """
    median_overall = np.mean(np.concatenate(list_distributions))

    # compute running sum
    running_sum = 0
    for distribution in list_distributions:
        pass
        p = len(distribution) / np.sum(
            list(map(len, list_distributions))
        )  # proportion of subpopulation
        d = np.abs(
            np.mean(distribution) - median_overall
        )  # distance between median of subpop and overall med
        sigma = np.std(distribution)
        rng = np.random.default_rng(seed=1)
        im = rng.random((128, 128))
        running_sum += (1 - (p * np.log(p + 1))) * (sigma + d)

    return running_sum


#%%
if __name__ == "__main__":
    pass
    #%% GENERATE DISTRIBUTIONS FOR TESTING
    x = np.linspace(1, 100, num=1000)

    # distributions to create
    list_dist_params = [
        # (mean, std)
        (20, 4),
        # (20,6),
        # (20,10),
        # (20,10),
        (20, 4),
        # (30,6),
        (50, 20),
        (50, 10),
    ]

    # list_dist_params = [
    #    #(mean, std)

    #     ]

    dist_all = np.zeros_like(x)
    list_distributions = []

    for dist_params in list_dist_params:
        pass
        mean, sigma = dist_params
        dist = norm(loc=mean, scale=sigma)
        list_distributions.append(dist)
        dist_all = dist_all + dist.pdf(x)

    # plot dataset
    plt.title("Distributions")
    for pos, dist in enumerate(list_distributions, start=1):
        pass
        plt.plot(
            x,
            dist.pdf(x),
            label=f"dist {pos} | mean: {dist.mean()} | stdev: {dist.std()}",
        )
    plt.plot(x, dist_all, label="all distributions")
    plt.xlabel("value")
    plt.ylabel("probability")
    plt.legend()
    plt.show()

    ## try out function
    # generate random values
    list_dist_values = [d.rvs(len(x)) for d in list_distributions]

    list_dist_values[0].mean()
    plt.tight_layout()
    # h_index_value = h_index(list_distributions)
    # print(h_index_value)

    #%% COMBINE DISTRIBUTIONS AND COMPUTE WEIGHTED AND SINGLY WEIGHTED H-INDEX

    ## plot one two three distributions
    n_components = 3

    plt.title(f"Distributions \nGaussian Mixture Model components : {n_components} ")

    for num_dist in np.arange(start=1, stop=5):  # slide in dist
        pass
        list_subdists = list_distributions[:num_dist]

        # compute h_index
        list_subdists_values = [dist.rvs(len(x)) for dist in list_subdists]

        # compute gaussiam mixture model
        all_data = np.array(list_subdists_values).reshape(-1, 1)
        # plt.hist(all_data, histtype="step")
        # plt.show()

        # fit data into gaussian mixtures
        # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4
        # https://stats.stackexchange.com/questions/139163/selecting-the-number-of-mixtures-hidden-states-latent-variables
        # https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
        gauss = GaussianMixture(n_components=n_components)
        labels = gauss.fit_predict(all_data)

        list_fit_dist = [
            np.asarray(all_data[labels == label]).flatten()
            for label in np.unique(labels)
        ]

        ## GAUSSIAN MIXTURE MODEL
        # plt.title(f"histogram of fit data | fixed GMM | components: {len(np.unique(labels))}")
        # for list_data in list_fit_dist:
        #     plt.hist(list_data, bins=100, histtype='step', label=f'mean: {np.mean(list_data):.3f} | weight:{(len(list_data)/len(all_data)):.2f}')
        #     plt.ylabel("count")
        #     plt.xlabel("value")
        # plt.legend()
        # plt.show()

        ##### BAYESIAN GAUSSIAN MIXTURE MODEL

        # from sklearn.mixture import BayesianGaussianMixture
        # bgmm = BayesianGaussianMixture(n_components=4,
        #                                max_iter=1000
        #                             )
        # model_labels = bgmm.fit_predict(all_data)
        # list_fit_dist = [np.asarray(all_data[model_labels == label]).flatten() for label in np.unique(model_labels)]

        # plt.title(f"histogram of fit data Bayesian | components: {len(np.unique(model_labels))} ")
        # for idx, list_data in enumerate(list_fit_dist):
        #     pass
        #     plt.hist(list_data, bins=100, histtype='step', label=f'{idx} | {np.mean(list_data):.3f} | weight:{(len(list_data)/len(all_data)):.2f}')
        #     plt.ylabel("count")
        #     plt.xlabel("value")
        # plt.legend()
        # plt.show()

        #############

        h_index_value = h_index(list_fit_dist)
        h_index_single_weighted_value = h_index_single_weighted(list_fit_dist)

        # plot pdf
        pdf_combined = [d.pdf(x) for d in list_subdists]
        pdf_combined_values = np.sum(pdf_combined, axis=0)
        plt.plot(
            x,
            pdf_combined_values,
            label=f"dist {num_dist} | Hi: {h_index_value:.2f} | wHi: {h_index_single_weighted_value:.2f}",
        )
    plt.legend()
    plt.show()
# %%
