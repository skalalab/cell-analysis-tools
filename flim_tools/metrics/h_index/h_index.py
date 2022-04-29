import numpy as np
from sklearn.datasets import make_gaussian_quantiles

from scipy.stats import norm
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

#%%
def h_index(list_distributions):
    '''
    Computes the h-index given a mixture of ditributions
    
    R = 

    Parameters
    ----------
    list_distributions : list
        List of arrays containing subdistributions

    Returns
    -------
    float
        calculated H-index

    '''
    median_overall = np.median(list_distributions)
    
    # compute running sum
    running_sum = 0
    for distribution in list_distributions:
        pass
        p = len(distribution) / np.sum(list(map(len,list_distributions)))# proportion of subpopulation
        d = np.abs(np.median(distribution) - median_overall)# distance between median of subpop and overall med
        
        running_sum += d * p * np.log(p)
    return -running_sum
#%%
    
if __name__ == "__main__":
    pass
   #%% 
    x = np.linspace(1,100,num=1000)

    #distributions to create
    list_dist_params = [
       #(mean, std)
        (20,4),
        (30,6),
        (50,20),
        # (50,10),
        ]
    
    dist_all = np.zeros_like(x)
    list_distributions = []
    
    for dist_params in list_dist_params:
        pass
        dist = norm(loc=dist_params[0], scale=dist_params[1])
        list_distributions.append(dist)
        dist_all = dist_all + dist.pdf(x)
    
    # plot dataset
    plt.title("Distributions")
    for pos, dist in enumerate(list_distributions,start=1):
        pass
        plt.plot(x, dist.pdf(x), label=f"dist {pos} | mean: {dist.mean()} | stdev: {dist.std()}")
    plt.plot(x, dist_all, label="all distributions")
    plt.legend()
    plt.show()
    
    ## try out function
    #generate random values
    list_dist_values = [d.rvs(len(x)) for d in list_distributions]
    
    list_dist_values[0].mean()
    # h_index_value = h_index(list_distributions)
    # print(h_index_value)
    
    #%%
    ## plot one two three distributions
    plt.title("Distributions")
    
    for num_dist in np.arange(start=1, stop=4): #slide in dist 
        pass
        list_subdists = list_distributions[:num_dist]
        
        # compute h_index
        list_subdists_values = [dist.rvs(len(x)) for dist in list_subdists]
        h_index_value = h_index(list_subdists_values)
        
        #plot pdf
        pdf_combined = [d.pdf(x) for d in list_subdists]
        pdf_combined_values = np.sum(pdf_combined, axis=0)
        plt.plot(x, pdf_combined_values, label=f"dist {num_dist} | Hi: {h_index_value:.5f}")
    plt.legend()
    plt.show()

        

#%%

# compare graphs with gaussianmixture fit using the same number of distributions
# 
    
    
