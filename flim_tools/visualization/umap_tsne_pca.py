import pandas as pd
import numpy as np


#%% UMAP

import umap
def compute_umap(data_values: np.ndarray,
        min_dist : float=0.1,
        n_neighbors : int=15,
        metric : str='euclidean',
         **kwargs) -> pd.DataFrame:
    
        reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,   
                metric=metric,
                n_components=2,
                random_state=0,
                **kwargs
            )
        fit_umap = reducer.fit_transform(data_values)
        df_umap = pd.DataFrame(fit_umap, columns=["umap_x",
                                                  "umap_y"])
        return df_umap, reducer
#%% PCA

from sklearn.decomposition import PCA

def compute_pca(data_values: np.ndarray, n_components: int=2)-> pd.DataFrame:
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_values)
    df_pca = pd.DataFrame(data = principal_components,
                  columns = ['principal component 1', 
                             'principal component 2'])
    return df_pca

#%% tSNE

from sklearn import manifold

def compute_tsne(data_values: np.ndarray,
         n_components : int=2,
         init : str='pca',
         random_state: int=0)->np.ndarray:
    # data values should already be normalized
    
    tsne = manifold.TSNE(n_components=n_components, 
                         init=init,
                         random_state=random_state,
                         )
    principal_components = tsne.fit_transform(data_values)
    df_tsne = pd.DataFrame(data = principal_components
                 , columns = ['tsne_1', 'tsne_2'])
    return df_tsne


#%% plot data 

# plotting
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300   

def plot_data(data, labels, s=3, title=""):
    
    data['labels'] = labels
    dict_classes = {0 : "class 1", 1 : "class 2"}
 
    # dict_colors = { 0 : 'b', 1 : 'g'}
    
    for label in np.unique(data['labels']):
        pass
        df_plot = data[data['labels']==label]
        x=df_plot.iloc[:,0]
        y=df_plot.iloc[:,1]
        scatter = plt.scatter(x,
                    y, 
                    # c=list_colors,
                    label= dict_classes[label],
                    s=s
                    )
    plt.title(title)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.legend()
    plt.show()

#%%
if __name__ == "__main__":
    #%%
    from sklearn.datasets import (make_classification)
    
    from sklearn.preprocessing import StandardScaler
    
    x, y = make_classification(
    n_classes=2, 
    n_features=13, # e.g. omi parameters
    n_samples=1000,
    random_state=1)
    
    list_features = [
        'nadh_intensity_mean',
        'nadh_a1_mean',  
        'nadh_a2_mean',
        'nadh_t1_mean',  
        'nadh_t2_mean',
        'nadh_tau_mean_mean', 
        'fad_intensity_mean',  
        'fad_a1_mean',
        'fad_a2_mean',  
        'fad_t1_mean',
        'fad_t2_mean',  
        'fad_tau_mean_mean',
        'redox_ratio_mean'
        ]
    
    df_data = pd.DataFrame(x, columns=list_features)
        
    scaled_data = StandardScaler().fit_transform(x)


    # umap
    df_umap = compute_umap(scaled_data)
    plot_data(df_umap, labels=y, title="UMAP")
    
    # tsne
    df_tsne = compute_tsne(scaled_data)
    plot_data(df_tsne, labels=y, title="t-SNE")
    
    # pca
    df_pca = compute_pca(scaled_data)
    plot_data(df_pca, labels=y, title="Principal Component Analysis")








