import numpy as np
import pandas as pd
import umap

#%% UMAP


def compute_umap(data_values: np.ndarray, **kwargs) -> pd.DataFrame:
    """
        Parameters
        ----------
        data_values : np.ndarray
            array of values to input into the reducer. Rows should be roi values,
            columns are features
        random_state : int, optional
            allows consistent initialization of reducer. The default is 0.
        **kwargs : dict
            additional parameters can be passed in here for the reducer.
        
        Returns
        -------
        df_umap : pd.DataFrame
            DataFrame containing umap embeddings.
        reducer : Object
            umap reducer object.
            
        .. note::
            See the umap-learn documentation for default and additional parameters
            `https://umap-learn.readthedocs.io/en/latest/api.html <https://umap-learn.readthedocs.io/en/latest/api.html>`_
            
        
        """

    reducer = umap.UMAP(random_state=0, **kwargs)
    fit_umap = reducer.fit_transform(data_values)
    df_umap = pd.DataFrame(fit_umap, columns=["umap_x", "umap_y"])
    return df_umap, reducer


#%% PCA

from sklearn.decomposition import PCA


def compute_pca(
    data_values: np.ndarray, n_components: int = 2, **kwargs
) -> pd.DataFrame:
    """
    Computes PCA given input data

    Parameters
    ----------
    data_values : np.ndarray
        array of values to input into the reducer. Rows should be roi values,
        columns are features
    n_components : int, optional
        number of components for the PCA. The default is 2.

    Returns
    -------
    df_pca : pd.DataFrame
        Dataframe containing the components of the PCA.
    pca : Object
        PCA reducer object.

    """
    pca = PCA(n_components=n_components, **kwargs)
    principal_components = pca.fit_transform(data_values)
    df_pca = pd.DataFrame(
        data=principal_components,
        columns=["principal component 1", "principal component 2"],
    )
    return df_pca, pca


#%% tSNE

from sklearn import manifold


def compute_tsne(
    data_values: np.ndarray, random_state: int = 0, **kwargs
) -> np.ndarray:

    """
    Computes t-distributed stochastic neighbor embedding. 

    Parameters
    ----------
    data_values : np.ndarray
        array of values to input into the reducer. Rows should be roi values,
        columns are features.
    random_state : int, optional
        allows consistent initialization of reducer. The default is 0.

    Returns
    -------
    df_tsne : pd.DataFrame
        Dataframe containing the embeddings of tsne.
    tsne : Object
        tsne reducer object.

    """

    tsne = manifold.TSNE(random_state=random_state, **kwargs)
    principal_components = tsne.fit_transform(data_values)
    df_tsne = pd.DataFrame(data=principal_components, columns=["tsne_1", "tsne_2"])

    return df_tsne, tsne


#%% plot data

import matplotlib as mpl

# plotting
import matplotlib.pylab as plt

mpl.rcParams["figure.dpi"] = 300


def plot_data(data, labels, s=3, title=""):

    data["labels"] = labels
    dict_classes = {0: "class 1", 1: "class 2"}

    # dict_colors = { 0 : 'b', 1 : 'g'}

    for label in np.unique(data["labels"]):
        pass
        df_plot = data[data["labels"] == label]
        x = df_plot.iloc[:, 0]
        y = df_plot.iloc[:, 1]
        scatter = plt.scatter(
            x,
            y,
            # c=list_colors,
            label=dict_classes[label],
            s=s,
        )
    plt.title(title)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.legend()
    plt.show()


#%%
if __name__ == "__main__":
    #%%
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    x, y = make_classification(
        n_classes=2,
        n_features=10,  # e.g. omi parameters
        n_samples=1000,
        random_state=0,
    )

    list_features = [
        # 'nadh_intensity_mean',
        "nadh_a1_mean",
        "nadh_a2_mean",
        "nadh_t1_mean",
        "nadh_t2_mean",
        "nadh_tau_mean_mean",
        # 'fad_intensity_mean',
        "fad_a1_mean",
        "fad_a2_mean",
        "fad_t1_mean",
        "fad_t2_mean",
        "fad_tau_mean_mean",
        # 'redox_ratio_mean'
    ]

    df_data = pd.DataFrame(x, columns=list_features)

    scaled_data = StandardScaler().fit_transform(x)

    # umap
    df_umap, reducer_umap = compute_umap(scaled_data, metric="manhattan")
    plot_data(df_umap, labels=y, title="UMAP")

    # tsne
    df_tsne, reducer_tsne = compute_tsne(scaled_data)
    plot_data(df_tsne, labels=y, title="t-SNE")

    # pca
    df_pca, reducer_pca = compute_pca(scaled_data)
    plot_data(df_pca, labels=y, title="Principal Component Analysis")
