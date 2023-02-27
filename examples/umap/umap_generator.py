import numpy as np
import pandas as pd
import umap

from cell_analysis_tools.visualization import (compute_tsne, 
                                               compute_umap, 
                                               compute_pca)


#%% plot data

import matplotlib as mpl

# plotting
import matplotlib.pylab as plt

mpl.rcParams["figure.dpi"] = 300


def plot_data(data, labels, s=3, title=""):

    data["labels"] = labels
    dict_classes = {0: "class 1", 1: "class 2"}


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
