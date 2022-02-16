###UMAP Coordinate calculations

#Loads in all required dependencies
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
from sklearn.model_selection import ParameterGrid
from pathlib import Path
#%% From Tiffany
# I asked how to select clusters that aren't "forced"  

# My assessment of “real” vs. “forced” clustering more comes from biological 
# understanding of the data and the resolution/sensitivity of the measurements. 
# For our OMI data, a more conservative approach is typically needed to avoid overfitting the data.
#  That’s why I’m normally more conservative on the selection of the min_dist parameter (esp over the default = 0.1).
#  Also since the UMAP projection actually preserves some meaning about how similar data points
#  are in the separation of clusters, I think it is somewhat informative to evaluate 
#  when testing parameters. If it’s pretty well separated even at more conservative 
#  min_dist values, I’d be more confident in that clustering. I think a more analytical
#  approach explanation can be found here: https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668 . 
 
# In general, I find UMAP is better as a way to qualitatively inform on 
# how the data represents/stratifies your experimental conditions. Then follow that
#  up with a more quantitative evaluation of your group (discriminant analysis, classification approaches).
#  But those may just be my own cautious biases :) 

#%% SELECT DATA
# UMAP documentation Penguin
# https://umap-learn.readthedocs.io/en/latest/basic_usage.html#penguin-data

df_data = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
print(df_data.head())

df_data = df_data.dropna()

# MAKE CHECKBOXES TO SELECT COLUMNS 
penguin_cols = list(df_data.keys()) 
remove_entries = ["species_short", "island", "sex"] # keys to remove
list(map(lambda item : penguin_cols.remove(item), remove_entries)) # remove entries 
print(penguin_cols)

# sns.pairplot(penguins, hue='species_short')
# plt.show()

penguin_data = df_data[penguin_cols].values
scaled_data = StandardScaler().fit_transform(penguin_data)


#%%

df_redox_ratio = pd.read_csv("2022_02_08_all_props.csv")
# (27,35,37,38)
# df_redox_ratio.iloc[:,35]

list_omi_parameters = [
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

df_data = df_redox_ratio[df_redox_ratio["treatment"] == "0-control"]

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
#%% FIT UMAP 

# reducer = umap.UMAP()
reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=0
    )

fit_umap = reducer.fit(scaled_data)
#%% PLOTTING 

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts

## additional params
# hover_vdim = "species_short"
hover_vdim = "base_name"
legend_entries = "cell_line"

df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])

scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                           vdims=vdims, label=entry) for entry in list_entries]

overlay = hv.Overlay(scatter_umaps)
umap_parameters = f"metric: {reducer.metric} | " \
                f"n_neighbors: {reducer.n_neighbors} " \
                    f"distance: {reducer.min_dist}"
                

overlay.opts(
    opts.Scatter(
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=800, 
        height=800),
    opts.Overlay(
        title=f"UMAP \n {umap_parameters}")       
    )

filename=f"metric_{reducer.metric}_nneighbors_{reducer.n_neighbors}_mindist_{reducer.min_dist}"
path_output = Path(r"C:\Users\Nabiki\Desktop\redox_ratio")
hv.save(overlay, path_output / f"umap_{filename}.html" )


#%% PLOT UMAP EMBEDDINGS DATA 

# umap_embeddings = reducer.fit_transform(data)

# umap_embeddings = draw_umap(scaled_penguin_data).fit_transform(data)

# # print(embedding.shape)

# fig = plt.figure()
# if n_components == 1:
#     ax = fig.add_subplot(111)
#     ax.scatter(umap_points[:,0], range(len(umap_points)), labels=labels , c=[sns.color_palette()[x] for x in labels.map({"SCM":0, "nonSCM":1})])
# if n_components == 2: # plot in 2d
#     # ax = fig.add_subplot(111)
#     # ax.scatter(u[:,0], u[:,1], labels=list(labels), c=[sns.color_palette()[x] for x in labels.map({"SCM":0, "nonSCM":1})])
#     #print(u)
#     sns.scatterplot(umap_points[:,0], umap_points[:,1], hue=list(labels), alpha=0.5)
# if n_components == 3: #plot in 3d
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(umap_points[:,0], umap_points[:,1], 
#                umap_points[:,2], 
#                c=[sns.color_palette()[x] for x in labels.map({"SCM":0, "nonSCM":1})], s=100)
# plt.legend(loc="upper right", prop={'size': 16})
# plt.title(title, fontsize=18)
# plt.show()



#%% OPTIONAL GENERATE MULTIPLE UMAPS 
# n_neighbors: captures local vs global features. Small n local, large n global. Range 1-200
# min_dist: how closely points are allowed to pack together. Range 0-.99
# n_components: 1 2 or 3d plot
# metric: distance metric between data

# param_grid = {
#     "n_neighbors":  [10], # [2, 5, 10, 20, 50, 100, 200], 
#     "metric": [
#                 "euclidean", ##### aka l2,  straight line 
#                 "manhattan", ##### aka l1, block by block
#                 "chebyshev", ##### chessboard distance (8 degrees of movement but only one space at a time (King))
#                 # # # # # "minkowski", # generalization of both the Euclidean distance and the Manhattan distance
#                 "canberra", ##### weighted version of L1 (Manhattan) distance
#                 # # "braycurtis",
#                 # ### "haversine", # error when running
#                 # ### "mahalanobis", # error when running 
#                 # # "wminkowski",
#                 # ### "seuclidean", # error when running - division by zero
#                 "cosine",
#                 "correlation", ##### Pearson correlation
#                 # "hamming",
#                 # "jaccard",
#                 # "dice",
#                 # "russellrao",
#                 # "kulsinski",
#                 # "rogerstanimoto",
#                 # "sokalmichener",
#                 # "sokalsneath",
#                 # "yule"
#                 ]
#     }

# parameters = list(ParameterGrid(param_grid))

# returned_umap = None
# scaled_data_umap = StandardScaler().fit_transform(data_umap.copy()) # don't forget this step!!
# for n in parameters:
#     print(f"parameters: {n}")
#     returned_umap = draw_umap(scaled_data_umap, n_neighbors=n['n_neighbors'], \
#               metric=n['metric'], n_components=2, min_dist=0.3, \
#                   title=f'parameters = {n}', labels=labels, random_state=0)  
        
#%% UMAP documentation Penguin
# https://umap-learn.readthedocs.io/en/latest/basic_usage.html#penguin-data
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import umap

# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
# print(penguins.head())

# penguins = penguins.dropna()
# penguins.species_short.value_counts()

# # sns.pairplot(penguins, hue='species_short')
# # plt.show()


# penguin_data = penguins[
#     [
#         "culmen_length_mm",
#         "culmen_depth_mm",
#         "flipper_length_mm",
#         "body_mass_g",
#     ]
# ].values

# scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

# reducer = umap.UMAP(random_state=0)
# embedding = reducer.fit_transform(scaled_penguin_data)
# print(embedding.shape)

# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Penguin dataset', fontsize=24)
# plt.show()

