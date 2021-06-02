# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:17:53 2021

@author: Nabiki
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from pprint import pprint
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 500


# LOAD DATA
working_directory = Path("C:/Users/Nabiki/Desktop/development/km_analysis_and_graphs")
path_data = working_directory / "Final_Full_MeanintensitiesEPCiPSCMid_USEDFORUMAP_Patterned.csv"
df_data = pd.read_csv(path_data)

#%% SELECT AND CLEAN DATA
# And I was hoping to make a dendogram based on these values across three different 
# classes: EPC, iPSC, Mid indicated by "Class" (Column X). 
# heat map can also be created for these based on "Class" and "Donor Number".

# SELECT CLASSES
list_class_subsets = list(df_data.Class.unique())
list_class_subsets.remove("H9")
df_classes = df_data[df_data['Class'].isin(list_class_subsets)]

# SELECT JUST METABOLIC COLUMNS
list_metabolic_info_headers = df_data.iloc[:,4:14].keys()
print("\ncolumns selected:")
pprint(list_metabolic_info_headers)
df_metabolic_features = df_classes[list_metabolic_info_headers]

# RENAME COLUMN HEADERS
print("\nrenaming metabolic features columns")
metabolic_headers = [title.rsplit("_", 1)[1] for title in df_metabolic_features.keys()]
print(metabolic_headers)
df_metabolic_features = pd.DataFrame(df_metabolic_features.values, columns=metabolic_headers)

# confirm renaming headers didn't shift data
assert df_metabolic_features.values.all() == df_metabolic_features.values.all()
#%% PLOT ALL ROWS IN DATAFRAME 
# Color classes
lut = dict(zip(list_class_subsets, "rbg"))
df_classes_col = df_classes["Class"]
row_colors = df_classes_col.map(lut)

# plot finally #z_score=0
g = sns.clustermap(df_metabolic_features, standard_scale = 1, row_colors=row_colors, cmap="viridis", figsize=(10,10))

# add class labels 
# https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
handles = [Patch(facecolor=lut[name]) for name in lut]
plt.legend(handles, lut, title='Class',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

# other plot parameters
plt.title(f"mean intensities")

plt.savefig(working_directory / "all_data.png", 
            dpi=400, 
            format="png",
            bbox_inches='tight')

#%% PLOT BY DONOR NUMBER AND AVERAGE rows

# DF
df_donor_number_class = pd.DataFrame(columns=metabolic_headers)

# get unique donor numbers 
list_donor_numbers = list(df_data.DonorNumber.unique())
list_donor_numbers.remove("H9")
# iterate through each donor number, calculating mean value for all rows
for donor_number in list_donor_numbers:
    pass
    df_class_donor_number = df_data[df_data["DonorNumber"] == donor_number]
    metabolic_data = df_class_donor_number[list_metabolic_info_headers].mean(axis=0)
    
    # store values in temporary df
    df_donor_number_class.loc[donor_number] = metabolic_data.values
    
# plot finally #z_score=0
g = sns.clustermap(df_donor_number_class, standard_scale = 1, cmap="viridis")

# other plot parameters
plt.title(f"mean intensities")
plt.savefig(working_directory / "heatmap_by_donor_number.png", 
            dpi=400, 
            format="png",
            bbox_inches='tight')
plt.show()
#%% PLOT BY CLASS AND AVERAGE rows

# DF
df_class_features = pd.DataFrame(columns=metabolic_headers)

# get unique classes
list_classes= list(df_data.Class.unique())
list_classes.remove("H9")
# iterate through each donor number, calculating mean value for all rows
for class_label in list_classes:
    pass
    df_class = df_data[df_data["Class"] == class_label]
    metabolic_data = df_class[list_metabolic_info_headers].mean(axis=0)
    
    # store values in temporary df
    df_class_features.loc[class_label] = metabolic_data.values
    
# plot finally #z_score=0
g = sns.clustermap(df_class_features, standard_scale = 1, cmap="viridis", figsize=(10,10))

# other plot parameters
plt.title(f"mean intensities")
plt.savefig(working_directory / "heatmap_by_class.png", 
            dpi=400, 
            format="png", 
            bbox_inches='tight')
plt.show()



#%%
# ##########
# lut = dict(zip(species.unique(), "rbg"))
# row_colors = species.map(lut)
# g = sns.clustermap(iris, row_colors=row_colors)

# ################ 
# import seaborn as sns; sns.set_theme(color_codes=True)
# iris = sns.load_dataset("iris")
# species = iris.pop("species")
# g = sns.clustermap(iris)



