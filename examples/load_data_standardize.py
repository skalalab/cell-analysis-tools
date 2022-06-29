# plotting libraries
import matplotlib as mpl
import matplotlib.pylab as plt

mpl.rcParams["figure.dpi"] = 300
# import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.model_selection import ParameterGrid

# sklearn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#%%
if __name__ == "__main__":

    # example data from Gina
    path_dataset = Path(
        r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\2022_04_11_all_props.csv"
    )
    df_data = pd.read_csv(path_dataset)

    #

    # select parameters
    list_omi_parameters = [
        "nadh_intensity_mean",
        "nadh_a1_mean",
        "nadh_a2_mean",
        "nadh_t1_mean",
        "nadh_t2_mean",
        "nadh_tau_mean_mean",
        "fad_intensity_mean",
        "fad_a1_mean",
        "fad_a2_mean",
        "fad_t1_mean",
        "fad_t2_mean",
        "fad_tau_mean_mean",
        "redox_ratio_mean",
    ]
