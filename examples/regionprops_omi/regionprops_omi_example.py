import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt
import pandas as pd
from tqdm import tqdm

import cell_analysis_tools as cat
from cell_analysis_tools.flim import regionprops_omi
from cell_analysis_tools.io import load_image

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from datetime import date

import numpy as np
import tifffile

#%% Placeholder dictionary of header entries

standard_dictionary = {
    # experiment details
    "date": "",  # YYYYMMDD
    # masks
    "mask_nuclei": "",  # path_to_file
    "mask_cytoplasm": "",  # path_to_file
    "mask_cell": "",  # path_to_file
    # metadata
    "analyzed_by": "",  # initials
    "reanalyzed_by": "",  # initials
    "nadh_image_number": "",
    "fad_image_number": "",
    "experiment": "",
    "media": "",
    "cell_type": "",
    "cell_line": "",
    "cancer": "",
    "dish": "",
    "experiment": "",  # ["confluency", "glucose" ,"ph","seahorse","duroquinone","tmre"]
    "treatment": "",  # 2DG, Rotenone, IAA, Antimycin
    # spc exports
    # fad
    "fad_photons": "",  # path_to_file,
    "fad_a1": "",  # path_to_file,
    "fad_a2": "",  # path_to_file,
    "fad_t1": "",  # path_to_file,
    "fad_t2": "",  # path_to_file,
    # nadh
    "nadh_photons": "",  # path_to_file,
    "nadh_a1": "",  # path_to_file,
    "nadh_a2": "",  # path_to_file,
    "nadh_t1": "",  # path_to_file,
    "nadh_t2": "",  # path_to_file,
    # other parameters
    "dye": "",
    "resolution": "",
    "objective": "",
    "filter_cube": "",
}

#%% Suffixes to use for matching files

file_suffixes = {
    "im_photons": "_photons.asc",
    "mask_cell": "_mask_cells.tiff",
    "mask_cytoplasm": "_mask_cyto.tiff",
    "mask_nuclei": "_mask_nuclei.tiff",
    "a1[%]": "_a1\[%\].asc",
    "a2[%]": "_a2\[%\].asc",
    "t1": "_t1.asc",
    "t2": "_t2.asc",
    "chi": "_chi.asc",
    "sdt": ".sdt",
}

#%% function for visualizing dictionary


def visualize_dictionary(
    dict_entry,
    extra_entries=[],
    remove_entries=[],
    channel=None,
    path_output: str = None,
):
    pass
    # show grid of all the images it's matching up
    # photons, masks, spc outputs
    entries = [
        "nadh_photons",
        "nadh_a1",
        "nadh_a2",
        "nadh_t1",
        "nadh_t2",
        #
        "fad_photons",
        "fad_a1",
        "fad_a2",
        "fad_t1",
        "fad_t2",
        "mask_cell",
        "mask_cytoplasm",
        "mask_nuclei",
    ]
    if len(extra_entries) != 0:
        entries += extra_entries
    for entry in remove_entries:
        entries.remove(entry)

    rows = 3
    cols = 5
    fig, ax = plt.subplots(rows, cols, figsize=(20, 12))
    filename_image = Path(dict_entry["nadh_photons"]).stem
    dataset_dir = str(Path(dict_entry["nadh_photons"]).parent).split("\\", 5)[5]
    fig.suptitle(f"{filename_image} \n {dataset_dir}")

    for pos, key in enumerate(entries):
        pass
        dict_entry[key]

        col = pos % cols
        row = pos // cols

        path_image = Path(dict_entry[key])
        # load correct format
        if path_image.suffix in [".tiff", ".tif"]:
            image = tifffile.imread(path_image)
        else:
            image = load_image(path_image)

        # load proper channel
        if len(image.shape) > 2:
            if image.shape[2] == 3:  # rgb image of cyto mask
                image = np.sum(image, axis=2)
            if image.shape[0] == 2:  # multi channel image, pick channel number
                image = image[channel, ...]
        ax[row, col].imshow(image)
        ax[row, col].set_title(
            f"{path_image.stem} \n min: {np.min(image):.3f}  max: {np.max(image):.3f}"
        )
        ax[row, col].set_axis_off()

    if str(path_output) != "None":
        plt.savefig(path_output / filename_image)
    plt.show()


#%% load and aggregate data into a set


def load_data_create_dict(path_dataset, path_output):
    ##%% Load dataset paths and find all nadh images

    # GET LIST OF ALL FILES FOR REGEX
    list_all_files = list(path_dataset.rglob("*"))
    list_str_all_files = [str(b) for b in list_all_files]

    # GET LIST OF ALL PHOTONS IMAGES
    list_all_nadh_photons_images = list(
        filter(re.compile(r".*n_photons.asc").search, list_str_all_files)
    )

    ##%% POPULATE DICTIONARY

    list_incomplete_sets = []

    dict_dir = {}
    for path_str_im_photons in tqdm(list_all_nadh_photons_images):
        pass

        # generate dict name
        path_im_photons_nadh = Path(path_str_im_photons)
        handle_im = path_im_photons_nadh.stem.rsplit("_", 1)[0][:-1]
        dict_dir[handle_im] = standard_dictionary.copy()

        # NADH
        handle_nadh = path_im_photons_nadh.stem.rsplit("_", 1)[0]

        # nadh image number and treatment
        # image_number_nadh = int(handle_nadh.rsplit("-",1)[1])
        # _, treatment, dish, _, _, = handle_nadh.split("_")

        # dict_dir[handle_im]["nadh_image_number"] = image_number_nadh
        # dict_dir[handle_im]["dish"] = int(dish.replace("d",""))

        # standardize treatment name
        # if treatment == "eto" : treatment = "etomoxir"
        # elif treatment == "IAA" : treatment = "iodoacetic acid"
        # elif treatment == "SA" : treatment = "sodium arsenite"
        # elif treatment == "SF" : treatment = "sodium fluoroacetate"

        # dict_dir[handle_im]["treatment"] = treatment

        # paths to files
        dict_dir[handle_im]["nadh_photons"] = list(
            filter(
                re.compile(handle_nadh + file_suffixes["im_photons"]).search,
                list_str_all_files,
            )
        )[0]
        dict_dir[handle_im]["nadh_a1"] = list(
            filter(
                re.compile(handle_nadh + file_suffixes["a1[%]"]).search,
                list_str_all_files,
            )
        )[0]
        dict_dir[handle_im]["nadh_a2"] = list(
            filter(
                re.compile(handle_nadh + file_suffixes["a2[%]"]).search,
                list_str_all_files,
            )
        )[0]
        dict_dir[handle_im]["nadh_t1"] = list(
            filter(
                re.compile(handle_nadh + file_suffixes["t1"]).search, list_str_all_files
            )
        )[0]
        dict_dir[handle_im]["nadh_t2"] = list(
            filter(
                re.compile(handle_nadh + file_suffixes["t2"]).search, list_str_all_files
            )
        )[0]

        # MASKS
        try:
            dict_dir[handle_im]["mask_cell"] = list(
                filter(
                    re.compile(handle_nadh + file_suffixes["mask_cell"]).search,
                    list_str_all_files,
                )
            )[0]
            dict_dir[handle_im]["mask_cytoplasm"] = list(
                filter(
                    re.compile(handle_nadh + file_suffixes["mask_cytoplasm"]).search,
                    list_str_all_files,
                )
            )[0]
            dict_dir[handle_im]["mask_nuclei"] = list(
                filter(
                    re.compile(handle_nadh + file_suffixes["mask_nuclei"]).search,
                    list_str_all_files,
                )
            )[0]
        except IndexError:
            print(f"{handle_im} | one or more masks missing skipping, set")
            list_incomplete_sets.append(f"{handle_im} | missing: mask files")
            del dict_dir[handle_im]
            continue

        # FAD
        # locate corresponding photons image
        try:
            path_str_im_photons_fad = list(
                filter(
                    re.compile(handle_im + "f" + file_suffixes["im_photons"]).search,
                    list_str_all_files,
                )
            )[0]
        except IndexError:
            print(f"{handle_im} | one or more fad files missing, skipping set")
            list_incomplete_sets.append(f"{handle_im} | missing: fad files")
            del dict_dir[handle_im]
            continue

        path_im_photons_fad = Path(path_str_im_photons_fad)
        handle_fad = path_im_photons_fad.stem.rsplit("_", 1)[0]

        # image number
        # image_number = handle_fad.rsplit("-",1)[1]
        # dict_dir[handle_im]["fad_image_number"] = int(image_number)

        # paths to images
        dict_dir[handle_im]["fad_photons"] = list(
            filter(
                re.compile(handle_fad + file_suffixes["im_photons"]).search,
                list_str_all_files,
            )
        )[0]
        dict_dir[handle_im]["fad_a1"] = list(
            filter(
                re.compile(handle_fad + file_suffixes["a1[%]"]).search,
                list_str_all_files,
            )
        )[0]
        dict_dir[handle_im]["fad_a2"] = list(
            filter(
                re.compile(handle_fad + file_suffixes["a2[%]"]).search,
                list_str_all_files,
            )
        )[0]
        dict_dir[handle_im]["fad_t1"] = list(
            filter(
                re.compile(handle_fad + file_suffixes["t1"]).search, list_str_all_files
            )
        )[0]
        dict_dir[handle_im]["fad_t2"] = list(
            filter(
                re.compile(handle_fad + file_suffixes["t2"]).search, list_str_all_files
            )
        )[0]

        # OTHER HEADER CSV INFO
        # dict_dir[handle_im]["date"] = metadata_experiment["date"][0]
        # dict_dir[handle_im]["cell_line"] = metadata_experiment["cell_line"][0]
        # dict_dir[handle_im]["cell_type"] = metadata_experiment["cell_type"][0]
        # dict_dir[handle_im]["tissue"] = metadata_experiment["tissue"][0]
        # dict_dir[handle_im]["cancer"] = metadata_experiment["cancer"][0]
        # dict_dir[handle_im]["media"] = metadata_experiment["media"][0]
        # dict_dir[handle_im]["dye"] = metadata_experiment["dye"][0]
        # dict_dir[handle_im]["resolution"] = metadata_experiment["resolution"][0]
        # dict_dir[handle_im]["objective"] = metadata_experiment["objective"][0]
        # dict_dir[handle_im]["filter_cube"] = metadata_experiment["filter_cube"][0]
        # dict_dir[handle_im]["experiment"] = metadata_experiment["experiment"][0]

        # DIRECTORY FILENAME INFO
        # dict_dir[handle_im]["analyzed_by"] = analyzed_by

        # OTHER VALUES
        dict_dir[handle_im]["resolution"] = load_image(
            Path(dict_dir[handle_im]["nadh_photons"])
        ).shape

    # export
    df = pd.DataFrame(dict_dir).transpose()
    df.index.name = "base_name"
    df.to_csv(path_output / "regionprops_omi.csv")

    if len(list_incomplete_sets) != 0:
        pass
        df_incomplete = pd.DataFrame(list_incomplete_sets, columns=["incomplete_sets"])
        df_incomplete.to_csv(path_output / "regionprops_omi_incomplete.csv")

    return df, list_incomplete_sets

    #%%


if __name__ == "__main__":
    pass
    #%% Load dictionary and compute the regionprops omi parameters
    
    # Comment in if running example
    HERE = Path(__file__).absolute().resolve().parent
    path_dataset = HERE.parent / "example_data/redox_ratio"
    path_output = HERE.parent / "regionprops_omi/outputs/"
    
    # comment in if running own code
    # path_dataset = Path(r"/home/nabiki/Desktop/")
    # path_output = path_dataset / "outputs"
    # path_output.mkdir(exist_ok=True)
    # path_dictionaries = path_output / "dictionaries"
    # path_dictionaries.mkdir(exist_ok=True)
    # path_features = path_output / "features" 
    # path_features.mkdir(exist_ok=True)
    # path_summary = path_output / "summary" 
    # path_summary.mkdir(exist_ok=True)


    df, incomplete = load_data_create_dict(
        path_dataset=path_dataset, path_output=(path_output / "dictionaries")
    )

    for key, item in list(df.iterrows())[:1]:
        pass
        visualize_dictionary(item)

    #%% Load set/dictionaries and compute omi parameters

    # load csv dicts with path sets
    list_path_csv_image_sets = list((path_output / "dictionaries").glob("*_omi.csv"))

    analysis_type = "whole_cell"

    for dict_path in list_path_csv_image_sets[:1]:  # iterate through csv files
        pass
        print(f"processing: {dict_path.stem}")
        df_image_set = pd.read_csv(dict_path)
        df_image_set = df_image_set.set_index("base_name", drop=True)

        # keep running list of dataframes
        # df_all_dicts = df_image_set if df_all_dicts is None else df_all_dicts.append(df_image_set)

        # iterate through rows(image sets) in dataframe,
        for base_name, row_data in tqdm(
            list(df_image_set.iterrows())
        ):  # iterate through sets in csv file
            pass

            # load mask based on analysis type
            if analysis_type == "whole_cell":
                mask = load_image(Path(row_data.mask_cell))  # whole cell
            elif analysis_type == "cytoplasm":
                mask = load_image(Path(row_data.mask_cytoplasm))  # cytoplasm
            elif analysis_type == "nuclei":
                mask = load_image(Path(row_data.mask_nuclei))  # nuclei

            # load images
            im_nadh_intensity = load_image(Path(row_data.nadh_photons))
            im_nadh_a1 = load_image(Path(row_data.nadh_a1))
            im_nadh_a2 = load_image(Path(row_data.nadh_a2))
            im_nadh_t1 = load_image(Path(row_data.nadh_t1))
            im_nadh_t2 = load_image(Path(row_data.nadh_t2))
            im_fad_intensity = load_image(Path(row_data.fad_photons))
            im_fad_a1 = load_image(Path(row_data.fad_a1))
            im_fad_a2 = load_image(Path(row_data.fad_a2))
            im_fad_t1 = load_image(Path(row_data.fad_t1))
            im_fad_t2 = load_image(Path(row_data.fad_t2))

            # compute ROI props
            omi_props = regionprops_omi(
                image_id=base_name,
                label_image=mask,  ### this selects what mask to summarize
                im_nadh_intensity=im_nadh_intensity,
                im_nadh_a1=im_nadh_a1,
                im_nadh_a2=im_nadh_a2,
                im_nadh_t1=im_nadh_t1,
                im_nadh_t2=im_nadh_t2,
                im_fad_intensity=im_fad_intensity,
                im_fad_a1=im_fad_a1,
                im_fad_a2=im_fad_a2,
                im_fad_t1=im_fad_t1,
                im_fad_t2=im_fad_t2,
            )

            ## create dataframe
            df = pd.DataFrame(omi_props).transpose()
            df.index.name = "base_name"

            ## add other dictionary data to df
            df["set_base_name"] = base_name
            for item_key in row_data.keys():
                df[item_key] = row_data[item_key]

            # finally export
            df.to_csv(
                path_output / "features" / f"features_{base_name}_{analysis_type}.csv"
            )

    #%% Aggregates all features into a single dataframe

    path_output_props = path_output / "summary"
    list_path_features_csv = list((path_output / "features").glob("*.csv"))

    df_all_props = None
    for path_feat_csv in tqdm(list_path_features_csv):
        pass
        # initialize first df entry else append to running df
        df_omi = pd.read_csv(path_feat_csv)
        df_all_props = (
            df_omi
            if df_all_props is None
            else pd.concat([df_all_props, df_omi], ignore_index=True)
        )

    # label index of complete dictionary
    df_all_props = df_all_props.set_index("base_name", drop=True)

    d = date.today().strftime("%Y_%m_%d")
    df_all_props.to_csv(path_output_props / f"{d}_all_props.csv")
