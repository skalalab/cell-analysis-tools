from pathlib import Path
import re
from tqdm import tqdm
import pandas as pd
from cell_analysis_tools.io import load_image
from cell_analysis_tools.flim import regionprops_omi_stain as reg


#%% Inputs
'''This should be the only section where you should need to make changes'''

path_dataset = Path(r'C:\Users\jriendeau\Desktop\test')

mask_suffix = 'n_photons_cellmask.tif'

#What data in addition to NADH would you like to analyze?
FAD = False
Stain_intensity = False
Stain_lifetime = False
Intensity_weighted_means = False
other_props = ['area', 'eccentricity'] #can also be empty

#%% This finds your file paths for EACH image so you don't have to input everything

file_suffixes = {
    'mask': mask_suffix,
    'photons': '_photons.tiff',
    'a1[%]': '_a1\[%\].tiff',
    'a2[%]': '_a2\[%\].tiff',
    't1': '_t1.tiff',
    't2': '_t2.tiff',
    'chi': '_chi.tiff',
    'sdt': '.sdt',
    }

standard_dictionary = {
    # Mask file
    "mask" : "",
    
    # NADH files
    "nadh_photons" : "",
    "nadh_a1" : "",
    "nadh_a2" : "",
    "nadh_t1" : "",
    "nadh_t2" : "",
    
    # FAD files
    "fad_photons" : "",
    "fad_a1" : "",
    "fad_a2" : "",
    "fad_t1" : "",
    "fad_t2" : "",
    
    # Stain files
    "stain_photons" : "",
    "stain_a1" : "",
    "stain_a2" : "",
    "stain_t1" : "",
    "stain_t2" : "",
    }
    
# GET LIST OF ALL FILES FOR REGEX
list_all_files = list(path_dataset.rglob("*"))
list_str_all_files = [str(b) for b in list_all_files]
    
list_all_nadh_photons_images = list(filter(re.compile("n" + file_suffixes["photons"]).search, list_str_all_files ))##############################################################
    
dict_dir = {}
for path_str_im_photons in tqdm(list_all_nadh_photons_images, desc='Assembling file dictionary'):
    pass

    # generate dict name
    path_im_photons_nadh = Path(path_str_im_photons)
    handle_im = path_im_photons_nadh.stem.rsplit('n_', 1)[0]
    dict_dir[handle_im] = standard_dictionary.copy()
    # NADH 
    handle_nadh = handle_im + "n"
    # paths to NADH files
    dict_dir[handle_im]["nadh_photons"] = list(filter(re.compile(handle_nadh +  file_suffixes['photons']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_a1"] = list(filter(re.compile(handle_nadh +  file_suffixes['a1[%]']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_a2"] = list(filter(re.compile(handle_nadh +  file_suffixes['a2[%]']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_t1"] = list(filter(re.compile(handle_nadh +  file_suffixes['t1']).search, list_str_all_files))[0]
    dict_dir[handle_im]["nadh_t2"] = list(filter(re.compile(handle_nadh +  file_suffixes['t2']).search, list_str_all_files))[0]
    
    # MASKS
    try:
        dict_dir[handle_im]["mask"] = list(filter(re.compile(handle_im +  file_suffixes['mask']).search, list_str_all_files))[0]
    except IndexError:
        print(f"{handle_im} | mask missing")    
        del dict_dir[handle_im]
        continue
    
    # locate corresponding FAD photons image
    if FAD==True:
        try:
            path_str_im_photons_fad = list(filter(re.compile(handle_im + "f" + file_suffixes["photons"]).search, list_str_all_files))[0] ###################################
        except IndexError:
            print(f"{handle_im} | FAD files missing")
            del dict_dir[handle_im]
            continue
        path_im_photons_fad = Path(path_str_im_photons_fad)
        handle_fad = handle_im + "f"
        # paths to FAD files
        dict_dir[handle_im]["fad_photons"] = list(filter(re.compile(handle_fad +  file_suffixes['photons']).search, list_str_all_files))[0]
        dict_dir[handle_im]["fad_a1"] = list(filter(re.compile(handle_fad +  file_suffixes['a1[%]']).search, list_str_all_files))[0]
        dict_dir[handle_im]["fad_a2"] = list(filter(re.compile(handle_fad +  file_suffixes['a2[%]']).search, list_str_all_files))[0]
        dict_dir[handle_im]["fad_t1"] = list(filter(re.compile(handle_fad +  file_suffixes['t1']).search, list_str_all_files))[0]
        dict_dir[handle_im]["fad_t2"] = list(filter(re.compile(handle_fad +  file_suffixes['t2']).search, list_str_all_files))[0]
    
    # locate corresponding Stain photons image
    if Stain_intensity==True:
        try:
            path_str_im_photons_stain = list(filter(re.compile(handle_im + "r" + file_suffixes["photons"]).search, list_str_all_files))[0] ###################################
        except IndexError:
            print(f"{handle_im} | Stain files missing")
            del dict_dir[handle_im]
            continue
        path_im_photons_fad = Path(path_str_im_photons_stain)
        handle_fad = handle_im + "r"
        dict_dir[handle_im]["stain_photons"] = list(filter(re.compile(handle_im +  file_suffixes['stain']).search, list_str_all_files))[0]
        if Stain_lifetime==True:
            dict_dir[handle_im]["stain_a1"] = list(filter(re.compile(handle_fad +  file_suffixes['a1[%]']).search, list_str_all_files))[0]
            dict_dir[handle_im]["stain_a2"] = list(filter(re.compile(handle_fad +  file_suffixes['a2[%]']).search, list_str_all_files))[0]
            dict_dir[handle_im]["stain_t1"] = list(filter(re.compile(handle_fad +  file_suffixes['t1']).search, list_str_all_files))[0]
            dict_dir[handle_im]["stain_t2"] = list(filter(re.compile(handle_fad +  file_suffixes['t2']).search, list_str_all_files))[0]   

        
df_paths = pd.DataFrame(dict_dir).transpose()
df_paths.index.name = "base"



#%% load csv dicts with path sets 
     
# iterate through rows(image sets) in dataframe,
outputs = pd.DataFrame()
for base, row_data in tqdm(list(df_paths.iterrows()), desc='Analyzing images'): # iterate through sets in csv file
            pass

            # load mask image
            label_image = load_image(Path(str(row_data['mask'])))
            # load NADH images 
            im_nadh_intensity = load_image(Path(str(row_data.nadh_photons)))
            im_nadh_a1 = load_image(Path(str(row_data.nadh_a1)))
            im_nadh_a2 = load_image(Path(str(row_data.nadh_a2)))
            im_nadh_t1 = load_image(Path(str(row_data.nadh_t1)))
            im_nadh_t2 = load_image(Path(str(row_data.nadh_t2)))
            # load FAD images 
            if FAD == True:
                im_fad_intensity = load_image(Path(str(row_data.fad_photons)))
                im_fad_a1 = load_image(Path(str(row_data.fad_a1)))
                im_fad_a2 = load_image(Path(str(row_data.fad_a2)))
                im_fad_t1 = load_image(Path(str(row_data.fad_t1)))
                im_fad_t2 = load_image(Path(str(row_data.fad_t2)))
            # load RED image
            if Stain_intensity == True:
                im_stain_intensity = load_image(Path(str(row_data.stain_photons)))
            if Stain_lifetime == True:
                im_stain_a1 = load_image(Path(str(row_data.stain_a1)))
                im_stain_a2 = load_image(Path(str(row_data.stain_a2)))
                im_stain_t1 = load_image(Path(str(row_data.stain_t1)))
                im_stain_t2 = load_image(Path(str(row_data.stain_t2)))
            
            # compute ROI props
            if FAD==True and Stain_intensity==True and Stain_lifetime==True:
                omi_props = reg.regionprops_omi_run(
                    Intensity_weighted_means = Intensity_weighted_means,
                    FAD = FAD,
                    Stain_intensity = Stain_intensity,
                    Stain_lifetime = Stain_lifetime,
                
                    image_id = base,
                    label_image = label_image,
                    im_nadh_intensity = im_nadh_intensity,
                    im_nadh_a1 = im_nadh_a1, 
                    im_nadh_a2 = im_nadh_a2, 
                    im_nadh_t1 = im_nadh_t1, 
                    im_nadh_t2 = im_nadh_t2,
                    im_fad_intensity = im_fad_intensity,
                    im_fad_a1 = im_fad_a1,
                    im_fad_a2 = im_fad_a2,
                    im_fad_t1 = im_fad_t1,
                    im_fad_t2 = im_fad_t2,
                    im_stain_intensity = im_stain_intensity,
                    im_stain_a1 = im_stain_a1,
                    im_stain_a2 = im_stain_a2,
                    im_stain_t1 = im_stain_t1,
                    im_stain_t2 = im_stain_t2,
                    other_props=other_props
                    )
            if FAD==True and Stain_intensity==True and Stain_lifetime==False:
                omi_props = reg.regionprops_omi_run(
                    Intensity_weighted_means = Intensity_weighted_means,
                    FAD = FAD,
                    Stain_intensity = Stain_intensity,
                    Stain_lifetime = Stain_lifetime,
                
                    image_id = base,
                    label_image = label_image,
                    im_nadh_intensity = im_nadh_intensity,
                    im_nadh_a1 = im_nadh_a1, 
                    im_nadh_a2 = im_nadh_a2, 
                    im_nadh_t1 = im_nadh_t1, 
                    im_nadh_t2 = im_nadh_t2,
                    im_fad_intensity = im_fad_intensity,
                    im_fad_a1 = im_fad_a1,
                    im_fad_a2 = im_fad_a2,
                    im_fad_t1 = im_fad_t1,
                    im_fad_t2 = im_fad_t2,
                    im_stain_intensity = im_stain_intensity,
                    other_props=other_props
                    )
            if FAD==True and Stain_intensity==False and Stain_lifetime==False:
                omi_props = reg.regionprops_omi_run(
                    Intensity_weighted_means = Intensity_weighted_means,
                    FAD = FAD,
                    Stain_intensity = Stain_intensity,
                    Stain_lifetime = Stain_lifetime,
                 
                    image_id = base,
                    label_image = label_image,
                    im_nadh_intensity = im_nadh_intensity,
                    im_nadh_a1 = im_nadh_a1, 
                    im_nadh_a2 = im_nadh_a2, 
                    im_nadh_t1 = im_nadh_t1, 
                    im_nadh_t2 = im_nadh_t2,
                    im_fad_intensity = im_fad_intensity,
                    im_fad_a1 = im_fad_a1,
                    im_fad_a2 = im_fad_a2,
                    im_fad_t1 = im_fad_t1,
                    im_fad_t2 = im_fad_t2,
                    other_props=other_props
                    )
                
            if FAD==False and Stain_intensity==False and Stain_lifetime==False:
                omi_props = reg.regionprops_omi_run(
                    Intensity_weighted_means = Intensity_weighted_means,
                    FAD = FAD,
                    Stain_intensity = Stain_intensity,
                    Stain_lifetime = Stain_lifetime,
                 
                    image_id = base,
                    label_image = label_image,
                    im_nadh_intensity = im_nadh_intensity,
                    im_nadh_a1 = im_nadh_a1, 
                    im_nadh_a2 = im_nadh_a2, 
                    im_nadh_t1 = im_nadh_t1, 
                    im_nadh_t2 = im_nadh_t2,
                    other_props=other_props
                    )
                
            if FAD==False and Stain_intensity==False and Stain_lifetime==False:
                    omi_props = reg.regionprops_omi_run(
                        Intensity_weighted_means = Intensity_weighted_means,
                        FAD = FAD,
                        Stain_intensity = Stain_intensity,
                        Stain_lifetime = Stain_lifetime,
                     
                        image_id = base,
                        label_image = label_image,
                        im_nadh_intensity = im_nadh_intensity,
                        im_nadh_a1 = im_nadh_a1, 
                        im_nadh_a2 = im_nadh_a2, 
                        im_nadh_t1 = im_nadh_t1, 
                        im_nadh_t2 = im_nadh_t2,
                        other_props=other_props
                        )
            #create dataframe
            df = pd.DataFrame(omi_props).transpose()
            df.index.name = "base"
            
            # add other dictionary data to df
            df["base"] = base
            for item_key in row_data.keys():
                df[item_key] = row_data[item_key]

            # combine all image data into one csv
            outputs = pd.concat([outputs, df], axis=0)
            pass

#%% Final df manipulations before export

if 'stdev' in outputs.columns:
    stdev_columns = [col for col in df.columns if 'stdev' in col.lower()] 
    outputs.drop(columns=stdev_columns, inplace=True) #removes stdev columns
    pass
elif 'weighted' in outputs.columns:
    weighted_columns = [col for col in df.columns if 'weighted' in col.lower()]
    outputs.drop(columns=weighted_columns, inplace=True) #removes intensity weighted columns
    pass

# remove path files from outputs csv
outputs = outputs.iloc[:,:outputs.columns.get_loc("mask")]

# finally.. export data
outputs.to_csv(path_dataset/ f"{path_dataset.stem}_features.csv")