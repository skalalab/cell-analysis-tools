import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 300
from cellpose import  io, models
from pathlib import Path
import tifffile
from cell_analysis_tools.io import load_image


if __name__ == "__main__":
    
    #%% Load Images
    HERE = Path(__file__).resolve().absolute().parent
    
    save_results = True
    
    filename_suffix = "cellpose" # 
    
    # Load from images directory
    files = list((HERE / "images").glob("*.tif*"))
    
    #%%
    # RUN CELLPOSE
    
    
    ##### PARAMETERS TO ADJUST
    
    path_models = r'Z:\0-segmentation\cellpose\COBA\Models'
    
    dict_Cellpose_params = {
        "gpu" : True,

        ### comment in one of these models only!
        # 'model_type' : 'cyto2',
        'pretrained_model' : path_models + '\\' +  'Adherent Cell\AdherentCells.zip',
        # 'pretrained_model' : path_models + '\\' +  "Adherent Cell\AdherentNuclei.zip",
        # 'pretrained_model' : path_models + '\\' +  "Organoid\OrganoidCells.zip",
        # 'pretrained_model' : path_models + '\\' +  "Organoid\OrganoidNuclei.zip",
        ####
        
        'net_avg' : True,
        }
    
    dict_eval_params = {
        'diameter' : 30,
        'cellprob_threshold' : -0.24, 
        'flow_threshold' : 13.24 # model match threshold on GUI
        }
    
    ###############################
    
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    # model = models.Cellpose(gpu=False, model_type='cyto')
    model = models.CellposeModel(**dict_Cellpose_params)
    
    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    channels = [[0,0]]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
    
    # or if you have different types of channels in each image
    # channels = [[2,3], [0,0], [0,0]]
    
    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended) 
    # diameter can be a list or a single number for all images
    
    # you can run all in a list e.g.
    # >>> imgs = [io.imread(filename) in for filename in files]
    # >>> masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
    # >>> io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)
    # >>> io.save_to_png(imgs, masks, flows, files)
    
    # or in a loop
    for chan, filename in zip(channels*len(files), files):
        pass
        
        img = load_image(filename) # tifffile.imread(filename)
        
        # masks, flows, styles, diams = model.eval(img, diameter=None, channels=chan)
        masks, flows, styles = model.eval(img, channels=chan, **dict_eval_params)

        # save results so you can load in gui

        # if save_results:
        #     pass
        #     print(f"*.npy not yet supported with models.CellposeModel object because it does not return a diams list")
        #     # https://github.com/MouseLand/cellpose/issues/492
        #     # io.masks_flows_to_seg(img, masks, flows, filename, chan)
    

        # save results as png
        # io.save_masks(img, masks, flows, filename, tif=True)
        if save_results:
            filename_mask =  f"{filename.stem.strip()}_{filename_suffix}.tiff"
            tifffile.imwrite(HERE / "masks_cellpose" / filename_mask, masks)
        
    #%%
    
        # DISPLAY RESULTS
        from cellpose import plot
        
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
        plt.tight_layout()
        plt.show()
        
        #%%
        
        ## code to prevent overwritting images, used in custom scritps if only one mask is being used
        # if save_results and not (filename.parent /  filename_mask).exists():
        #     tifffile.imwrite(filename.parent /  filename_mask, masks)
        #     print(filename.parent /  filename_mask)
        # else:
        #     print(f"Mask already exists, not saving file: {(filename.parent /  filename_mask)}")
        
        