from pathlib import Path
import pandas as pd
import shutil
import tifffile
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import numpy as np    
import napari
import sys

if __name__ == "__main__":
    
    HERE = Path(__file__).resolve().absolute().parent
    
    # paths to folders
    path_images = HERE / "images"
    path_masks_original = HERE / "masks_original" 
    path_masks_edited = HERE / "masks_edited"
    path_masks_cellpose = HERE / "masks_cellpose"

    # list of files in each folder
    list_paths_images = list(map(str,list(path_images.glob("*.tif*"))))
    list_path_masks_original = list(map(str,list(path_masks_original.glob("*.tif*"))))
    list_path_masks_edited = list(map(str,list(path_masks_edited.glob("*.tif*"))))
    
    # Prompt to figure out which images to iterate through
    print(f"Number of images in dataset: {len(list_paths_images)}")
    idx_start = input("start image: ")
    idx_stop = input("end image: ")
 
    idx_start = int(idx_start)
    idx_stop = int(idx_stop)
    ### end prompt
    
    for path_image in list_paths_images[idx_start-1 : idx_stop]:
        pass
    
    
        # Load intensity image
        intensity = tifffile.imread(path_image) 
        
        # Load Mask
        filename = Path(path_image).stem
        mask=None
        print(f"loading: {filename}")
        
        # load edited mask if available
        list_mask_edited = [p for p in list_path_masks_edited if filename.strip() in p]
        if len(list_mask_edited) == 0: 
            print("no edited mask found")
        else:
            print("loading edited mask")
            path_mask = list_mask_edited[0]
            mask = tifffile.imread(path_mask)
        
        # load original mask if available    
        if mask is None:
            list_mask_original = [p for p in list_path_masks_original if filename.strip() in p]
            if len(list_mask_original) == 0: 
                print("no original mask found")
            else:
                print("loading original mask")
                path_mask = list_mask_original[0]
                mask = tifffile.imread(path_mask)
        
        # No mask found, add empty mask layer
        bool_no_mask = False
        if mask is None:
            print("No mask found, adding empty mask")
            mask = np.zeros_like(intensity, dtype=np.uint16)
            bool_no_mask = True
    
        viewer = napari.Viewer(show=False)

        @viewer.bind_key("n") #, overwrite=True
        def next_image(viewer):
            print("loading next image")
            viewer.close()
            # sys.exit(0)
            # exit()
        
        # POPULATE NAPARI VIEWER
        # load mask from generated if available
        layer_intensity = viewer.add_image(intensity, name=Path(path_image).name)
        generate_mask_filename = f"{Path(path_image).stem.strip()}_mask.tiff"
        mask_name = generate_mask_filename if bool_no_mask else Path(path_mask).name
        layer_mask = viewer.add_labels(mask, name=mask_name)
        
        layer_mask.opacity = 0.4
        viewer.show(block=True)
        
        # find cellpose mask
        for layer in viewer.layers:
            if "_cp_masks_000" in layer.name:
                print(f"found cellpose mask: {layer.name}")
                tifffile.imwrite(path_masks_cellpose / f"{Path(path_mask).name}_cellpose.tiff",  layer.data)
        
        
        # SAVE IMAGE 
        path_im_output = path_masks_edited / mask_name
        print(path_im_output)
        tifffile.imwrite(path_im_output, layer_mask.data)
        

        
        
        
        
        