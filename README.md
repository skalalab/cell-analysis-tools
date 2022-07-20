### flim_tools

A library for loading, processing and summarizing flim data.

---
#### Dependencies

* numpy
* tifffile 
* from pathlib import Path
* read_roi
* os
* matplotlib
* re
* skimage
* pandas as pd

---
#### Installation

To install this library change directory to the root of the flim_tools folder then execute:

`$ pip install -e .`

you should then be able to import it into your script


`import flim_tools`

--- 
### Summarizing your data

The flim_tools library contains a script `main.py` that can be used to summarize your data.

The script was created with 30 images from the T cell dataset used in the paper [Classification of T-cell activation via autofluorescence lifetime imaging](https://www.nature.com/articles/s41551-020-0592-z) if is strongly recommended you run the code as is with this dataset to makes sure you know how to modify it to your needs.

The algorithm takes in a dictionary of paths pointing to the various tiffs that will be used as data or masks.


    set_dict = {
            "sdt": path_sdt,
            "mask_whole_cell" : path_m_cell, # store paths to cell/cyto masks
            "mask_cyto" : path_m_cyto,
            "photons": path_photons,
            "a1" : path_a1,
            "a2" :path_a2,
            "t1" : path_t1,
            "t2" : path_t2,
            "chisq" : path_chisq,
            }

