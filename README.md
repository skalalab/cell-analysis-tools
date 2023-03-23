### CELL ANALYSIS TOOLS

A library for loading, processing and summarizing single cell imaging data. This library was originally created for the Skala lab at 
the Morgridge Institute for Research/Univeristy of Wisconsin-Madison to process and analyze 2-photon data from time correlated single photon counting. 
Many functions and implementations can be reused for different image types. It also has examples of how to use some of these functions along with 
example data. 


More documentation 

[Documentation on ReadTheDocs](https://cell-analysis-tools.readthedocs.io/en/latest/)

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

The library can be install through pip 

`pip install cell-analysis-tools`

or the latest version can be installed by cloning the repository, changeing directory to the root of the cell_analysis_tools folder then execute:

`$ pip install -e .`

you should then be able to import it into your script

`import cell_analysis_tools`

--- 

### Cite cell-analysis-tools
``` 
@inproceedings{10.1117/12.2647280,
author = {Emmanuel Contreras Guzman and Peter R. Rehani and Melissa C. Skala},
title = {{Cell analysis tools: an open-source library for single-cell analysis of multi-dimensional microscopy images}},
volume = {12383},
booktitle = {Imaging, Manipulation, and Analysis of Biomolecules, Cells, and Tissues XXI},
editor = {Attila Tarnok and Jessica P. Houston},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {123830G},
keywords = {Single-cell, image analysis, data analysis, open-source software, python},
year = {2023},
doi = {10.1117/12.2647280},
URL = {https://doi.org/10.1117/12.2647280}
}
```


Emmanuel Contreras Guzman, Peter R. Rehani, Melissa C. Skala, "Cell analysis tools: an open-source library for single-cell analysis of multi-dimensional microscopy images," Proc. SPIE 12383, Imaging, Manipulation, and Analysis of Biomolecules, Cells, and Tissues XXI, 123830G (15 March 2023); https://doi.org/10.1117/12.2647280
