import scipy.ndimage as ndi


def fill_and_label_rois(curr_nuclei):
    """
    Fills and labels ROI outlines using unique ints for each region.

    Parameters
    ---------- 
    param curr_nucelei : ndarray
        Current image to process

    Returns
    -------
    output : ndarray 
        ROIs filled and labeled with unique int representations

    """

    return ndi.label(ndi.binary_fill_holes(curr_nuclei))[0]

    
if __name__ == "__main__":
    
    from pathlib import Path
    
    
    HERE = Path(__file__).absolute().resolve()
    print(HERE)
    path_mask = Path(r"../../examples/example_data/redox_ratio/HPDE_2DG_10n_mask_cells.tiff")
    
    
    