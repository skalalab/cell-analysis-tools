import numpy as np


def aggregate_sdt_roi(mask, sdt_cube, debug=False):

    """
    Given a labeled maks and an sdt cube (x, y, t), this function will 
    aggregate the decays in a given roi

    Parameters
    ----------
    mask : ndarray
        2d array of labeled rois.
    sdt_cube : ndarray
        3d array of photons x,y,t.
    debug : bool, optional
        Show debugging output. The default is False.

    Returns
    -------
    list_decays : list
        list of aggregate decays for each roi.
    list_roi_values : TYPE
        roi value from the given mask.

    """
    list_roi_values = np.unique(mask)
    list_roi_values = np.delete(
        list_roi_values, np.where(list_roi_values == 0)
    )  # exclude bg

    # store decays
    list_decays = []
    # ITERATE THROUGH ROI's
    for roi_idx, roi_value in enumerate(list_roi_values):
        if debug:
            print(f"processing roi {roi_idx+1}/{len(list_roi_values)}")
        pass

        # CREATE MASK OF ROI
        m_roi = (mask == roi_value).astype(int)

        #################
        # AGGREGATE DECAY
        masked_sdt_data = (
            sdt_cube * m_roi[..., np.newaxis]
        )  # mask out rest of pixels/decays
        decay = masked_sdt_data.sum(axis=(0, 1))  # sum over x and y, leave t alone
        list_decays.append(decay)

        return list_decays, list_roi_values
