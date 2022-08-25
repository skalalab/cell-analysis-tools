# https://python.plainenglish.io/solve-graph-coloring-problem-with-greedy-algorithm-and-python-6661ab4154bd

from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt

mpl.rcParams["figure.dpi"] = 300
from collections import OrderedDict
from pprint import pprint

import numpy as np
import tifffile
from scipy.ndimage import label
from skimage.morphology import dilation, disk


def four_color_theorem(mask: np.ndarray) -> np.ndarray:
    """
    Converts a n-labeled image to an up to 4 color image.

    Parameters
    ----------
    mask : np.ndarray
        labeled mask with unique rois.

    Returns
    -------
    four_color_mask : np.ndarray
        returns a labeled image that uses up to 4 colors.
    solution_nodes : dict
        dictionary of {value : color} for each unique roi value

    """

    ### Generate Agencency Matrix
    id_nodes = list(np.unique(mask))
    id_nodes.remove(0)  # remove bg
    dict_nodes_key_roi_value = {}
    dict_nodes_key_idx = {}

    for idx, roi_value in enumerate(id_nodes):
        dict_nodes_key_roi_value[roi_value] = {}
        dict_nodes_key_roi_value[roi_value]["adj_mat_idx"] = idx
        dict_nodes_key_roi_value[roi_value]["colors"] = ["b", "c", "m", "y", 'r', 'g']
        dict_nodes_key_idx[idx] = roi_value

    adjacency_matrix = np.zeros((len(id_nodes), len(id_nodes)))

    for roi_value in dict_nodes_key_roi_value:
        pass
        roi_idx = dict_nodes_key_roi_value[roi_value]["adj_mat_idx"]
        mask_roi = np.array(mask == roi_value, dtype=np.uint16)
        # plt.imshow(mask_roi)
        # plt.show()

        # determine neighbors / degree
        # dilate and overlap with labeled mask to determine neighbors
        mask_roi_dialated = dilation(mask_roi, disk(3))
        mask_overlap = mask * mask_roi_dialated
        nodes_neighbors = list(np.unique(mask_overlap))
        nodes_neighbors.remove(roi_value)  # remove current roi
        nodes_neighbors.remove(0)  # remove bg

        #  populate adgacency matrix
        if nodes_neighbors:
            for neighbor_value in nodes_neighbors:
                neighbor_index = dict_nodes_key_roi_value[neighbor_value]["adj_mat_idx"]
                adjacency_matrix[roi_idx][neighbor_index] = 1
                adjacency_matrix[neighbor_index][roi_idx] = 1

        # dict increase degree
        dict_nodes_key_roi_value[roi_value]["degree"] = len(nodes_neighbors)

    ### Sort dictionary by largest to smallest degree
    # this will color regions with many neighbors first
    sorted_nodes = sorted(
        dict_nodes_key_roi_value.items(),
        key=lambda item: item[1]["degree"],
        reverse=True,
    )
    nodes = OrderedDict(sorted_nodes)

    # Main Algorithm
    solution_nodes = {}  # roi_value : color
    for roi_value in nodes.keys():  # iterate through roi_values
        pass
        # get list of colors in current roi, choose current color for roi
        list_roi_colors = nodes[roi_value]["colors"]
        # if roi_value == 302:
        #     print('stop')
        curr_color = list_roi_colors[0]
        solution_nodes[roi_value] = curr_color  # set solutions

        # get adjacency matrix to compare color against neighbors
        adjacent_nodes = adjacency_matrix[
            nodes[roi_value]["adj_mat_idx"]
        ]  # get adj_matrix row at this index

        # iterate through each row/neighbor in adj matrix
        for pos in range(len(adjacent_nodes)):
            pass
            # if it's a neighbor and it has the color in it's list, remove it for that neighbors list
            adj_node_idx = dict_nodes_key_idx[pos]
            if adjacent_nodes[pos] == 1 and (
                curr_color in nodes[adj_node_idx]["colors"]
            ):
                nodes[adj_node_idx]["colors"].remove(curr_color)

    # create array to store solution
    four_color_mask = np.zeros_like(mask)

    # convert from color to an intensity value for each color
    for roi_value in solution_nodes:
        pass
        # convert color to intensity
        if solution_nodes[roi_value] == "b":
            value = 50
        elif solution_nodes[roi_value] == "c":
            value = 100
        elif solution_nodes[roi_value] == "m":
            value = 150
        elif solution_nodes[roi_value] == "r":
            value = 200
        elif solution_nodes[roi_value] == "g":
            value = 225
        elif solution_nodes[roi_value] == "y":
            value = 255

        # change roi color to value
        four_color_mask[mask == roi_value] = value

    return four_color_mask, solution_nodes


if __name__ == "__main__":

    path_mask = Path("mask.tiff")
    mask = tifffile.imread(path_mask)
    four_color_mask, dict_solution = four_color_theorem(mask)

    ## Display image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].set_axis_off()
    ax[0].set_title(f"original mask \n unique rois: {len(np.unique(mask))-1}")
    ax[0].imshow(mask)

    ax[1].set_axis_off()
    ax[1].set_title(
        f"four color mask \n unique rois: {len(np.unique(four_color_mask))-1}"
    )
    ax[1].imshow(four_color_mask)

    plt.imshow(four_color_mask)
    plt.show()
