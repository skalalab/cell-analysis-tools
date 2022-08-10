# Dependencies
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc

if __name__ == "__main__":

    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pylab as plt

    from cell_analysis_tools.io import load_sdt_file

    mpl.rcParams["figure.dpi"] = 600
    import numpy as np

    from cell_analysis_tools.image_processing import bin_sdt

    # variables
    laser_angular_frequency = 80e6

    ### T Cells
    # path_sdt = Path("C:/Users/Nabiki/Desktop/data/T_cells-paper/Data/011118 - Donor 4/SDT Files/Tcells-001.sdt")
    # sdt_im = load_sdt_file(path_sdt).squeeze()
    # n_rows, n_cols, n_timebins = sdt_im.shape
    # integration_time = 1 / laser_angular_frequency
    # timebins = np.linspace(0, integration_time, n_timebins, endpoint=False)
    # decay = np.sum(sdt_im, axis=(0,1))
    # plt.plot(decay)
    # plt.show()
    # plt.imshow(sdt_im.sum(axis=2))
    # plt.show()

    HERE = Path(__file__).resolve().parent
    ########### neutrohpils
    # working_dir = Path(r"C:\Users\Nabiki\Desktop\development\cell-analysis-tools\cell_analysis_tools\example_data\t_cell".replace('\\','/'))
    # path_sdt = working_dir / "Tcells-001.sdt"
    # path_sdt = Path(HERE.parent / "example_data/neutrophils/Neutrophils-021_NADH.sdt")

    ###### LOAD SDT FILE
    # Kelsey IRF's
    # irf = tifffile.imread( Path(HERE.parent / "example_data/neutrophils/Neutrophils-021_IRF.tiff"))
    # irf = read_asc("Z:/0-Projects and Experiments/KM - OMI Phasor Plots/40x_WID_2019Mar_IRF.asc")
    irf = read_asc(HERE / "example_data/irf_40xW_02_dec2017_IRF.asc")

    irf_timebins = irf[:, 0] * 1e-9  # timebins in ns
    irf_decay = irf[:, 1]  # photons count

    ###### LOAD SDT FILE
    path_sdt = Path(
        HERE / "example_data/EPC16_Day8_4n-063/LifetimeData_Cycle00001_000001.sdt"
    )

    im_sdt = load_sdt_file(path_sdt).squeeze()
    n_rows, n_cols, n_timebins = im_sdt.shape
    integration_time = 1 / laser_angular_frequency
    timebins = np.linspace(0, integration_time, n_timebins, endpoint=False)
    decay = np.sum(im_sdt, axis=(0, 1))
    plt.plot(decay)
    plt.show()
    plt.imshow(im_sdt.sum(axis=2))
    plt.show()

    # REMOVE DECAY OFFSET
    # 7x7 bin
    im_sdt_binned = bin_sdt(im_sdt, bin_size=3, debug=True)

    # threshold decays
    decays = im_sdt_binned[im_sdt_binned.sum(axis=2) > 2000]

    # calculate shift here after removing bg

    # show First 100 decays
    # for d in decays[:100]:
    #     plt.plot(d)
    # plt.show()

    # compute calibration after irf aligned
    irf_lifetime = 0
    calibration = phasor_calibration(
        laser_angular_frequency, irf_lifetime, irf_timebins, irf_decay
    )

    # COMPUTE G AND S VALUES
    array_phasor = [
        td_to_fd(laser_angular_frequency, irf_timebins, decay) for decay in decays
    ]

    # compute g and s for
    list_gs = [
        phasor_to_rectangular_point(
            ph.angle + calibration.angle, ph.magnitude * calibration.scaling_factor
        )
        for ph in array_phasor
    ]

    g = [point.g for point in list_gs]
    s = [point.s for point in list_gs]
    counts = decays.sum(axis=1)

    # plot
    figure = plt.figure()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.axis("equal")
    plt.scatter(
        g, s, s=1, cmap="viridis_r", alpha=1
    )  # s=size, c= colormap_data, cmap=colormap to use
    # plt.colorbar()
    draw_universal_semicircle(figure, laser_angular_frequency)
