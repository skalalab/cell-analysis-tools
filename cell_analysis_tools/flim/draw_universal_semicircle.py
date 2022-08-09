# Dependencies
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc


def draw_universal_semicircle(
    figure, laser_angular_frequency, title="Phasor Plot", debug=False
):
    """
    Draws the universal semicircle over the give figure.

    Parameters
    ----------
        figure : matplotlib figure object
            figure object to draw semicircle over
        laser_angular_frequency : int
            rep rate or laser angular frequency, affects position of labeled points

    Returns
    -------
        None
    """

    plt.title(title)
    """ get universal semicircle values for this rep rate """
    x_circle, y_circle, g, s, lifetime_labels = universal_semicircle_series(
        laser_angular_frequency
    )

    # Labels and axis of phasor plot
    # figure = plt.figure()
    figure.suptitle(title)
    plt.xlabel("g", fontsize=20)
    plt.ylabel("s", fontsize=20)

    # add circle and lifetime estimates
    plt.plot(x_circle, y_circle, "-", color="teal")
    plt.plot(g, s, ".", color="magenta")

    if debug:
        print("type: ", type(lifetime_labels), " labels: ", lifetime_labels)

    """ ADD LIFETIME LABELS """
    for i, txt in enumerate(lifetime_labels):
        # self.ax.annotate(txt, (g[i], s[i]))
        plt.annotate(txt, (g[i], s[i]))
    plt.show()


def universal_semicircle_series(frequency):
    """

    Given the frequency this function will return x and y points for plotting the
    universal semicircle, as well as the corresponding g and s values for
    common lifetimes between 0.5ns to 10ns

    Parameters
    ----------
        frequency {float} : angular laser repetition frequency
    
    Returns
    -------

        x_circle : float
            x coordinates for universal semicircle
        y_circle : float
            y coordinates for universal semicircle
        g : float
            x coordinates for labeled lifetimes
        s : float
            y coordinates for labeled lifetimes
        lifetime_labels : list
            labels to be applied to the (g,s) coordinates
    """
    x_coord = np.linspace(0, 1, num=1000)
    y_circle = np.sqrt(x_coord - x_coord ** 2)
    x_circle = x_coord ** 2 + y_circle ** 2

    omega = 2.0 * np.pi * frequency  # modulation frequency
    tau = np.asarray(
        [0.5e-9, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9, 10e-9]
    )  # lifetimes in ns
    g = 1 / (1 + np.square(omega) * np.square(tau))
    s = (omega * tau) / (1 + np.square(omega) * np.square(tau))
    lifetime_labels = [
        "0.5ns",
        "1ns",
        "2ns",
        "3ns",
        "4ns",
        "5ns",
        "6ns",
        "7ns",
        "8ns",
        "9ns",
        "10ns",
    ]
    return x_circle, y_circle, g, s, lifetime_labels
