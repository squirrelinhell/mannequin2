
import numpy as np
import matplotlib.pyplot as plt

def plot_function_2d(function, *,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        vlim=(None, None),
        grid=20,
        file_name=None):
    coords = (np.mgrid[0:grid+1,0:grid+1].reshape(2,-1).T / grid
        * [xlim[1] - xlim[0], ylim[1] - ylim[0]] + [xlim[0], ylim[0]])
    values = np.array([float(function(c)) for c in coords])
    values = values.reshape((grid+1, grid+1))
    plt.clf()
    plt.imshow(
        values.T[::-1,:], vmin=vlim[0], vmax=vlim[1],
        zorder=0, aspect="auto",
        cmap="inferno", interpolation="bicubic",
        extent=[xlim[0], xlim[1], xlim[0], ylim[1]]
    )
    plt.colorbar()
    if file_name is None:
        plt.ion()
        plt.show()
        plt.pause(0.0001)
    else:
        plt.gcf().set_size_inches(10, 8)
        plt.gcf().savefig(file_name, dpi=100)
