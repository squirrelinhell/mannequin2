
import numpy as np

def endswith(a, b):
    if len(a) < len(b):
        return False
    return a[len(a)-len(b):] == b

def discounted(values, *, horizon):
    step = 1.0 / float(horizon)
    assert (step > 1e-6) and (step < 1.0)

    values = np.array(values, dtype=np.float64)
    rew_sum = 0.0

    for i in range(len(values)-1, -1, -1):
        rew_sum += step * (values[i] - rew_sum)
        values[i] = rew_sum

    return values

def bar(value, max_value=100.0, *, length=50):
    value = float(value)
    max_value = abs(float(max_value))
    length = max(1, int(length))

    if value >= 0.0:
        bar = max(0.0, min(1.0, value / max_value))
        bar, frac = divmod(int(round(bar * length * 8)), 8)
        bar = chr(0x2588) * bar
        if frac >= 1:
            bar += chr(0x2590 - frac)
        bar += " " * length
        bar = bar[:length]
    else:
        bar = max(0.0, min(1.0, abs(value) / max_value))
        bar, frac = divmod(int(round(bar * length * 2)), 2)
        bar = chr(0x2501) * bar
        if frac >= 1:
            bar = chr(0x257A) + bar
        bar = " " * length + bar
        bar = bar[-length:]

    fmt_len = len("-%.2f" % max_value)
    fmt = ("%%%d.2f " % fmt_len) % value
    return fmt[0:fmt_len+1] + "[" + bar + "]"

def plot_function_2d(function, *,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        vlim=(None, None),
        grid=20,
        file_name=None):
    import matplotlib.pyplot as plt
    coords = (np.mgrid[0:grid+1,0:grid+1].reshape(2,-1).T / grid
        * [xlim[1] - xlim[0], ylim[1] - ylim[0]] + [xlim[0], ylim[0]])
    values = np.array([float(function(*c)) for c in coords])
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
