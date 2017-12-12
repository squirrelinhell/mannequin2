
import numpy as np

class RunningMean(object):
    def __init__(self, shape=(), *, horizon=None):
        mean = np.zeros(shape, dtype=np.float64)
        total_weight = 0.0

        def update(values, *, weight=1.0):
            nonlocal mean, total_weight

            values = np.asarray(values, dtype=np.float64)
            values = values.reshape(shape)

            weight = float(weight)
            assert weight >= 0.0

            if horizon is not None:
                total_weight *= np.power(
                    1.0 - 1.0 / float(horizon),
                    weight
                )

            total_weight += weight
            mean += (weight / total_weight) * (values - mean)

        self.get = lambda: np.array(mean)
        self.update = update

    def __call__(self, values, **kwargs):
        self.update(values, **kwargs)
        return self.get()

class RunningNormalize(object):
    def __init__(self, shape=(), *, horizon=None):
        r_mean = RunningMean(shape=shape, horizon=horizon)
        r_var = RunningMean(shape=shape, horizon=horizon)
        n_samples = 0

        def update(value):
            nonlocal n_samples

            value = np.array(value, dtype=np.float64)
            value = value.reshape((-1,) + shape)

            d1 = value - r_mean.get()
            d2 = value - r_mean(np.mean(value, axis=0))

            if n_samples >= 1:
                r_var.update(np.mean(np.multiply(d1, d2), axis=0))
            elif len(value) >= 2:
                weight = (len(value) - 1.0) / len(value)
                r_var.update(
                    np.mean(np.multiply(d1, d2), axis=0) / weight,
                    weight=weight
                )

            n_samples += len(value)

        def apply(value):
            if n_samples < 2:
                return np.zeros_like(value)
            return (
                (value - r_mean.get())
                / np.maximum(1e-8, np.sqrt(r_var.get()))
            )

        self.update = update
        self.apply = apply
        self.get_mean = lambda: r_mean.get()
        self.get_var = lambda: r_var.get()
        self.get_std = lambda: np.sqrt(r_var.get())

    def __call__(self, value):
        self.update(value)
        return self.apply(value)

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
