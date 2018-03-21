
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
                return np.zeros_like(value, dtype=np.float64)
            return (
                (value - r_mean.get())
                / np.maximum(1e-8, np.sqrt(r_var.get()))
            )

        def get_var():
            if n_samples < 2:
                return np.ones(shape, dtype=np.float64)
            return r_var.get()

        self.update = update
        self.apply = apply
        self.get_mean = lambda: r_mean.get()
        self.get_var = get_var
        self.get_std = lambda: np.sqrt(get_var())

    def __call__(self, value):
        self.update(value)
        return self.apply(value)
