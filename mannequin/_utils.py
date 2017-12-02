
class RunningMean(object):
    def __init__(self, shape=(), *, horizon):
        import numpy as np

        discount = 1.0 - 1.0 / float(horizon)
        assert (discount > 0.0) and (discount <= 0.999999)

        mean = np.zeros(shape, dtype=np.float64)
        missing_weight = 1.0

        def update(values, *, weight=1.0):
            nonlocal mean, missing_weight

            values = np.asarray(values, dtype=np.float64)
            values = values.reshape(shape)

            cur_discount = np.power(discount, weight)
            missing_weight *= cur_discount
            mean += (1.0 - cur_discount) * (values - mean)

        self.get = lambda: mean * (1.0 / (1.0 - missing_weight))
        self.update = update

    def __call__(self, values, **kwargs):
        self.update(values, **kwargs)
        return self.get()

class RunningNormalize(object):
    def __init__(self, shape=(), *, horizon):
        import numpy as np

        r_mean = RunningMean(shape=shape, horizon=horizon)
        r_var = RunningMean(shape=shape, horizon=horizon)

        def update(value):
            value = np.array(value, dtype=np.float64)
            value = value.reshape((-1,) + shape)
            value -= r_mean(np.mean(value, axis=0))
            r_var.update(np.mean(np.square(value), axis=0))

        def apply(value):
            return (
                (value - r_mean.get())
                / np.maximum(1e-8, np.sqrt(r_var.get()))
            )

        self.update = update
        self.apply = apply

    def __call__(self, value):
        self.update(value)
        return self.apply(value)

def discounted_rewards(rewards, *, horizon):
    import numpy as np

    step = 1.0 / float(horizon)
    assert (step > 1e-6) and (step < 1.0)

    rewards = np.array(rewards, dtype=np.float64)
    rew_sum = 0.0

    for i in range(len(rewards)-1, -1, -1):
        rew_sum += step * (rewards[i] - rew_sum)
        rewards[i] = rew_sum

    return rewards

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
