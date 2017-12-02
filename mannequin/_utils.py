
class RunningMean(object):
    def __init__(self, *, horizon):
        step = 1.0 / float(horizon)
        assert (step > 0.0) and (step < 1.0)

        biased_mean = 0.0
        power = 1.0

        def update(value):
            nonlocal biased_mean, power

            biased_mean += step * (value - biased_mean)
            power *= 1.0 - step

        self.get = lambda: biased_mean / (1.0 - power)
        self.update = update

class RunningNormalize(object):
    def __init__(self, *, horizon):
        import numpy as np

        mean = RunningMean(horizon=horizon)
        var = RunningMean(horizon=horizon)

        def normalize(value):
            value = np.array(value, dtype=np.float32)
            mean.update(np.mean(value))
            value -= mean.get()
            var.update(np.mean(np.square(value)))
            return value / max(1e-8, np.sqrt(var.get()))

        self.normalize = normalize

    def __call__(self, value):
        return self.normalize(value)

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
