
import numpy as np

from test_setup import timer
from mannequin import RunningMean

###

m = RunningMean()
data = np.arange(10) + 10
for d in [data[:i] for i in range(1, len(data))]:
    assert np.abs(m(d[-1]) - np.mean(d)) < 1e-10

###

m = RunningMean((7,), horizon=2)
data = np.random.rand(4, 7)
np_data = []
for i, d in enumerate(data):
    for _ in range(2**i):
        np_data.append(d)
    np_mean = np.mean(np_data, axis=0)
    assert np.mean(np.abs(m(d) - np_mean)) < 1e-10

###

m = RunningMean((5,))
data = np.random.rand(4, 5)
m.update(data[0], weight=0.2)
m.update(data[0], weight=0.8)
m.update(data[1])
m.update(data[2], weight=1.9)
m.update(data[2], weight=0.1)
m.update(data[3])
np_mean = np.mean(data[[0, 1, 2, 2, 3]], axis=0)
assert np.mean(np.abs(m.get() - np_mean)) < 1e-10

###

assert timer() < 0.1
