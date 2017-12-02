
import numpy as np

from test_setup import timer
from mannequin import RunningNormalize

n = RunningNormalize()
data = np.arange(10) + 10
n.update(data[0])
for d in [data[:i] for i in range(2, len(data))]:
    n.update(d[-1])
    assert np.abs(n.get_mean() - np.mean(d)) < 1e-10
    assert np.abs(n.get_var() - np.var(d, ddof=1)) < 1e-10

data = np.random.rand(10, 5)
n = RunningNormalize((5,))
for d in data:
    n.update(d)
np_normalized = (
    (data - np.mean(data, axis=0))
    / np.std(data, axis=0, ddof=1)
)
assert np.mean(np.abs(n.apply(data) - np_normalized)) < 1e-10
