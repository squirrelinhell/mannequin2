
import numpy as np
from mannequin import RunningNormalize
from test_setup import timer

###

n = RunningNormalize()
data = np.arange(10) + 10
n.update(data[0])
for d in [data[:i] for i in range(2, len(data))]:
    n.update(d[-1])
    assert np.abs(n.get_mean() - np.mean(d)) < 1e-10
    assert np.abs(n.get_var() - np.var(d, ddof=1)) < 1e-10

###

n = RunningNormalize()
np.random.seed(123)
data = np.random.rand(10, 5)
for d in [data[:i] for i in range(1, len(data))]:
    n.update(d[-1])
    np_normalized = (
        (np.array([4.0, 5.0]) - np.mean(d))
        / np.std(d, ddof=1)
    )
    assert np.mean(np.abs(n.apply([4.0, 5.0]) - np_normalized)) < 1e-10

###

n = RunningNormalize((5,))
data = np.random.rand(10, 5)
for d in data:
    n.update(d)
np_normalized = (
    (data - np.mean(data, axis=0))
    / np.std(data, axis=0, ddof=1)
)
assert np.mean(np.abs(n.apply(data) - np_normalized)) < 1e-10

###

n = RunningNormalize((5,), horizon=2)
data = np.random.rand(4, 5)
np_data = []
for i, d in enumerate(data):
    n.update(d)
    for _ in range(2**i):
        np_data.append(d)
np_data = np.array(np_data)
np_normalized = (
    (data - np.mean(np_data, axis=0))
    / np.std(np_data, axis=0, ddof=1)
)
assert np.mean(np.abs(n.apply(data) - np_normalized)) < 1e-10

###

assert timer(print_info=False) < 0.1
