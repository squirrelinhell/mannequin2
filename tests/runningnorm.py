
import numpy as np

from test_setup import timer
from mannequin import RunningNormalize

n = RunningNormalize((3,), horizon=5)
for i in range(10):
    a, b = n([[10, -10, 0.1], [12, 10, 0.2]])
    print(a, b)

print()
n = RunningNormalize((3,), horizon=5)
for i in range(10):
    a, b = n([10.0, -10, 0.1]), n([12, 10, 0.2])
    a = np.round(a, 5)
    a[a == 0.] = 0.
    print(a, b)

print()
n = RunningNormalize(horizon=5)
n(10.0)
print(np.array([n(-10.0) for i in range(10)]))
print(np.array([n(10.0) for i in range(10)]))

print()
n = RunningNormalize(horizon=5)
print(np.array([n(np.arange(11) + i) for i in range(3)]))
