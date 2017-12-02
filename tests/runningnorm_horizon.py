
import numpy as np

from test_setup import timer
from mannequin import RunningNormalize

print("Horizon = 5, paired values")
n = RunningNormalize((3,), horizon=5)
for i in range(10):
    print(n([[10, -10, 0.1], [12, 10, 0.2]]).reshape(-1))

print("Horizon = 5, alternating values")
n = RunningNormalize((3,), horizon=5)
for i in range(10):
    print(np.concatenate((n([10.0, -10, 0.1]), n([12, 10, 0.2])), axis=0))

print("Horizon = 5, up and down")
n = RunningNormalize(horizon=5)
n(-10.0)
print(np.array([n(10.0) for i in range(10)]))
print(np.array([n(-10.0) for i in range(10)]))

print("Horizon = 2, linear shift")
n = RunningNormalize(horizon=5)
print(np.array([n(np.arange(11) + i) for i in range(3)]))
