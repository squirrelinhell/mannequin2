
import numpy as np

from test_setup import timer
from mannequin import RunningMean

print(np.array([
    RunningMean(horizon=5)(v)
    for v in [1e-4, 0.01, 0.1, 1.0, 10.0, 1e+4]
]))

print()
m = RunningMean(horizon=5)
print(np.array([m(i) for i in range(1, 11)]))
m = RunningMean((2,), horizon=5)
print(np.array([m([i, 11-i]) for i in range(1, 11)]).T)

print()
m = RunningMean((2,), horizon=100)
print(np.array([m([i, 11-i]) for i in range(1, 11)]))

print()
m = RunningMean((2,), horizon=10)
print(np.array([
    [m([i, 11-i]), m([i, 11-i])][1]
    for i in range(1, 11)
]).T)
m = RunningMean((2,), horizon=10)
print(np.array([
    m([i, 11-i], weight=2)
    for i in range(1, 11)
]).T)

assert timer() < 0.1
