
import numpy as np

from test_setup import timer
from mannequin import RunningMean

print("No horizon")
m = RunningMean()
print(np.array([m(i) for i in range(1, 11)]))
m = RunningMean((2,))
print(np.array([m([i, 11-i]) for i in range(1, 11)]).T)

print("Horizon = 5")
m = RunningMean(horizon=5)
print(np.array([m(i) for i in range(1, 11)]))
m = RunningMean((2,), horizon=5)
print(np.array([m([i, 11-i]) for i in range(1, 11)]).T)

print("Horizon = 100")
m = RunningMean((2,), horizon=100)
print(np.array([m([i, 11-i]) for i in range(1, 11)]).T)

print("Double updates vs weight=2.0")
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
