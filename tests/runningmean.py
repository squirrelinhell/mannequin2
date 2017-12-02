
import numpy as np

from test_setup import timer
from mannequin import RunningMean

for v in [1e-4, 0.01, 0.1, 1.0, 10.0, 1e+4]:
    m = RunningMean(horizon=5)
    m.update(v)
    print(np.round(m.get(), 5))

print()
m = RunningMean(horizon=5)
for i in range(1, 11):
    m.update(i)
    print(np.round(m.get(), 3))

print()
m = RunningMean(horizon=5)
m.update(np.arange(1, 6))
print(np.round(m.get(), 3))
m.update(np.arange(6, 11))
print(np.round(m.get(), 3))

print()
m = RunningMean((2,), horizon=5)
for i in range(1, 11):
    m.update([i, 11-i])
    print(m.get())

print()
m = RunningMean((2,), horizon=100)
for i in range(1, 11):
    m.update([i, 11-i])
    print(m.get())

assert timer() < 0.1
