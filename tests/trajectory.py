
from test_setup import timer
from mannequin import Trajectory

t = Trajectory(
    [[1, 2], [3, 4], [5, 6]],
    [10, 11, 12],
    [20, 21, 22]
)

print(t)
print(t[1:])
print(t[1])

t = t[::-1]
print(t.o)
print(t.a)
print(t.r)

t = t.joined(t[0])
print(t.r)

t = t.modified(rewards=sum)
print(t.r)

assert timer() < 0.01
