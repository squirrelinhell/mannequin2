
from mannequin import bar
from test_setup import timer

for i in range(-1500, 1500, 234):
    print(bar(i * 0.1))

print(bar(1000.0))
print(bar(10000.0))
print(bar(100000.0))
print(bar(1000000.0))

for i in range(-16, 17):
    print(bar(i / 16.0, 1.0, length=2))

print(bar(0.12))
print(bar(0.13))
print(bar(0.37))
print(bar(0.38))

assert timer(print_info=False) < 0.01
