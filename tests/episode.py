
import numpy as np
import gym.spaces
from mannequin.gym import episode, one_step
from test_setup import timer

class FakeEnv(object):
    def __init__(self):
        x = 0.0
        def reset():
            nonlocal x
            x = 0.0
            return [x]
        def step(action):
            nonlocal x
            x += 1.0 if action[1] > action[0] else -1.0
            return [x], (1.0 if x > 3.5 else 0.0), abs(x) > 3.5, {}
        self.reset = reset
        self.step = step
        self.observation_space = gym.spaces.Box(-4.0, 4.0, (1,))
        self.action_space = gym.spaces.Box(0.0, 1.0, (2,))

env = FakeEnv()

traj = episode(env, lambda i: [1.0, 0.0])
for e in zip(traj.o, traj.a, traj.r):
    print(*e)

print()
for i in range(6):
    print(*one_step(env, lambda i: [0.0, 1.0]))

print()
traj = episode(env, lambda i: [0.0, 1.0])
for e in zip(traj.o, traj.a, traj.r):
    print(*e)

assert timer(print_info=False) < 0.02
