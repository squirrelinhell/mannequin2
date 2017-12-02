
import gym
import numpy as np

from test_setup import timer
from mannequin.gym import ArgmaxActions, episode

env = ArgmaxActions(gym.make("CartPole-v1"))
trajs = [episode(env, lambda o: np.random.rand(2)) for _ in range(32)]

mean_reward = np.mean([np.sum(t.r) for t in trajs])
assert (mean_reward > 15.0) and (mean_reward < 30.0)

assert timer() < 0.1
