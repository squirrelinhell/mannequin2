
import gym
import numpy as np
rng = np.random.RandomState()

from test_setup import timer
from mannequin.gym import ArgmaxActions, LimitedEpisode, episode

def random_policy(obs):
    return np.eye(2)[rng.choice(2)]

env = ArgmaxActions(gym.make("CartPole-v1"))

assert len(episode(LimitedEpisode(env, 3), random_policy)) == 3

trajs = [episode(env, random_policy) for _ in range(32)]
mean_reward = np.mean([np.sum(t.r) for t in trajs])
assert (mean_reward > 15.0) and (mean_reward < 25.0)

assert timer() < 0.1
