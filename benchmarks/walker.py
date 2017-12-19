#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin.basicnet import Input, Affine, Tanh, Multiplier
from mannequin.logprob import Gauss
from mannequin.gym import PrintRewards, NormalizedObservations

def run():
    from _algo import ppo as optimize ### policy / ppo
    from _adv import gae as target ### discounting / gae

    print("# steps reward")
    env = gym.make("BipedalWalker-v2")
    env = PrintRewards(env, every=2048)
    env = NormalizedObservations(env)

    policy = Input(env.observation_space.low.size)
    policy = Tanh(Affine(policy, 64))
    policy = Tanh(Affine(policy, 64))
    policy = Affine(policy, env.action_space.low.size)
    policy = Multiplier(policy, 0.1)
    policy = Gauss(mean=policy)

    optimize(policy, target(env, policy.sample), steps=400000)

if __name__ == "__main__":
    run()
