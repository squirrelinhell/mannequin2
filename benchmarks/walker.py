#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin.basicnet import Input, Affine, Tanh, Multiplier
from mannequin.logprob import Gauss
from mannequin.gym import PrintRewards, NormalizedObservations

def run():
    from _algo import ppo as solve ### policy / ppo

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

    solve(env, logprob=policy, steps=400000)

if __name__ == "__main__":
    run()
