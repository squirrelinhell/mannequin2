#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin.basicnet import Input, Affine, LReLU
from mannequin.logprob import Discrete
from mannequin.gym import PrintRewards

def run():
    from _algo import policy as solve ### policy / ppo

    print("# steps reward")
    env = gym.make("CartPole-v1")
    env = PrintRewards(env)

    policy = Input(env.observation_space.low.size)
    policy = LReLU(Affine(policy, 64))
    policy = Affine(policy, env.action_space.n)
    policy = Discrete(logits=policy)

    solve(env, logprob=policy, steps=40000)

if __name__ == "__main__":
    run()
