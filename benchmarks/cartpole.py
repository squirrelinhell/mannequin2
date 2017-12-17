#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin.basicnet import Input, Affine, LReLU
from mannequin.gym import PrintRewards

from _algo import policy_gradient as solve ### policy_gradient / ppo
from _policy import SoftmaxPolicy

def run():
    print("# steps reward")
    env = gym.make("CartPole-v1")
    env = PrintRewards(env)

    policy = Input(env.observation_space.low.size)
    policy = LReLU(Affine(policy, 64))
    policy = Affine(policy, env.action_space.n)
    policy = SoftmaxPolicy(policy)

    solve(env, policy, steps=40000)

if __name__ == "__main__":
    run()
