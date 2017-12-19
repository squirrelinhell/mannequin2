#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin.basicnet import Input, Affine, LReLU
from mannequin.logprob import Discrete
from mannequin.gym import PrintRewards

def run():
    from _algo import policy as optimize ### policy / ppo
    from _adv import discounting as target ### discounting / gae

    print("# steps reward")
    env = gym.make("CartPole-v1")
    env = PrintRewards(env)

    policy = Input(env.observation_space.low.size)
    policy = LReLU(Affine(policy, 64))
    policy = Affine(policy, env.action_space.n)
    policy = Discrete(logits=policy)

    optimize(policy, target(env, policy.sample), steps=100000)

if __name__ == "__main__":
    run()
