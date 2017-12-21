#!/usr/bin/env python3

import sys
import gym
import numpy as np

sys.path.append("..")
from mannequin import RunningNormalize, Adam
from mannequin.basicnet import Input, Affine, Tanh, Multiplier
from mannequin.logprob import Gauss
from mannequin.gym import NormalizedObservations, episode

from _env import lander as build_env ### walker / lander

class DiscountedChunks(object):
    def __init__(self, env, *, horizon=500):
        buf = []

        def get_chunk(policy, length):
            nonlocal buf
            while len(buf) < length:
                t = episode(env, policy).discounted(horizon=horizon)
                buf = t if len(buf) <= 0 else buf.joined(t)
            ret = buf[:length]
            buf = buf[length:] if len(buf) >= length + 1 else []
            return ret

        self.get_chunk = get_chunk

def ppo(logprob, env, get_progress):
    chunks = DiscountedChunks(env)
    opt = Adam(logprob.get_params())
    normalize = RunningNormalize(horizon=2)

    while get_progress() < 1.0:
        traj = chunks.get_chunk(logprob.sample, 2048)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        baseline = logprob(traj.o, sample=traj.a)

        for _ in range(320):
            idx = np.random.randint(len(traj), size=64)
            logp, backprop = logprob.evaluate(traj.o[idx],
                sample=traj.a[idx])

            grad = np.exp(logp - baseline[idx]).reshape(-1)
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]

            opt.apply_gradient(backprop(grad), lr=0.0003)
            logprob.load_params(opt.get_value())

def run():
    env = build_env()
    get_progress = (lambda e: (lambda: e.progress))(env)
    env = NormalizedObservations(env)

    policy = Input(env.observation_space.low.size)
    policy = Tanh(Affine(policy, 64))
    policy = Tanh(Affine(policy, 64))
    policy = Affine(policy, env.action_space.low.size)
    policy = Multiplier(policy, 0.1)
    policy = Gauss(mean=policy)

    ppo(policy, env, get_progress)

if __name__ == '__main__':
    run()
