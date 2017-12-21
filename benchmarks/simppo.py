#!/usr/bin/env python3

import sys
import gym
import numpy as np

sys.path.append("..")
from mannequin import RunningNormalize, Adam
from mannequin.gym import NormalizedObservations, episode

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

def run():
    from _env import build_env, get_progress, mlp_policy
    env = build_env()
    env = NormalizedObservations(env)

    logprob = mlp_policy(env)
    opt = Adam(logprob.get_params(), horizon=10)

    normalize = RunningNormalize(horizon=2)
    env = DiscountedChunks(env)

    while get_progress() < 1.0:
        traj = env.get_chunk(logprob.sample, 2048)
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

if __name__ == '__main__':
    run()
