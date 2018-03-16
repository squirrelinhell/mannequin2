#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("..")

def run():
    from mannequin import RunningNormalize, Adam
    from mannequin.gym import NormalizedObservations
    from _env import build_env, get_progress
    from policy import stochastic_policy
    from gae import gae

    env = build_env()
    env = NormalizedObservations(env)

    policy = stochastic_policy(env)
    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=2)
    get_chunk = gae(env, policy.sample)

    while get_progress() < 1.0:
        traj = get_chunk()
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        baseline = policy.logprob(traj.o, sample=traj.a)

        for _ in range(300):
            idx = np.random.randint(len(traj), size=64)
            logp, backprop = policy.logprob.evaluate(traj.o[idx],
                sample=traj.a[idx])

            grad = np.exp(logp - baseline[idx]).reshape(-1)
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]

            opt.apply_gradient(backprop(grad), lr=0.0003)
            policy.load_params(opt.get_value())

if __name__ == '__main__':
    run()
