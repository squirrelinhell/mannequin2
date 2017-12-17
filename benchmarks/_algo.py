
import numpy as np

from mannequin import RunningNormalize, Adam, Adams
from mannequin.gym import episode

from _gae import GAE

def policy_gradient(env, policy, *, steps):
    opt = Adams(
        policy.get_params(),
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    while steps > 0:
        traj = episode(env, policy.sample)
        steps -= len(traj)

        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = policy.logprob(traj.o, traj.a)
        opt.apply_gradient(backprop(traj.r))
        policy.load_params(opt.get_value())

def ppo(env, policy, *, steps):
    gae = GAE(env)
    opt = Adam(policy.get_params())

    for i in range(steps // 2048):
        traj = gae.get_chunk(policy.sample, 2048)
        baseline, _ = policy.logprob(traj.o, traj.a)

        for _ in range(320):
            idx = np.random.randint(len(traj), size=64)
            logp, backprop = policy.logprob(traj.o[idx], traj.a[idx])

            grad = np.exp(logp - baseline[idx])
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]

            opt.apply_gradient(backprop(grad), lr=0.0003)
            policy.load_params(opt.get_value())
