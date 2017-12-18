
import numpy as np

from mannequin import RunningNormalize, Adam, Adams
from mannequin.gym import episode

from _gae import GAE

def policy(env, *, logprob, steps):
    opt = Adams(
        logprob.get_params(),
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    while steps > 0:
        traj = episode(env, logprob.sample)
        steps -= len(traj)

        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = logprob.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r))
        logprob.load_params(opt.get_value())

def ppo(env, *, logprob, steps,
        lr=0.5, optim_batch=64, optim_steps=300):
    gae = GAE(env)
    opt = Adam(logprob.get_params())

    while steps > 0:
        traj = gae.get_chunk(logprob.sample, 2048)
        steps -= len(traj)

        baseline, _ = logprob.evaluate(traj.o, sample=traj.a)

        for _ in range(optim_steps):
            idx = np.random.randint(len(traj), size=optim_batch)
            p, backprop = logprob.evaluate(traj.o[idx],
                sample=traj.a[idx])

            grad = np.exp(p - baseline[idx]).reshape(-1)
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]

            opt.apply_gradient(backprop(grad), lr=lr/optim_steps)
            logprob.load_params(opt.get_value())
