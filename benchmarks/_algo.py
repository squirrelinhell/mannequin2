
import numpy as np

from mannequin import Trajectory, RunningNormalize, Adam, Adams

def get_chunk(experience, length):
    buf = [next(experience) for _ in range(length)]
    return Trajectory(*zip(*buf))

def policy(logprob, experience, *, steps):
    opt = Adam(logprob.get_params())
    normalize = RunningNormalize(horizon=50)

    steps = steps // 64
    for step in range(steps):
        traj = get_chunk(experience, 64)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = logprob.evaluate(traj.o, sample=traj.a)
        grad = backprop(traj.r)

        opt.apply_gradient(grad, lr=0.01 * (1.0 - step/steps))
        logprob.load_params(opt.get_value())

def ppo(logprob, experience, *, steps, lr=0.5, optim_steps=300):
    opt = Adam(logprob.get_params())

    def normalize(v):
        return (v - np.mean(v)) / max(1e-6, np.std(v))

    for _ in range(steps // 2048):
        traj = get_chunk(experience, 2048)
        traj = traj.modified(rewards=normalize)

        baseline = logprob(traj.o, sample=traj.a)

        for _ in range(optim_steps):
            idx = np.random.randint(len(traj), size=64)
            p, backprop = logprob.evaluate(traj.o[idx],
                sample=traj.a[idx])

            grad = np.exp(p - baseline[idx]).reshape(-1)
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]
            grad = backprop(grad)

            opt.apply_gradient(grad, lr=lr/optim_steps)
            logprob.load_params(opt.get_value())
