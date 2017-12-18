
import numpy as np

from mannequin import RunningNormalize, Adam, Adams

def chunks(trajs, length):
    buf = []
    for t in trajs:
        buf = t if len(buf) <= 0 else buf.joined(t)
        if len(buf) >= length:
            yield buf[:length]
            buf = buf[length:] if len(buf) >= length + 1 else []

def policy(*, logprob, trajs):
    opt = Adams(
        logprob.get_params(),
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    for traj in trajs:
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = logprob.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r))
        logprob.load_params(opt.get_value())

def ppo(*, logprob, trajs,
        lr=0.5, optim_batch=64, optim_steps=300):
    opt = Adam(logprob.get_params())

    def normalize(v):
        return (v - np.mean(v)) / max(1e-6, np.std(v))

    for traj in chunks(trajs, 2048):
        traj = traj.modified(rewards=normalize)

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
