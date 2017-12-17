#!/usr/bin/env python3

import sys
import ctypes
import multiprocessing
import numpy as np
import gym

sys.path.append("../..")
from mannequin import Adam, bar
from mannequin.basicnet import Layer, Input, Tanh, Affine
from mannequin.autograd import AutogradLayer
from mannequin.gym import NormalizedObservations, PrintRewards, episode

from _gae import GAE

class GaussLogDensity(AutogradLayer):
    def __init__(self, inner):
        import autograd.numpy as np

        logstd = np.zeros(inner.n_outputs, dtype=np.float32)

        def f(inps, logstd, *, sample):
            return -0.5 * np.sum(
                logstd + np.square((sample - inps) / np.exp(logstd)),
                axis=-1,
                keepdims=True
            )

        super().__init__(inner, f=f, n_outputs=1, params=logstd)
        self.get_std = lambda: np.exp(logstd[:])

class Policy(object):
    def __init__(self, ob_space, ac_space):
        rng = np.random.RandomState()

        mean = Input(ob_space.low.size)
        mean = Tanh(Affine(mean, 64))
        mean = Tanh(Affine(mean, 64))
        mean = Affine(mean, ac_space.low.size)
        density = GaussLogDensity(mean)

        def sample(obs):
            m, _ = mean.evaluate(obs)
            return m + rng.randn(*m.shape) * density.get_std()

        def param_gradient(traj, baseline, clip=0.2):
            outs, backprop = density.evaluate(traj.o, sample=traj.a)
            outs = outs.reshape(-1)
            grad = np.exp(outs - baseline)
            grad[np.logical_and(grad > 1.0 + clip, traj.r > 0.0)] = 0.0
            grad[np.logical_and(grad < 1.0 - clip, traj.r < 0.0)] = 0.0
            grad *= traj.r
            return backprop(grad)

        def baseline(inps, sample):
            outs, _ = density.evaluate(inps, sample=sample)
            return outs.reshape(-1)

        self.n_params = density.n_params
        self.get_params = density.get_params
        self.load_params = density.load_params
        self.sample = sample
        self.param_gradient = param_gradient
        self.baseline = baseline

def start_render_thread(env, opt):
    def shared_array(shape):
        size = int(np.prod(shape))
        buf = multiprocessing.Array(ctypes.c_double, size)
        arr = np.frombuffer(buf.get_obj())
        return arr.reshape(shape)

    shared_params = shared_array(opt.get_value().size)
    shared_mean = shared_array(env.get_mean().size)
    shared_std = shared_array(env.get_std().size)

    def render_loop():
        env = gym.make("BipedalWalker-v2")
        policy = Policy(env.observation_space, env.action_space)
        def action(obs):
            obs = (obs - shared_mean) / np.maximum(1e-8, shared_std)
            return policy.sample(obs)
        while True:
            policy.load_params(shared_params)
            try:
                episode(env, action, render=True, max_steps=400)
            except:
                pass

    orig_apply = opt.apply_gradient
    def apply_and_update(*args, **kwargs):
        orig_apply(*args, **kwargs)
        shared_params[:] = opt.get_value()
        shared_mean[:] = env.get_mean()
        shared_std[:] = env.get_std()
    opt.apply_gradient = apply_and_update

    multiprocessing.Process(
        target=render_loop,
        daemon=True
    ).start()

def run(render=False):
    env = gym.make("BipedalWalker-v2")
    env = NormalizedObservations(env)

    policy = Policy(env.observation_space, env.action_space)
    opt = Adam(policy.get_params())

    if render:
        start_render_thread(env, opt)
        env = PrintRewards(env, every=2048,
            print=lambda s, r: print(bar(r, 300.0)))
    else:
        print("# steps reward")
        env = PrintRewards(env, every=2048)

    gae = GAE(env)

    for _ in range(200):
        policy.load_params(opt.get_value())
        traj = gae.get_chunk(policy.sample, 2048)

        baseline = policy.baseline(traj.o, traj.a)
        for _ in range(320):
            policy.load_params(opt.get_value())
            idx = np.random.randint(len(traj), size=64)
            grad = policy.param_gradient(traj[idx], baseline[idx])
            opt.apply_gradient(grad, lr=0.0003)

if __name__ == "__main__":
    run(**{a[2:]: True for a in __import__("sys").argv[1:]})
