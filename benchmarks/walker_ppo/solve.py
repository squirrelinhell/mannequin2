#!/usr/bin/env python3

import sys
import ctypes
import multiprocessing
import numpy as np
import gym

sys.path.append("../..")
from mannequin import Adam, bar
from mannequin.basicnet import Input, Tanh, Affine
from mannequin.gym import NormalizedObservations, PrintRewards, episode

from gae import GAE

def GaussLogDensity(inner):
    import autograd.numpy as np
    from mannequin.autograd import AutogradLayer

    params = np.zeros(inner.n_outputs)
    def density(mean, logstd):
        return -0.5 * np.sum(
            logstd + np.square((layer.sampled - mean) / np.exp(logstd)),
            axis=-1,
            keepdims=True
        )

    def sample(obs):
        m, _ = inner.evaluate(obs)
        layer.sampled = m + np.random.randn(*m.shape) * np.exp(params)
        return layer.sampled

    layer = AutogradLayer(inner, f=density, n_outputs=1, params=params)
    layer.sample = sample
    return layer

def ppo():
    import autograd.numpy as np
    from mannequin.autograd import Input, AutogradLayer
    old_logprobs = np.zeros(64, dtype=np.float32)
    atarg = np.zeros(64, dtype=np.float32)
    def f(inps):
        adv = atarg.reshape(inps.shape)
        ratio = np.exp(inps - old_logprobs.reshape(inps.shape))
        surr1 = np.multiply(ratio, adv)
        surr2 = np.clip(ratio, 0.8, 1.2) * adv
        return np.minimum(surr1, surr2)
    input_layer = Input(1)
    loss = AutogradLayer(input_layer, f=f)
    def grad(logprobs, old_logprobs_v, atarg_v):
        old_logprobs[:] = old_logprobs_v
        atarg[:] = atarg_v
        _, backprop = loss.evaluate(logprobs)
        backprop(np.ones(64, dtype=np.float32) / -64.0)
        return -input_layer.last_gradient.reshape(-1)
    return grad
ppo = ppo()

class Policy(object):
    def __init__(self, ob_space, ac_space):
        model = Input(ob_space.low.size)
        model = Tanh(Affine(model, 64))
        model = Tanh(Affine(model, 64))
        model = Affine(model, ac_space.low.size)
        model = GaussLogDensity(model)

        def param_gradient(traj, baseline):
            model.sampled = traj.a
            outs, backprop = model.evaluate(traj.o)
            grad = ppo(outs, baseline, traj.r)
            return backprop(grad)

        def logprobs(inps, sampled):
            model.sampled = sampled
            outs, backprop = model.evaluate(inps)
            return outs.reshape(-1)

        self.n_params = model.n_params
        self.get_params = model.get_params
        self.load_params = model.load_params
        self.sample = model.sample
        self.param_gradient = param_gradient
        self.logprobs = logprobs

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

    opt = Adam(
        policy.get_params(),
        epsilon=1e-5,
        lr=0.003
    )

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

        baseline = policy.logprobs(traj.o, traj.a)
        for _ in range(320):
            policy.load_params(opt.get_value())
            idx = np.random.randint(len(traj), size=64)
            grad = policy.param_gradient(traj[idx], baseline[idx])
            opt.apply_gradient(grad)

if __name__ == "__main__":
    run(**{a[2:]: True for a in __import__("sys").argv[1:]})
