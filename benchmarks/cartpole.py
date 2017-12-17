#!/usr/bin/env python3

import sys
import ctypes
import multiprocessing
import numpy as np
import gym

sys.path.append("..")
from mannequin import RunningNormalize, Adams, bar
from mannequin.basicnet import Input, Affine, LReLU
from mannequin.gym import PrintRewards, episode

class Policy(object):
    def __init__(self, ob_space, ac_space):
        rng = np.random.RandomState()
        eye = np.eye(ac_space.n, dtype=np.float32)

        model = Input(ob_space.low.size)
        model = LReLU(Affine(model, 64))
        model = Affine(model, ac_space.n)

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def sample(obs):
            outs, backprop = model.evaluate(obs)
            outs = softmax(outs)
            return rng.choice(ac_space.n, p=outs)

        def param_gradient(traj):
            outs, backprop = model.evaluate(traj.o)
            outs = softmax(outs)
            actual_dist = eye[traj.a]
            assert actual_dist.shape == outs.shape
            grad = np.multiply((actual_dist - outs).T, traj.r).T
            return backprop(grad)

        self.n_params = model.n_params
        self.get_params = model.get_params
        self.load_params = model.load_params
        self.sample = sample
        self.param_gradient = param_gradient

def start_render_thread(opt):
    def shared_array(shape):
        size = int(np.prod(shape))
        buf = multiprocessing.Array(ctypes.c_double, size)
        arr = np.frombuffer(buf.get_obj())
        return arr.reshape(shape)

    shared_params = shared_array(opt.get_value().size)

    def render_loop():
        env = gym.make("CartPole-v1")
        policy = Policy(env.observation_space, env.action_space)
        while True:
            policy.load_params(shared_params)
            try:
                episode(env, policy.sample, render=True)
            except:
                pass

    orig_apply = opt.apply_gradient
    def apply_and_update(*args, **kwargs):
        orig_apply(*args, **kwargs)
        shared_params[:] = opt.get_value()
    opt.apply_gradient = apply_and_update

    multiprocessing.Process(
        target=render_loop,
        daemon=True
    ).start()

def run(render=False):
    env = gym.make("CartPole-v1")

    policy = Policy(env.observation_space, env.action_space)

    opt = Adams(
        policy.get_params(),
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    if render:
        start_render_thread(opt)
        env = PrintRewards(env, print=lambda s, r: print(bar(r, 500.0)))
    else:
        print("# steps reward")
        env = PrintRewards(env)

    normalize = RunningNormalize(horizon=10)

    while env.total_steps < 40000:
        # Run one episode using current policy
        policy.load_params(opt.get_value())
        traj = episode(env, policy.sample)

        # Policy gradient step
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)
        opt.apply_gradient(policy.param_gradient(traj))

if __name__ == "__main__":
    run(**{a[2:]: True for a in __import__("sys").argv[1:]})
