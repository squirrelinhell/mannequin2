#!/usr/bin/env python3

import sys
import ctypes
import multiprocessing
import numpy as np
import gym

sys.path.append("../..")
from mannequin import Adam, bar
from mannequin.basicnet import Input, Tanh, Affine
from mannequin.gym import NormalizedObservations, episode

from gae import GAE

def sanitize(arr):
    return np.array([float(v) for v in arr.reshape(-1)]).reshape(arr.shape)

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

class Policy(object):
    def __init__(self, ob_space, ac_space):
        model = Input(ob_space.low.size)
        model = Tanh(Affine(model, 64))
        model = Tanh(Affine(model, 64))
        model_mean = Affine(model, ac_space.low.size)
        model = GaussLogDensity(model_mean)

        def param_gradient(inps, sampled, density_grad):
            model.sampled = sanitize(sampled)
            outs, backprop = model.evaluate(inps)
            return backprop(density_grad)

        def logprobs(inps, sampled):
            model.sampled = sanitize(sampled)
            outs, backprop = model.evaluate(inps)
            return outs.reshape(-1)

        self.n_params = model.n_params
        self.get_params = model.get_params
        self.load_params = model.load_params
        self.sample = model.sample
        self.param_gradient = param_gradient
        self.logprobs = logprobs

class PPO(object):
    def __init__(self):
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

        def logprobs_grad(logprobs, old_logprobs_v, atarg_v):
            old_logprobs[:] = old_logprobs_v
            atarg[:] = atarg_v
            _, backprop = loss.evaluate(logprobs)
            backprop(np.ones(64, dtype=np.float32) / -64.0)
            return -input_layer.last_gradient.reshape(-1)

        self.logprobs_grad = logprobs_grad

def start_render_thread(opt):
    def shared_array(shape):
        size = int(np.prod(shape))
        buf = multiprocessing.Array(ctypes.c_double, size)
        arr = np.frombuffer(buf.get_obj())
        return arr.reshape(shape)

    shared_params = shared_array(opt.get_value().size)

    def render_loop():
        env = gym.make("BipedalWalker-v2")
        env = NormalizedObservations(env)
        policy = Policy(env.observation_space, env.action_space)
        while True:
            policy.load_params(shared_params)
            try:
                episode(env, policy.sample, render=True, max_steps=400)
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

class PrintRewards(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        print("# episode step reward", flush=True)
        ep_number = 0
        ep_rew = None
        total_steps = 0
        def do_step(action):
            nonlocal ep_rew, total_steps
            obs, reward, done, info = self.env._step(action)
            assert ep_rew is not None
            total_steps += 1
            ep_rew += reward
            if done:
                print("%d %d %.4f"
                    % (ep_number, total_steps, ep_rew), flush=True)
                sys.stderr.write(bar(ep_rew, 300.0) + "\n")
                ep_rew = None
            return obs, reward, done, info
        def do_reset():
            nonlocal ep_number, ep_rew
            ep_number += 1
            ep_rew = 0.0
            return self.env._reset()
        self._step = do_step
        self._reset = do_reset

def run(render=False):
    env = gym.make("BipedalWalker-v2")
    env = PrintRewards(env)
    env = NormalizedObservations(env)

    gae = GAE(env)
    agent = Policy(env.observation_space, env.action_space)
    ppo = PPO()

    opt = Adam(
        agent.get_params() * 0.01,
        epsilon=1e-5,
        lr=0.003
    )

    if render:
        start_render_thread(opt)

    for i in range(200):
        sys.stderr.write("Preparing segment %d...\n" % (i+1))
        agent.load_params(opt.get_value())
        traj = gae.get_chunk(agent.sample, 2048)

        sys.stderr.write("Baseline...\n")
        baseline = agent.logprobs(traj.o, traj.a)
        sys.stderr.write("Optimizing on segment %d...\n" % (i+1))
        for _ in range(320):
            agent.load_params(opt.get_value())
            idx = np.random.randint(len(traj), size=64)
            logprobs = agent.logprobs(traj.o[idx], traj.a[idx])
            logprobs_grad = ppo.logprobs_grad(logprobs, baseline[idx], traj.r[idx])
            grad = agent.param_gradient(traj.o[idx], traj.a[idx], logprobs_grad)
            opt.apply_gradient(grad)

    if render:
        while True:
            episode(env, agent.policy, render=True)

if __name__ == "__main__":
    run(**{a[2:]: True for a in __import__("sys").argv[1:]})
