#!/usr/bin/env python3

def build_problem():
    import numpy as np
    import gym

    from mannequin.basicnet import Input, Affine, LReLU
    from mannequin.gym import ArgmaxActions

    env = gym.make("CartPole-v1")
    env = ArgmaxActions(env)

    rng = np.random.RandomState()

    model = Input(env.observation_space.low.size)
    model = LReLU(Affine(model, 64))
    model = Affine(model, env.action_space.low.size)

    def softmax(v):
        v = v.T
        v = np.exp(v - np.amax(v, axis=0))
        v /= np.sum(v, axis=0)
        return v.T

    def policy(obs):
        outs, backprop = model.evaluate(obs)
        outs = softmax(outs)
        return np.eye(model.n_outputs)[
            rng.choice(model.n_outputs, p=outs)
        ]

    def param_gradient(traj):
        outs, backprop = model.evaluate(traj.o)
        outs = softmax(outs)
        grad = np.multiply((traj.a - outs).T, traj.r).T
        return backprop(grad)

    model.policy = policy
    model.param_gradient = param_gradient

    return env, model

def start_render_thread(opt):
    import ctypes
    import multiprocessing
    import numpy as np
    from mannequin.gym import episode

    def shared_array(shape):
        size = int(np.prod(shape))
        buf = multiprocessing.Array(ctypes.c_double, size)
        arr = np.frombuffer(buf.get_obj())
        return arr.reshape(shape)

    shared_params = shared_array(opt.get_value().size)

    def render_loop():
        env, actor = build_problem()
        while True:
            actor.load_params(shared_params)
            try:
                episode(env, actor.policy, render=True)
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
    import sys
    import numpy as np

    sys.path.append("../..")
    from mannequin import RunningNormalize, Adams
    from mannequin.gym import episode

    env, agent = build_problem()

    opt = Adams(
        np.random.randn(agent.n_params) * 0.1,
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    if render:
        start_render_thread(opt)

    print("# episode reward", flush=True)
    for ep in range(1, 201):
        # Run one episode using current policy
        agent.load_params(opt.get_value())
        traj = episode(env, agent.policy)
        print("%d %.4f" % (ep, np.sum(traj.r)), flush=True)

        # Policy gradient step
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)
        opt.apply_gradient(agent.param_gradient(traj))

if __name__ == "__main__":
    run(**{a[2:]: True for a in __import__("sys").argv[1:]})
