#!/usr/bin/env python3

class Agent(object):
    def __init__(self, env):
        import numpy as np
        from mannequin.basicnet import Input, Affine, LReLU

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

        self.n_params = model.n_params
        self.load_params = model.load_params
        self.policy = policy
        self.param_gradient = param_gradient

def run():
    import sys
    import numpy as np
    import gym

    sys.path.append("../..")
    from mannequin import RunningNormalize, Adams
    from mannequin.gym import ArgmaxActions, PrintRewards, episode

    env = gym.make("CartPole-v1")
    env = PrintRewards(env)
    env = ArgmaxActions(env)

    agent = Agent(env)

    opt = Adams(
        np.random.randn(agent.n_params) * 0.1,
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    time = 0
    while time < 20000:
        # Run one episode using current policy
        agent.load_params(opt.get_value())
        traj = episode(env, agent.policy)
        time += len(traj)

        # Policy gradient step
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)
        opt.apply_gradient(agent.param_gradient(traj))

if __name__ == "__main__":
    run()
