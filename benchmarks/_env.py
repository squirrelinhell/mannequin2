
import gym
from mannequin.gym import PrintRewards

def builder(name, print_every=1000, steps=400000):
    def build():
        env = gym.make(name)
        env = PrintRewards(env, every=print_every)
        return env, lambda: env.total_steps / steps
    return build

cartpole = builder("CartPole-v1", steps=40000)
walker = builder("BipedalWalker-v2", print_every=2048)
lander = builder("LunarLander-v2", print_every=2048)
