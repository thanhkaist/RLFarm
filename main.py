from DDPG import DDPG
from SAC import SAC
import gym


def main():
    env = gym.make('CartPole-v0')
    obs, rw, done, info = env.reset()
    ddpg = DDPG(10, 10e-3, 10e-3, 10e-3)
    model, env = ddpg.train(env)
    print(DDPG.__mro__)
    no_test = 10
    avg_return = ddpg.test(no_test, model, env)
    print("Test score %0.2f" % (avg_return))


if __name__ == "__main__":
    main()
