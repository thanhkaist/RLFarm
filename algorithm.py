import numpy as np
from abc import ABC, abstractmethod
import gym
from model import weights_init

class Algorithm(ABC):
    def __init__(self, epoch, learning_rate):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.env = None

        # env cache
        self.obs = None
        self.act = None
        self.done = None
        self.info = None
        self.reward = None


    def train(self, env, epoch=None):
        self.env = env
        self._prepair()
        if epoch != None:
            self.epoch = epoch

        for i in range(self.epoch):
            print("train %d" % (i))
            self._train()
        return (self.get_model(), self.get_env())

    def _prepair(self):
        networks = self.get_networks()
        def reset_network(nw):
            print("Init nw %s" %(nw.name))
            nw.apply(weights_init)

        for i in range(len(networks)):
          reset_network(networks[i])

        self.obs, self.reward, self.done, self.info = self.env.reset()
        print("Prepair buffer")

    @abstractmethod
    def _train(self):
        print("train the algorithm")

    def get_model(self):
        model = "My trained model"
        return model

    def get_env(self):
        print("reset env")
        return self.env

    @abstractmethod
    def get_networks():
        # return a list of get_networks
        pass

    def test(self, no_test, model, env):
        returns = []
        print("Start to test")
        print("Reset env")
        for i in range(no_test):
            print("\ttest %d" % (i))
            returns.append(i)
            print("\treset %d" % (i))
        return np.average(returns)

    def get_dim(self,env):
        # given env => output (action_dim,obs_dim)
        if isinstance(env.action_space, gym.spaces.Discrete):
            act_dim = env.action_space.n
        else:
            act_dim = env.action_space.shape[0]  # only support vector

        if isinstance(env.observation_space, gym.spaces.Discrete):
            obs_dim = env.observation_space.n
        else:
            obs_dim = env.observation_space.shape[0] # only support vector
        return (act_dim, obs_dim)


