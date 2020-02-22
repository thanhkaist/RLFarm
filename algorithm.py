import numpy as np
from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self, epoch, learning_rate):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.env = "No env"
        ## Should not use this , use only one underscore instead so we can access by child object in some special case 
        self.__private = "Private variable"

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
        for i in range(len(networks)):
          reset_network(networks[i])

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