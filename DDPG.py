from algorithm import Algorithm
from model import QModel,PiModel

class DDPG(Algorithm):
    def __init__(self, epoch=100, q_lr=10e-3, m_lr=10e-3, poliak=10e-3):
        super().__init__(epoch, q_lr)
        self.q_lr = q_lr
        self.m_lr = m_lr
        self.poliak = poliak

    def _train(self):
        print("\t Rollout")
        print("\t UpdateQ")
        print("\t UpdatePi")

    def get_networks(self):
        act_dim, obs_dim = self.get_dim(self.env)
        self.qModel = QModel("critic", obs_dim, act_dim)
        print(self.qModel)
        self.piModel = PiModel("actor",obs_dim, act_dim)
        print(self.piModel)
        return [self.qModel, self.piModel]