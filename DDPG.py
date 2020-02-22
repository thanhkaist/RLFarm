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
        self.qModel = QModel("critic", 100, 10)
        self.piModel = PiModel("actor", 100, 10)
        return [self.qModel, self.piModel]