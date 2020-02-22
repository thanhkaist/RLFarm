from algorithm import Algorithm
from model import QModel,PiModel

class SAC(Algorithm):
    def __init__(self, epoch=100, q_lr=10e-3, m_lr=10e-3, poliak=10e-3):
        super().__init__(epoch, q_lr)
        self.q_lr = q_lr
        self.m_lr = m_lr
        self.poliak = poliak

    def _train(self):
        print("\t SACRollout")
        print("\t SACUpdateQ")
        print("\t SACUpdatePi")

    def get_networks(self):
        self.qModel = QModel("critic", 100, 10)
        self.qModel1 = QModel("critic1", 100, 10)
        self.piModel = PiModel("actor", 100, 10)
        self.piModel2 = PiModel("actor2", 100, 10)
        return [self.qModel,self.qModel1, self.piModel,self.piModel2]