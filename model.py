from abc import ABC, abstractmethod

# Provide multiple model


class Model:
    def __init__(self, name, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name

    @abstractmethod
    def forward(self, obs):
        return "Q(a|s)"

    def __call__(self, obs):
        return self.forward(obs)

    def act(self, obs):
        print("Disable backprob")
        return self.forward(obs)


class QModel(Model):
    def __init__(self, name, obs_dim, action_dim):
        super().__init__(name, obs_dim, action_dim)

    def forward(self, obs):
        return "Q(a|s)"


class PiModel(Model):
    def __init__(self, name, obs_dim, action_dim):
        super().__init__(name, obs_dim, action_dim)

    def forward(self, obs):
        return "P(a|s)"
