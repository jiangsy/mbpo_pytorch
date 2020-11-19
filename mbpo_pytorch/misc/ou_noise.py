import torch

from mbpo_pytorch.models.actor import Actor


class OUNoise(object):

    def __init__(self, action_space, mu=0.0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = action_space
        self.state = None
        self.actor = None

        self.shape = action_space.shape

        self.reset()

    def reset(self):
        self.state = torch.ones(self.shape) * self.mu

    def next(self):
        delta = self.theta * (self.mu - self.state) + self.sigma * torch.randn_like(self.state)
        self.state = self.state + delta
        return self.state

    def act(self, states):
        result = self.actor.act(states)
        return (result[0] + self.next(), *result[1:])

    def wrap(self, actor: Actor):
        self.actor = actor
        self.state = self.state.to(next(actor.parameters()).device)
        return self

