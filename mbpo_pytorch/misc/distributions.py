import torch
from torch.distributions import Distribution, Normal
import math

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        Z ~ N(mean, std)
        X ~ tanh(Z)
    Note: this is not very numerically stable.
    """
    def __init__(self, mean, std, epsilon=1e-6):
        """
        :param mean: Mean of the normal distribution
        :param std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        super().__init__()
        self.normal_mean = mean
        self.normal_std = std
        self.normal = Normal(mean, std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def log_probs(self, value, pre_tanh_value):
        return self.log_prob(value, pre_tanh_value).sum(-1, keepdim=True)

    def sample(self, sample_shape=torch.Size([])):
        z = self.normal.sample(sample_shape)
        return torch.tanh(z), z

    def rsample(self, sample_shape=torch.Size([]), return_pretanh_value=False):
        z = (
                self.normal_mean +
                self.normal_std *
                Normal(
                    torch.zeros_like(self.normal_mean),
                    torch.ones_like(self.normal_std)
                ).sample()
        )
        z.requires_grad_()
        return torch.tanh(z), z

    def entropy(self):
        return self.normal.entropy().sum(-1)

    def mode(self):
        return torch.tan(self.normal_mean), self.normal_mean


class FixedLimitedEntNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        limit = 2.
        lo, hi = (-limit - self.loc) / self.scale / math.sqrt(2), (limit - self.loc) / self.scale / math.sqrt(2)
        return (0.5 * (self.scale.log() + math.log(2 * math.pi) / 2) * (hi.erf() - lo.erf()) + 0.5 *
                (torch.exp(-hi * hi) * hi - torch.exp(-lo * lo) * lo)).sum(-1)

    def mode(self):
        return self.mean


class FixedCategorical(torch.distributions.Categorical):
    def sample(self, **kwargs):
        return super().sample(**kwargs).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class FixedBernoulli(torch.distributions.Bernoulli):

    def log_probs(self, actions):
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()

