from abc import ABC

import torch
import torch.nn as nn

from mbpo_pytorch.misc.distributions import FixedNormal, FixedCategorical, FixedBernoulli, \
    TanhNormal, FixedLimitedEntNormal
from .utils import init


class CategoricalActorLayer(nn.Module, ABC):
    def __init__(self, num_inputs, num_outputs):
        super(CategoricalActorLayer, self).__init__()

        self.logit_actor = nn.Linear(num_inputs, num_outputs)
        init(self.logit_actor, lambda x: nn.init.orthogonal_(x, 0.01), lambda x: nn.init.constant_(x, 0))

    def forward(self, states):
        logits = self.logit_actor(states)
        return FixedCategorical(logits=logits)


class GaussianActorLayer(nn.Module, ABC):
    def __init__(self, num_inputs, num_outputs, use_state_dependent_std):
        super(GaussianActorLayer, self).__init__()

        self.actor_mean = nn.Linear(num_inputs, num_outputs)
        init(self.actor_mean, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.use_state_dependent_std = use_state_dependent_std
        if self.use_state_dependent_std:
            self.actor_logstd = nn.Linear(num_inputs, num_outputs)
            init(self.actor_logstd, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        else:
            self.logstd = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)

    def forward(self, x):
        action_mean = self.actor_mean(x)

        if self.use_state_dependent_std:
            logstd = self.actor_logstd(x)
        else:
            logstd = self.logstd

        return FixedNormal(action_mean, logstd.exp()), action_mean, logstd


class LimitedEntGaussianActorLayer(nn.Module, ABC):
    def __init__(self, num_inputs, num_outputs, use_state_dependent_std):
        super(LimitedEntGaussianActorLayer, self).__init__()

        self.mean_actor = nn.Linear(num_inputs, num_outputs)
        init(self.mean_actor, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.use_state_dependent_std = use_state_dependent_std
        if self.use_state_dependent_std:
            self.logstd_actor = nn.Linear(num_inputs, num_outputs)
            init(self.logstd_actor, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        else:
            self.logstd = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
            self.logstd_actor = lambda _: self.logstd

    def forward(self, x):
        action_mean = self.mean_actor(x)
        logstd = self.logstd_actor(x)
        return FixedLimitedEntNormal(action_mean, logstd.exp()), action_mean, logstd


class BernoulliActorLayer(nn.Module, ABC):
    def __init__(self, num_inputs, num_outputs):
        super(BernoulliActorLayer, self).__init__()

        self.logit_actor = nn.Linear(num_inputs, num_outputs)
        init(self.logit_actor, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0))

    def forward(self, states):
        logits = self.logit_actor(states)
        return FixedBernoulli(logits=logits)


class TanhGaussainActorLayer(nn.Module, ABC):
    def __init__(self, num_inputs, num_outputs, use_state_dependent_std, init_w=1e-3):
        super(TanhGaussainActorLayer, self).__init__()

        self.mean_actor = nn.Linear(num_inputs, num_outputs)
        init(self.mean_actor, lambda x: nn.init.uniform_(x, -init_w, init_w),
             lambda x: nn.init.uniform_(x, -init_w, init_w))

        self.state_dependent_std = use_state_dependent_std
        if self.state_dependent_std:
            self.logstd_actor = nn.Linear(num_inputs, num_outputs)
            init(self.logstd_actor, lambda x: nn.init.uniform_(x, -init_w, init_w),
                 lambda x: nn.init.uniform_(x, -init_w, init_w))
        else:
            self.logstd = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
            self.logstd_actor = lambda _: self.logstd

    def forward(self, states):
        action_means = self.mean_actor(states)
        action_logstds = self.logstd_actor(states)
        action_logstds = torch.clamp(action_logstds, -20, 2)

        return TanhNormal(action_means, action_logstds.exp()), torch.tanh(action_means), action_logstds
