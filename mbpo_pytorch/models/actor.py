from __future__ import annotations

from abc import ABC
from typing import List, Optional

from .initializer import normc_init
from .utils import MLP
from .actor_layer import *


# noinspection DuplicatedCode
class Actor(nn.Module, ABC):
    def __init__(self, state_dim: int, action_space, hidden_dims: List[int],
                 state_normalizer: Optional[nn.Module], use_limited_entropy=False, use_tanh_squash=False,
                 use_state_dependent_std=False, **kwargs):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_space
        self.hidden_dims = hidden_dims
        self.use_limited_entropy = use_limited_entropy
        self.use_tanh_squash = use_tanh_squash

        mlp_kwargs = kwargs.copy()
        mlp_kwargs['activation'] = kwargs.get('activation', 'relu')
        mlp_kwargs['last_activation'] = kwargs.get('activation', 'relu')

        self.actor_feature = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1], **mlp_kwargs)

        self.state_normalizer = state_normalizer or nn.Identity()

        self.actor_layer = TanhGaussainActorLayer(hidden_dims[-1], action_space.shape[0],
                                                  use_state_dependent_std)

        init_ = lambda m: init(m, normc_init, lambda x: nn.init.constant_(x, 0))
        self.actor_feature.init(init_, init_)

    def act(self, state, deterministic=False, reparameterize=False):
        action_feature = self.actor_feature(state)
        action_dist, action_means, action_logstds = self.actor_layer(action_feature)

        log_probs = None
        pretanh_actions = None

        if deterministic:
            actions = action_means
        else:
            if reparameterize:
                result = action_dist.rsample()
            else:
                result = action_dist.sample()
            actions, pretanh_actions = result
            log_probs = action_dist.log_probs(actions, pretanh_actions)

        entropy = action_dist.entropy().mean()

        return {'actions': actions, 'log_probs': log_probs, 'entropy': entropy,
                'action_means': action_means, 'action_logstds': action_logstds, 'pretanh_actions': pretanh_actions}

    def evaluate_actions(self, states, actions, pretanh_actions=None):
        states = self.state_normalizer(states)

        action_feature = self.actor_feature(states)
        action_dist, *_ = self.actor_layer(action_feature)

        if pretanh_actions:
            log_probs = action_dist.log_probs(actions, pretanh_actions)
        else:
            log_probs = action_dist.log_probs(actions)

        entropy = action_dist.entropy().mean()

        return {'log_probs': log_probs, 'entropy': entropy}
