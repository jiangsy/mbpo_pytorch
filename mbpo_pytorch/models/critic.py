from abc import ABC
from typing import List

from gym.spaces import Box, MultiBinary, Discrete
import torch
import torch.nn as nn

from .initializer import normc_init, fanin_init
from .utils import MLP, init


class QCritic(nn.Module, ABC):
    def __init__(self, state_dim, action_space, hidden_dims: List[int], init_w=3e-3, init_b=0.1,
                 use_multihead_output=False, **kwargs):
        super(QCritic, self).__init__()

        assert not use_multihead_output or action_space.__class__.__name__ == 'Discrete'

        if isinstance(action_space, Box) or isinstance(action_space, MultiBinary):
            action_dim = action_space.shape[0]
        else:
            assert isinstance(action_space, Discrete)
            action_dim = action_space.n

        mlp_kwargs = kwargs.copy()
        mlp_kwargs['activation'] = kwargs.get('activation', 'ReLU')
        mlp_kwargs['last_activation'] = kwargs.get('last_activation', 'Identity')

        if use_multihead_output:
            action_dim = action_space.n
            self.critic = MLP(state_dim, action_dim, hidden_dims, **kwargs)
            self.forward = self._get_q_value_discrete
        else:
            self.critic = MLP(state_dim + action_dim, 1, hidden_dims, **kwargs)
            self.forward = self._get_q_value_continuous

        init_ = lambda m: init(m, fanin_init, lambda x: nn.init.constant_(x, init_b))
        init_last_ = lambda m: init(m, lambda x: nn.init.uniform_(x, -init_w, init_w),
                                       lambda x: nn.init.uniform_(x, -init_w, init_w))
        self.critic.init(init_, init_last_)

    def _get_q_value_continuous(self, state, action):
        return self.critic(torch.cat([state, action], dim=-1))

    def _get_q_value_discrete(self, state, action):
        return self.critic_feature(state)[action.type(torch.bool)].unsqueeze(dim=-1)


class VCritic(nn.Module, ABC):
    def __init__(self, dim_state: int, hidden_dims: List[int], state_normalizer=None, activation='Tanh'):
        super(VCritic, self).__init__()
        self.critic = MLP(dim_state, 1, hidden_dims, activation=activation)
        self.normalizer = state_normalizer or nn.Identity()

        init_ = lambda m: init(m, normc_init, lambda x: nn.init.constant_(x, 0))
        self.critic.init(init_, init_)

    def forward(self, state):
        state = self.normalizer(state)
        return self.critic(state)
