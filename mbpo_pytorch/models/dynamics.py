from abc import ABC, abstractmethod
from operator import itemgetter
from typing import List, Dict, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import Normal

from .initializer import truncated_norm_init
from .utils import MLP, init


class BaseDynamics(nn.Module, ABC):
    @abstractmethod
    def predict(self, states, actions, **kwargs) -> Dict[str, torch.Tensor]:
        pass


class RDynamics(BaseDynamics, ABC):
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, hidden_dims: List[int], output_state_dim=None,
                 **kwargs):
        super(RDynamics, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.output_state_dim = output_state_dim or state_dim

        assert getattr(kwargs, 'last_activation', 'identity') == 'identity'
        self.diff_dynamics = MLP(state_dim + action_dim, output_state_dim
                                 + reward_dim, hidden_dims, activation='swish', **kwargs)

        def init_(m): init(m, truncated_norm_init, lambda x: nn.init.constant_(x, 0))
        self.diff_dynamics.init(init_, init_)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.diff_dynamics(x)
        diff_states = x[..., :self.output_state_dim]
        rewards = x[..., self.output_state_dim:]
        return {'diff_states': diff_states, 'rewards': rewards}

    def predict(self, states, actions, **kwargs):
        diff_states, rewards = itemgetter('diff_states', 'rewards')(self.forward(states, actions))
        return {'next_states': states + diff_states, 'rewards': rewards}

    def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
        weight_norms = []
        for name, weight in self.diff_dynamics.named_parameters():
            if "weight" in name:
                weight_norms.append(weight.norm(2))
        weight_norms = torch.stack(weight_norms, dim=0)
        weight_decay = (torch.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay


class EnsembleRDynamics(BaseDynamics, ABC):
    def __init__(self,  state_dim: int, action_dim: int, reward_dim: int, hidden_dims: List[int],
                 num_networks, num_elite_networks, state_normalizer=None, action_normalizer=None):
        super(EnsembleRDynamics, self).__init__()
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.num_networks = num_networks

        self.networks = nn.ModuleList([RDynamics(state_dim, action_dim, 2 * reward_dim, hidden_dims, 2 * state_dim)
                                       for _ in range(num_networks)])

        self.max_diff_state_logvar = nn.Parameter(torch.ones([1, state_dim]) / 2.)
        self.min_diff_state_logvar = nn.Parameter(-torch.ones([1, state_dim]) * 10.)
        self.max_reward_logvar = nn.Parameter(torch.ones([1, reward_dim]) / 2.)
        self.min_reward_logvar = nn.Parameter(-torch.ones([1, reward_dim]) * 10.)

        self.state_normalizer = state_normalizer or nn.Identity()
        self.action_normalizer = action_normalizer or nn.Identity()
        self.num_elite_networks = num_elite_networks
        self.elite_indices = None

        self.best_snapshots = [(None, 0, None) for _ in self.networks]

    def load(self, load_path):
        pass

    def forward(self, states, actions) -> Dict[str, torch.Tensor]:

        states, actions = self.state_normalizer(states), self.action_normalizer(actions)
        outputs = [network(states, actions) for network in self.networks]
        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}

        # diff_states, rewards is [num_networks, batch_size, *]
        diff_states, rewards = torch.stack(outputs['diff_states'], dim=0), torch.stack(outputs['rewards'], dim=0)
        factored_diff_state_means, factored_diff_state_logvars = \
            diff_states[..., :self.state_dim], diff_states[..., self.state_dim:]
        factored_reward_means, factored_reward_logvars = \
            rewards[..., :self.reward_dim], rewards[..., self.reward_dim:]

        factored_diff_state_logvars = self.max_diff_state_logvar - \
                                      softplus(self.max_diff_state_logvar - factored_diff_state_logvars)
        factored_diff_state_logvars = self.min_diff_state_logvar + \
                                      softplus(factored_diff_state_logvars - self.min_diff_state_logvar)
        factored_reward_logvars = self.max_reward_logvar - softplus(self.max_reward_logvar - factored_reward_logvars)
        factored_reward_logvars = self.min_reward_logvar + softplus(factored_reward_logvars - self.min_reward_logvar)

        # return num_ensemble * batch_size * dim
        return {'diff_state_means': factored_diff_state_means,
                'diff_state_logvars': factored_diff_state_logvars,
                'reward_means': factored_reward_means,
                'reward_logvars': factored_reward_logvars}

    def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]) -> torch.Tensor:
        l2_losses = [network.compute_l2_loss(l2_loss_coefs) for network in self.networks]
        return torch.stack(l2_losses, dim=0)

    def update_elite_indices(self, losses: torch.Tensor) -> np.ndarray:
        elite_indices = torch.argsort(losses)[:self.num_elite_networks].cpu().numpy()
        self.elite_indices = elite_indices.copy()
        return elite_indices

    def predict(self, states: torch.Tensor, actions: torch.Tensor, deterministic=False) -> Dict[str, torch.Tensor]:
        assert self.elite_indices is not None
        batch_size = states.shape[0]

        # only use elite networks for prediction
        indices = np.random.choice(self.elite_indices, batch_size)
        diff_state_means, diff_state_logvars, reward_means, reward_logvars = \
            itemgetter('diff_state_means', 'diff_state_logvars', 'reward_means', 'reward_logvars')\
                (self.forward(states, actions))

        diff_state_means, diff_state_logvars = diff_state_means[indices, np.arange(batch_size)], \
                                               diff_state_logvars[indices, np.arange(batch_size)]
        reward_means, reward_logvars = reward_means[indices, np.arange(batch_size)], \
                                       reward_logvars[indices, np.arange(batch_size)]

        if deterministic:
            next_state_means = states + diff_state_means
            return {'next_states': next_state_means, 'rewards': reward_means}
        else:
            diff_state_dists = Normal(diff_state_means, diff_state_logvars.exp().sqrt())
            rewards_dists = Normal(reward_means, reward_logvars.exp().sqrt())
            diff_states = diff_state_dists.sample()
            rewards = rewards_dists.sample()
            next_states = states + diff_states
            return {'next_states': next_states, 'rewards': rewards}

    def update_best_snapshots(self, losses: torch.Tensor, epoch: int) -> bool:
        updated = False
        for idx, (loss, snapshot) in enumerate(zip(losses, self.best_snapshots)):
            loss = loss.item()
            best_loss: Optional[float] = snapshot[0]
            improvement_ratio = ((best_loss - loss) / best_loss) if best_loss else 0.
            if (best_loss is None) or improvement_ratio > 0.01:
                # noinspection PyTypeChecker
                self.best_snapshots[idx] = (loss, epoch, self.networks[idx].state_dict())
                updated = True
        return updated

    def reset_best_snapshots(self) -> None:
        self.best_snapshots = [(None, 0, None) for _ in self.networks]

    def load_best_snapshots(self) -> List[int]:
        best_epochs = []
        for network, (_, epoch, state_dict) in zip(self.networks, self.best_snapshots):
            network.load_state_dict(state_dict)
            best_epochs.append(epoch)
        return best_epochs
