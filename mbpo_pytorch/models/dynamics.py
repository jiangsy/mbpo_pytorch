from abc import ABC, abstractmethod
from operator import itemgetter
from typing import List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .initializer import truncated_norm_init
from .normalizer import RunningNormalizer
from .utils import MLP, init


class BaseDynamics(nn.Module, ABC):
    @abstractmethod
    def predict(self, states, actions, **kwargs) -> Dict[str, torch.Tensor]:
        pass


class Dynamics(BaseDynamics, ABC):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int],
                 state_normalizer: RunningNormalizer = None, diff_normalizer: RunningNormalizer = None):
        super(Dynamics, self).__init__()
        self.dim_state = state_dim
        self.dim_action = action_dim
        self.state_normalizer = state_normalizer or nn.Identity()
        self.diff_normalizer = diff_normalizer or nn.Identity()
        self.diff_dynamics = MLP(state_dim + action_dim, state_dim, hidden_dims, activation='relu')

        init_ = lambda m: init(m, truncated_norm_init, lambda x: nn.init.constant_(x, 0))
        self.diff_dynamics.init(init_, init_)

    def forward(self, states, actions):
        # action clip is the best normalization according to the authors
        x = torch.cat([self.state_normalizer(states), actions.clamp(-1., 1.)], dim=-1)
        normalized_diff = self.diff_dynamics(x)
        next_states = states + self.diff_normalizer(normalized_diff, inverse=True)
        next_states = self.normalizer.state_normalizer(self.state_normalizer(next_states).clamp(-100, 100),
                                                       inverse=True)
        return next_states

    def predict(self, states, actions, **kwargs):
        next_states = self.forward(states, actions)
        return {'next_states': next_states}


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

        init_ = lambda m: init(m, truncated_norm_init, lambda x: nn.init.constant_(x, 0))
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

        self.max_state_logvar = nn.Parameter(torch.ones([1, state_dim]) / 2.)
        self.min_state_logvar = nn.Parameter(-torch.ones([1, state_dim]) * 10.)
        self.max_reward_logvar = nn.Parameter(torch.ones([1, reward_dim]) / 2.)
        self.min_reward_logvar = nn.Parameter(-torch.ones([1, reward_dim]) * 10.)

        self.state_normalizer = state_normalizer or nn.Identity()
        self.action_normalizer = action_normalizer or nn.Identity()
        self.num_elite_networks = num_elite_networks
        self.elite_indices = None

        def backward_hook_fn(x: nn.Module, grad_input, grad_output):
            x.elite_indices = None
            return None
        self.register_backward_hook(backward_hook_fn)

        self.best_snapshots = [(None, 0, None) for _ in self.networks]

    def load(self, load_path):
        pass

    def forward(self, states, actions, use_factored=False) -> Dict[str, torch.Tensor]:
        assert states.ndim == actions.ndim
        assert use_factored

        states, actions = self.state_normalizer(states), self.action_normalizer(actions)
        ndim = states.ndim
        if ndim == 3:
            assert states.shape[0] == self.num_networks
            outputs = [network(states_, actions_) for network, states_, actions_ in zip(self.networks, states, actions)]
        elif ndim == 2:
            outputs = [network(states, actions) for network in self.networks]
        else:
            assert False

        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}

        # diff_states, rewards is [num_networks, batch_size, *]
        diff_states, rewards = torch.stack(outputs['diff_states'], dim=0), torch.stack(outputs['rewards'], dim=0)
        farctored_diff_state_means, farctored_diff_state_logvars = \
            diff_states[..., :self.state_dim], diff_states[..., self.state_dim:]
        farctored_reward_means, farctored_reward_logvars = \
            rewards[..., :self.reward_dim], rewards[..., self.reward_dim:]

        farctored_diff_state_logvars = self.max_state_logvar -\
                                       F.softplus(self.max_state_logvar - farctored_diff_state_logvars)
        farctored_diff_state_logvars = self.min_state_logvar + \
                                       F.softplus(farctored_diff_state_logvars - self.min_state_logvar)
        farctored_reward_logvars = self.max_reward_logvar - F.softplus(self.max_reward_logvar - farctored_reward_logvars)
        farctored_reward_logvars = self.min_reward_logvar + F.softplus(farctored_reward_logvars - self.min_reward_logvar)

        if ndim == 2 and not use_factored:
            diff_state_means = torch.mean(farctored_diff_state_means, dim=0)
            diff_state_vars = torch.mean((farctored_diff_state_means - diff_state_means) ** 2, dim=0) + \
                              torch.mean(torch.exp(farctored_diff_state_logvars), dim=0)
            reward_means = torch.mean(farctored_reward_means, dim=0)
            reward_vars = torch.mean((farctored_reward_means - reward_means) ** 2, dim=0) + \
                          torch.mean(torch.exp(farctored_reward_logvars), dim=0)

            # return batch_size * dim
            return {'diff_states': diff_state_means,
                    'diff_state_means': diff_state_means,
                    'diff_state_logvars': diff_state_vars,
                    'rewards': reward_means,
                    'reward_means': reward_means,
                    'reward_logvars': reward_vars}

        # return num_ensemble * batch_size * dim
        return {'diff_state_means': farctored_diff_state_means,
                'diff_state_logvars': farctored_diff_state_logvars,
                'reward_means': farctored_reward_means,
                'reward_logvars': farctored_reward_logvars}

    def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]) -> torch.Tensor:
        l2_losses = [network.compute_l2_loss(l2_loss_coefs) for network in self.networks]
        return torch.stack(l2_losses, dim=0)

    def update_elite_indices(self, losses: torch.Tensor) -> np.ndarray:
        assert losses.ndim == 1 and losses.shape[0] == self.num_networks
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
                (self.forward(states, actions, True))

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

    def update_best_snapshots(self, losses, epoch) -> bool:
        # assert losses.ndim == 1 and losses.shape[0] == self.num_networks and torch.all(losses.isfinite()).item()
        updated = False
        for idx, (loss, snapshot) in enumerate(zip(losses, self.best_snapshots)):
            loss = loss.item()
            best_loss = snapshot[0]
            if best_loss is not None:
                improvement_ratio = (best_loss - loss) / best_loss
            if (best_loss is None) or improvement_ratio > 0.01:
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
