from __future__ import annotations

from itertools import count
from operator import itemgetter
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import torch

from mbpo_pytorch.misc import logger

if TYPE_CHECKING:
    from mbpo_pytorch.storages import SimpleUniversalBuffer as Buffer
    from mbpo_pytorch.models.dynamics import BaseDynamics
    from mbpo_pytorch.envs.virtual_env import VecVirtualEnv


# noinspection DuplicatedCode
def split_model_buffer(buffer: Buffer, ratio: float):
    full_indices = np.arange(buffer.size)
    np.random.shuffle(full_indices)
    train_indices = full_indices[:int(ratio * buffer.size)]
    val_indices = full_indices[int(ratio * buffer.size):]
    return train_indices, val_indices


class MBPO:
    def __init__(self, dynamics: BaseDynamics, batch_size: int, max_num_epochs: int,
                 rollout_schedule: List[int], l2_loss_coefs: List[float], lr, max_grad_norm=2, verbose=0):
        self.dynamics = dynamics
        self.epoch = 0

        self.max_num_epochs = max_num_epochs
        self.num_rollout_steps = 0
        self.rollout_schedule = rollout_schedule
        self.batch_size = batch_size
        self.l2_loss_coefs = l2_loss_coefs
        self.max_grad_norm = max_grad_norm
        self.training_ratio = 0.8

        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr)
        self.elite_dynamics_indices = []
        self.verbose = verbose

    @staticmethod
    def check_buffer(buffer):
        assert {'states', 'actions', 'rewards', 'masks', 'next_states'}.issubset(buffer.entry_infos.keys())

    def compute_loss(self, samples: Dict[str, torch.Tensor], use_var_loss=True, use_l2_loss=True):
        states, actions, next_states, rewards, masks = \
            itemgetter('states', 'actions', 'next_states', 'rewards', 'masks')(samples)

        diff_state_means, diff_state_logvars, reward_means, reward_logvars = \
            itemgetter('diff_state_means', 'diff_state_logvars', 'reward_means', 'reward_logvars') \
                (self.dynamics.forward(states, actions))

        means, logvars = torch.cat([diff_state_means, reward_means], dim=-1), \
                         torch.cat([diff_state_logvars, reward_logvars], dim=-1)
        targets = torch.cat([next_states - states, rewards], dim=-1)
        targets, masks = targets.repeat(means.shape[0], 1, 1), masks.repeat(means.shape[0], 1, 1)

        inv_vars = torch.exp(-logvars)

        mse_losses = torch.mean(((means - targets) ** 2) * inv_vars * masks, dim=[-2, -1])
        if use_var_loss:
            var_losses = torch.mean(logvars * masks, dim=[-2, -1])
            model_losses = mse_losses + var_losses
        else:
            model_losses = mse_losses

        if use_l2_loss:
            l2_losses = self.dynamics.compute_l2_loss(self.l2_loss_coefs)
            return model_losses, l2_losses
        else:
            return model_losses, None

    def update(self, model_buffer: Buffer) -> Dict[str, float]:
        model_loss_epoch = 0.
        l2_loss_epoch = 0.

        if self.max_num_epochs:
            epoch_iter = range(self.max_num_epochs)
        else:
            epoch_iter = count()

        train_indices, val_indices = split_model_buffer(model_buffer, self.training_ratio)

        num_epoch_after_update = 0
        num_updates = 0

        self.dynamics.reset_best_snapshots()

        for epoch in epoch_iter:
            train_gen = model_buffer.get_batch_generator_epoch(self.batch_size, train_indices)
            val_gen = model_buffer.get_batch_generator_epoch(None, val_indices)

            for samples in train_gen:
                train_model_loss, train_l2_loss = self.compute_loss(samples, True, True)
                train_model_loss, train_l2_loss = train_model_loss.sum(), train_l2_loss.sum()
                train_model_loss += \
                    0.01 * (torch.sum(self.dynamics.max_state_logvar) + torch.sum(self.dynamics.max_reward_logvar) -
                            torch.sum(self.dynamics.min_state_logvar) - torch.sum(self.dynamics.min_reward_logvar))

                model_loss_epoch += train_model_loss.item()
                l2_loss_epoch += train_l2_loss.item()

                self.dynamics_optimizer.zero_grad()
                (train_l2_loss + train_model_loss).backward()
                self.dynamics_optimizer.step()

                num_updates += 1

            with torch.no_grad():
                val_model_loss, _ = self.compute_loss(next(val_gen), False, False)
            updated = self.dynamics.update_best_snapshots(val_model_loss, epoch)
            if updated:
                num_epoch_after_update = 0
            else:
                num_epoch_after_update += 1
            if num_epoch_after_update > 5:
                break

        model_loss_epoch /= num_updates
        l2_loss_epoch /= num_updates

        val_gen = model_buffer.get_batch_generator_epoch(None, val_indices)
        best_epochs = self.dynamics.load_best_snapshots()
        with torch.no_grad():
            val_model_loss, _ = self.compute_loss(next(val_gen), False, False)
        self.dynamics.update_elite_indices(val_model_loss)

        if self.verbose > 0:
            logger.log('[ Model Traning ] Converge at epoch {}'.format(epoch))
            logger.log('[ Model Traning ] Load best state_dict from epoch {}'.format(best_epochs))
            logger.log('[ Model Traning ] Validation Model loss of elite networks: {}'.
                       format(val_model_loss.cpu().numpy()[self.dynamics.elite_indices]))

        return {'model_loss': model_loss_epoch, 'l2_loss': l2_loss_epoch}

    def update_rollout_length(self, epoch: int):
        min_epoch, max_epoch, min_length, max_length = self.rollout_schedule
        if epoch <= min_epoch:
            y = min_length
        else:
            dx = (epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length
        y = int(y)
        if self.verbose > 0 and self.num_rollout_steps != y:
            logger.log('[ Model Rollout ] Max rollout length {} -> {} '.format(self.num_rollout_steps, y))
        self.num_rollout_steps = y

    def collect_data(self, virtual_envs: VecVirtualEnv, policy_buffer: Buffer, initial_states: torch.Tensor, actor):
        states = initial_states
        batch_size = initial_states.shape[0]
        num_total_samples = 0
        for step in range(self.num_rollout_steps):
            with torch.no_grad():
                actions = actor.act(states)['actions']
            next_states, rewards, dones, _ = virtual_envs.step_with_states(states, actions)
            masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float32)
            policy_buffer.insert(states=states, actions=actions, masks=masks, rewards=rewards,
                                 next_states=next_states)
            num_total_samples += next_states.shape[0]
            states = next_states[torch.where(masks > 0.5)[0], :]
            if states.shape[0] == 0:
                logger.warn('[ Model Rollout ] Breaking early: {}'.format(step))
                break
        if self.verbose:
            logger.log('[ Model Rollout ] {} samples with average rollout length {:.2f}'.
                       format(num_total_samples, num_total_samples / batch_size))
