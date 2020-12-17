import time
from collections import deque

from mbpo_pytorch.algos import SAC
from mbpo_pytorch.configs.config import Config
from mbpo_pytorch.misc.utils import *
from mbpo_pytorch.models import Actor, QCritic
from mbpo_pytorch.storages import SimpleUniversalBuffer as Buffer


# noinspection DuplicatedCode
def main():
    config, hparam_dict = Config('sac.yaml')
    set_seed(config.seed)

    writer, log_dir, eval_log_dir, save_dir = init_logging(config, hparam_dict)

    mf_config = config.sac

    torch.set_num_threads(1)
    device = torch.device(config.device)

    envs = make_vec_envs(config.env.env_name, get_seed(), config.env.num_envs, config.env.gamma, log_dir, device,
                         allow_early_resets=True, norm_reward=False, norm_obs=False)

    state_dim = envs.observation_space.shape[0]
    action_space = envs.action_space
    action_dim = action_space.shape[0]

    datatype = {'states': {'dims': [state_dim]}, 'next_states': {'dims': [state_dim]},
                'actions': {'dims': [action_dim]}, 'rewards': {'dims': [1]}, 'masks': {'dims': [1]}}

    actor = Actor(state_dim, action_space, mf_config.actor_hidden_dims, None, False, True, True)
    actor.to(device)

    q_critic1 = QCritic(state_dim, action_space, mf_config.critic_hidden_dims)
    q_critic2 = QCritic(state_dim, action_space, mf_config.critic_hidden_dims)
    q_critic_target1 = QCritic(state_dim, action_space, mf_config.critic_hidden_dims)
    q_critic_target2 = QCritic(state_dim, action_space, mf_config.critic_hidden_dims)
    q_critic1.to(device)
    q_critic2.to(device)
    q_critic_target1.to(device)
    q_critic_target2.to(device)

    target_entropy = mf_config.target_entropy or -np.prod(envs.action_space.shape).item()

    agent = SAC(actor, q_critic1, q_critic2, q_critic_target1, q_critic_target2, mf_config.batch_size,
                mf_config.num_grad_steps, config.env.gamma, 1.0,mf_config.actor_lr, mf_config.critic_lr,
                mf_config.soft_target_tau, target_entropy=target_entropy)

    off_policy_buffer = Buffer(mf_config.buffer_size, datatype)
    off_policy_buffer.to(device)

    episode_rewards = deque(maxlen=10)
    episode_lengths = deque(maxlen=10)

    start = time.time()
    num_updates = mf_config.num_total_steps // mf_config.num_epoch_steps // config.env.num_envs

    rewards, lengths = collect_traj(actor, envs, None, off_policy_buffer, mf_config.num_warmup_steps)
    episode_rewards.extend(rewards)
    episode_lengths.extend(lengths)

    for j in range(num_updates):

        rewards, lengths = collect_traj(actor, envs, None, off_policy_buffer, mf_config.num_epoch_steps)
        episode_rewards.extend(rewards)
        episode_lengths.extend(lengths)

        losses = agent.update(off_policy_buffer)

        serial_timsteps = (j + 1) * mf_config.num_epoch_steps + mf_config.num_warmup_steps
        total_num_steps = config.env.num_envs * (j + 1) * mf_config.num_epoch_steps + mf_config.num_warmup_steps
        end = time.time()

        fps = int(total_num_steps / (end - start))

        if j % config.log_interval == 0 and len(episode_rewards) > 0:
            log_info = [('serial_timesteps', serial_timsteps), ('total_timesteps', total_num_steps),
                        ('ep_rew_mean', np.mean(episode_rewards)), ('ep_len_mean', np.mean(episode_lengths)),
                        ('fps', fps), ('time_elapsed', end - start)]

            for loss_name, loss_value in losses.items():
                log_info.append((loss_name, loss_value))
            log_and_write(logger, writer, log_info, global_step=j)

        if (config.eval_interval is not None and len(episode_rewards) > 0
                and j % config.eval_interval == 0):
            episode_rewards_eval, episode_lengths_eval = \
                evaluate(actor, config.env.env_name, get_seed(), num_episode=10, eval_log_dir=eval_log_dir,
                         device=device, norm_reward=False, norm_obs=False)
            log_info = [('ep_rew_mean_eval', np.mean(episode_rewards_eval)),
                        ('ep_len_mean_eval', np.mean(episode_lengths_eval))]
            log_and_write(logger, writer, log_info, global_step=j)


if __name__ == "__main__":
    main()
