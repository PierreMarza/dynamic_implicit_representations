#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

import torch
import torch.nn as nn
import torch.optim as optim

EPS_PPO = 1e-5


class PPONonOracle(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        aux_loss_seen_coef=None,
        aux_loss_direction_coef=None,
        aux_loss_distance_coef=None,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Auxiliray loss coefficients
        self.aux_loss_seen_coef = aux_loss_seen_coef
        self.aux_loss_direction_coef = aux_loss_direction_coef
        self.aux_loss_distance_coef = aux_loss_distance_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        loss_seen_epoch = 0
        loss_directions_epoch = 0
        loss_distances_epoch = 0
        pred_seen_epoch = []
        seen_labels_epoch = []

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    global_map_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    loss_seen,
                    loss_directions,
                    loss_distances,
                    pred_seen,
                    seen_labels,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    global_map_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                # Augmenting vanilla PPO loss with auxiliray losses
                if self.aux_loss_seen_coef is not None:
                    total_loss += self.aux_loss_seen_coef * loss_seen
                else:
                    assert loss_seen is None

                if self.aux_loss_direction_coef is not None:
                    total_loss += self.aux_loss_direction_coef * loss_directions
                else:
                    assert loss_directions is None

                if self.aux_loss_distance_coef is not None:
                    total_loss += self.aux_loss_distance_coef * loss_distances
                else:
                    assert loss_distances is None

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                if loss_seen is not None:
                    loss_seen_epoch += loss_seen.item()
                if loss_directions is not None:
                    loss_directions_epoch += loss_directions.item()
                if loss_distances is not None:
                    loss_distances_epoch += loss_distances.item()
                if pred_seen is not None:
                    pred_seen_epoch.extend(pred_seen[:, 1].tolist())
                    seen_labels_epoch.extend(seen_labels.tolist())

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        loss_seen_epoch /= num_updates
        loss_directions_epoch /= num_updates
        loss_distances_epoch /= num_updates

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            loss_seen_epoch,
            loss_directions_epoch,
            loss_distances_epoch,
            pred_seen_epoch,
            seen_labels_epoch,
        )

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass


class PPOOracle(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        aux_loss_seen_coef=None,
        aux_loss_direction_coef=None,
        aux_loss_distance_coef=None,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.aux_loss_seen_coef = aux_loss_seen_coef
        self.aux_loss_direction_coef = aux_loss_direction_coef
        self.aux_loss_distance_coef = aux_loss_distance_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        loss_seen_epoch = 0
        loss_directions_epoch = 0
        loss_distances_epoch = 0
        pred_seen_epoch = []
        seen_labels_epoch = []

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    saved_impl_queries_res_batch,
                    saved_map_latent_codes_batch,
                ) = sample

                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    loss_seen,
                    loss_directions,
                    loss_distances,
                    pred_seen,
                    seen_labels,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                    saved_impl_queries_res_batch,
                    saved_map_latent_codes_batch,
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                if self.aux_loss_seen_coef is not None:
                    total_loss += self.aux_loss_seen_coef * loss_seen
                else:
                    assert loss_seen is None

                if self.aux_loss_direction_coef is not None:
                    total_loss += self.aux_loss_direction_coef * loss_directions
                else:
                    assert loss_directions is None

                if self.aux_loss_distance_coef is not None:
                    total_loss += self.aux_loss_distance_coef * loss_distances
                else:
                    assert loss_distances is None

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                if loss_seen is not None:
                    loss_seen_epoch += loss_seen.item()
                if loss_directions is not None:
                    loss_directions_epoch += loss_directions.item()
                if loss_distances is not None:
                    loss_distances_epoch += loss_distances.item()
                if pred_seen is not None:
                    pred_seen_epoch.extend(pred_seen[:, 1].tolist())
                    seen_labels_epoch.extend(seen_labels.tolist())

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        loss_seen_epoch /= num_updates
        loss_directions_epoch /= num_updates
        loss_distances_epoch /= num_updates

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            loss_seen_epoch,
            loss_directions_epoch,
            loss_distances_epoch,
            pred_seen_epoch,
            seen_labels_epoch,
        )

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass
