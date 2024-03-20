#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

import abc
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.utils import CategoricalNet, Flatten, to_grid
from habitat_baselines.rl.models.implicit_projection import q_conj, q_prod
from habitat_baselines.rl.models.implicit_representation import (
    Identity,
    ReplayBuffer,
    MLP_sem_finder,
    run_epoch,
    target_id_map,
)
from habitat_baselines.rl.models.implicit_representation_exploration_occupancy import (
    MLP_expl_occ,
    MLPReader,
    reading_net_expl_occ_weights,
    run_epoch_expl_occ,
)
from habitat_baselines.rl.models.projection import Projection, RotateTensor, get_grid
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import (
    RGBCNNNonOracle,
    RGBCNNOracle,
    MapCNN,
    SegHead,
)
from habitat_baselines.rl.ppo.aux_losses_utils import (
    get_obj_poses,
    compute_distance_labels,
    compute_direction_labels,
)


class PolicyNonOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        deterministic=False,
        nb_steps=None,
        current_episodes=None,
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations,
            rnn_hidden_states,
            global_map,
            prev_actions,
            masks,
            nb_steps,
            current_episodes,
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, global_map

    def get_value(
        self, observations, rnn_hidden_states, global_map, prev_actions, masks
    ):
        features, _, _ = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, global_map, prev_actions, masks, action
    ):
        (
            features,
            rnn_hidden_states,
            global_map,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        ) = self.net(
            observations,
            rnn_hidden_states,
            global_map,
            prev_actions,
            masks,
            ev=1,
            aux_loss=True,
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        )


class PolicyOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        (
            features,
            rnn_hidden_states,
            saved_impl_queries_res,
            saved_map_latent_codes,
            replay_buffer_rgb,
            *_,
        ) = self.net(observations, rnn_hidden_states, prev_actions, masks)

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
            saved_impl_queries_res,
            saved_map_latent_codes,
            replay_buffer_rgb,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        if hasattr(self.net, "num_envs"):  # if dealing with implicit representations
            net_saved_impl_queries_res = []
            optim_saved_impl_queries_res = []
            for i in range(self.net.num_envs):
                net_saved_impl_queries_res.append(
                    copy.deepcopy(self.net.net_sem_finder_list[i].state_dict())
                )
                optim_saved_impl_queries_res.append(
                    copy.deepcopy(self.net.optimizer_sem_finder_list[i].state_dict())
                )

            if self.net.net_list_expl_occ is not None:
                net_saved_map_latent_codes = []
                optim_saved_map_latent_codes = []
                for i in range(self.net.num_envs):
                    net_saved_map_latent_codes.append(
                        copy.deepcopy(self.net.net_list_expl_occ[i].state_dict())
                    )
                    optim_saved_map_latent_codes.append(
                        copy.deepcopy(self.net.optimizer_list_expl_occ[i].state_dict())
                    )

        features, *_ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, get_value=True
        )

        if hasattr(self.net, "num_envs"):
            for i in range(self.net.num_envs):
                self.net.net_sem_finder_list[i].load_state_dict(
                    net_saved_impl_queries_res[i]
                )
                self.net.optimizer_sem_finder_list[i].load_state_dict(
                    optim_saved_impl_queries_res[i]
                )

            if self.net.net_list_expl_occ is not None:
                for i in range(self.net.num_envs):
                    self.net.net_list_expl_occ[i].load_state_dict(
                        net_saved_map_latent_codes[i]
                    )
                    self.net.optimizer_list_expl_occ[i].load_state_dict(
                        optim_saved_map_latent_codes[i]
                    )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        states_dict_batch,
        saved_map_latent_codes_batch,
    ):
        (
            features,
            rnn_hidden_states,
            _,
            _,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        ) = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            False,
            True,
            states_dict_batch,
            saved_map_latent_codes_batch,
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            loss_seen,
            loss_directions,
            loss_distances,
            pred_seen,
            seen_labels,
        )


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class BaselinePolicyNonOracle(PolicyNonOracle):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        coordinate_min,
        coordinate_max,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetNonOracle(
                batch_size,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                egocentric_map_size=egocentric_map_size,
                global_map_size=global_map_size,
                global_map_depth=global_map_depth,
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
                aux_loss_seen_coef=aux_loss_seen_coef,
                aux_loss_direction_coef=aux_loss_direction_coef,
                aux_loss_distance_coef=aux_loss_distance_coef,
            ),
            action_space.n,
        )


class BaselinePolicyOracle(PolicyOracle):
    def __init__(
        self,
        agent_type,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
        hidden_size=512,
        implicit_config=None,
    ):
        super().__init__(
            BaselineNetOracle(
                agent_type,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                aux_loss_seen_coef=aux_loss_seen_coef,
                aux_loss_direction_coef=aux_loss_direction_coef,
                aux_loss_distance_coef=aux_loss_distance_coef,
                implicit_config=implicit_config,
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, global_map, prev_actions):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class BaselineNetNonOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        batch_size,
        observation_space,
        hidden_size,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        coordinate_min,
        coordinate_max,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth

        self.visual_encoder = RGBCNNNonOracle(observation_space, hidden_size)
        self.map_encoder = MapCNN(51, 256, "non-oracle")

        self.projection = Projection(
            egocentric_map_size, global_map_size, device, coordinate_min, coordinate_max
        )

        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        self.image_features_linear = nn.Linear(32 * 28 * 28, 512)

        self.flatten = Flatten()

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size
                + 256
                + object_category_embedding_size
                + previous_action_embedding_size,
                self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (0 if self.is_blind else self._hidden_size)
                + object_category_embedding_size,
                self._hidden_size,
            )
        self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        self.full_global_map = torch.zeros(
            batch_size,
            global_map_size,
            global_map_size,
            global_map_depth,
            device=self.device,
        )

        # Auxiliary losses
        if aux_loss_seen_coef is not None:
            self.fc_seen = nn.Linear(self._hidden_size, 2)
            nn.init.orthogonal_(self.fc_seen.weight)
            nn.init.constant_(self.fc_seen.bias, 0)
        else:
            self.fc_seen = None

        if aux_loss_direction_coef is not None:
            self.fc_direction = nn.Linear(self._hidden_size, 12)
            nn.init.orthogonal_(self.fc_direction.weight)
            nn.init.constant_(self.fc_direction.bias, 0)
        else:
            self.fc_direction = None

        if aux_loss_distance_coef is not None:
            self.fc_distance = nn.Linear(self._hidden_size, 35)
            nn.init.orthogonal_(self.fc_distance.weight)
            nn.init.constant_(self.fc_distance.bias, 0)
        else:
            self.fc_distance = None

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        nb_steps=None,
        current_episodes=None,
        ev=0,
        aux_loss=False,
    ):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = self.goal_embedding(
            (target_encoding).type(torch.LongTensor).to(self.device)
        ).squeeze(1)

        loss_seen = None
        pred_softmax = None
        loss_directions = None
        loss_distances = None
        seen_labels = None

        if aux_loss and (
            (self.fc_seen is not None)
            or (self.fc_direction is not None)
            or (self.fc_distance is not None)
        ):
            # Get positions of target objs
            mean_i_obj, mean_j_obj, gt_seen, not_visible_goals = get_obj_poses(
                observations
            )

            # Compute euclidian distance
            if self.fc_distance is not None:
                distance_labels = compute_distance_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

            if self.fc_seen is not None:
                seen_labels = np.array(gt_seen).astype(np.int_)
                seen_labels = torch.from_numpy(seen_labels).to(self.device)

            # Compute directions
            if self.fc_direction is not None:
                direction_labels = compute_direction_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
        projection = self.projection.forward(
            perception_embed, observations["depth"] * 10, -(observations["compass"])
        )
        perception_embed = self.image_features_linear(self.flatten(perception_embed))
        grid_x, grid_y = self.to_grid.get_grid_coords(observations["gps"])
        bs = global_map.shape[0]
        ##forward pass specific
        if ev == 0:
            self.full_global_map[:bs, :, :, :] = self.full_global_map[
                :bs, :, :, :
            ] * masks.unsqueeze(1).unsqueeze(1)
            if bs != 18:
                self.full_global_map[bs:, :, :, :] = (
                    self.full_global_map[bs:, :, :, :] * 0
                )
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    agent_view = torch.cuda.FloatTensor(
                        bs,
                        self.global_map_depth,
                        self.global_map_size,
                        self.global_map_size,
                    ).fill_(0)
            else:
                agent_view = (
                    torch.FloatTensor(
                        bs,
                        self.global_map_depth,
                        self.global_map_size,
                        self.global_map_size,
                    )
                    .to(self.device)
                    .fill_(0)
                )
            agent_view[
                :,
                :,
                self.global_map_size // 2
                - math.floor(self.egocentric_map_size / 2) : self.global_map_size // 2
                + math.ceil(self.egocentric_map_size / 2),
                self.global_map_size // 2
                - math.floor(self.egocentric_map_size / 2) : self.global_map_size // 2
                + math.ceil(self.egocentric_map_size / 2),
            ] = projection
            st_pose = torch.cat(
                [
                    -(grid_y.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    -(grid_x.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    observations["compass"],
                ],
                dim=1,
            )

            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)

            self.full_global_map[:bs, :, :, :] = torch.max(
                self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1)
            )
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    (grid_x.unsqueeze(1) - (self.global_map_size // 2))
                    / (self.global_map_size // 2),
                    torch.zeros_like(observations["compass"]),
                ],
                dim=1,
            )
            _, trans_mat_retrieval = get_grid(
                st_pose_retrieval, agent_view.size(), self.device
            )
            translated_retrieval = F.grid_sample(
                self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2),
                trans_mat_retrieval,
            )
            translated_retrieval = translated_retrieval[
                :,
                :,
                self.global_map_size // 2
                - math.floor(51 / 2) : self.global_map_size // 2
                + math.ceil(51 / 2),
                self.global_map_size // 2
                - math.floor(51 / 2) : self.global_map_size // 2
                + math.ceil(51 / 2),
            ]
            final_retrieval = self.rotate_tensor.forward(
                translated_retrieval, observations["compass"]
            )

            global_map_embed = self.map_encoder(final_retrieval.permute(0, 2, 3, 1))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat(
                (perception_embed, global_map_embed, goal_embed, action_embedding),
                dim=1,
            )
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1)
        else:

            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)
            with torch.cuda.device(0):
                agent_view = torch.cuda.FloatTensor(
                    bs, self.global_map_depth, 51, 51
                ).fill_(0)
            agent_view[
                :,
                :,
                51 // 2
                - math.floor(self.egocentric_map_size / 2) : 51 // 2
                + math.ceil(self.egocentric_map_size / 2),
                51 // 2
                - math.floor(self.egocentric_map_size / 2) : 51 // 2
                + math.ceil(self.egocentric_map_size / 2),
            ] = projection

            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            global_map_embed = self.map_encoder(final_retrieval)

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat(
                (perception_embed, global_map_embed, goal_embed, action_embedding),
                dim=1,
            )
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

            # Compute auxiliary losses
            if aux_loss and (
                (self.fc_seen is not None)
                or (self.fc_direction is not None)
                or (self.fc_distance is not None)
            ):
                if self.fc_seen is not None:
                    pred_seen = self.fc_seen(x)
                    loss_seen = F.cross_entropy(pred_seen, seen_labels)
                    pred_softmax = F.softmax(pred_seen, dim=1)

                indices_to_keep = (seen_labels == 1).nonzero()[:, 0]
                if len(indices_to_keep) > 0:
                    if self.fc_direction is not None:
                        pred_directions = self.fc_direction(x)
                        loss_directions = F.cross_entropy(
                            pred_directions[indices_to_keep],
                            direction_labels[indices_to_keep],
                        )

                    if self.fc_distance is not None:
                        pred_distances = self.fc_distance(x)
                        loss_distances = F.cross_entropy(
                            pred_distances[indices_to_keep],
                            distance_labels[indices_to_keep],
                        )
                else:
                    if self.fc_direction is not None:
                        loss_directions = torch.zeros(1).squeeze(0).to(self.device)
                    if self.fc_distance is not None:
                        loss_distances = torch.zeros(1).squeeze(0).to(self.device)

            return (
                x,
                rnn_hidden_states,
                final_retrieval.permute(0, 2, 3, 1),
                loss_seen,
                loss_directions,
                loss_distances,
                pred_softmax,
                seen_labels,
            )


class BaselineNetOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        agent_type,
        observation_space,
        hidden_size,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        aux_loss_seen_coef,
        aux_loss_direction_coef,
        aux_loss_distance_coef,
        implicit_config,
    ):
        super().__init__()
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = RGBCNNOracle(observation_space, 512)
        if agent_type == "oracle":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        elif agent_type == "no-map":
            self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        elif agent_type == "oracle-ego":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.object_embedding = nn.Embedding(10, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        elif agent_type == "implicit":
            self.goal_embedding = nn.Embedding(
                8, object_category_embedding_size
            )  # Same as no-map

            # Modifying policy visual encoder
            self.image_features_linear = nn.Linear(32 * 28 * 28, 512)
            self.flatten = Flatten()
            self.visual_encoder.cnn[5] = nn.ReLU()
            self.visual_encoder.cnn[6] = Identity()
            self.visual_encoder.cnn[7] = Identity()

            # Semantic Finder (sem implicit representation)
            self.learn_sem_finder = implicit_config["learn_sem_finder"]
            if self.learn_sem_finder:
                # Loading visual encoder and segmentation head used only for the Semantic Finder
                #### Visual encoder
                self.sem_finder_visual_encoder = RGBCNNOracle(
                    observation_space, 512
                ).to(device)
                self.sem_finder_visual_encoder.cnn[5] = nn.ReLU()
                self.sem_finder_visual_encoder.cnn[6] = Identity()
                self.sem_finder_visual_encoder.cnn[7] = Identity()
                sem_finder_visual_encoder_path = implicit_config[
                    "sem_finder_visual_encoder_path"
                ]
                self.sem_finder_visual_encoder.load_state_dict(
                    torch.load(sem_finder_visual_encoder_path, map_location="cpu")[
                        "expe_visual_encoder_state_dict"
                    ]
                )
                for param in self.sem_finder_visual_encoder.parameters():
                    param.requires_grad = False

                #### Segmentation head
                self.sem_finder_seg_head = SegHead(additional_convs=True)
                seg_path = implicit_config["sem_finder_seg_path"]
                self.sem_finder_seg_head.load_state_dict(
                    torch.load(seg_path, map_location="cpu")["model_state_dict"]
                )
                for param in self.sem_finder_seg_head.parameters():
                    param.requires_grad = False

                # Hyperparameters - Semantic Finder training
                self.batch_size = implicit_config["batch_size"]
                self.batches = implicit_config["batches"]
                self.num_envs = implicit_config["num_envs"]

            # Exploration and Occupancy Implicit Representation
            self.learn_impl_net_expl_occ = implicit_config["learn_impl_net_expl_occ"]
            if self.learn_impl_net_expl_occ:
                assert self.learn_sem_finder
                self.expl_occ_loss_threshold = implicit_config[
                    "expl_occ_loss_threshold"
                ]

                # Used to compute episode relative heading
                self.init_rot_egomap = torch.zeros((self.num_envs, 1)).to(device)

                self.net_list_expl_occ = []
                self.optimizer_list_expl_occ = []
                for _ in range(self.num_envs):
                    net_sem_finder_expl_occ = MLP_expl_occ(
                        D=3, W=512, input_ch=40, output_ch=3
                    ).to(device)
                    optimizer_sem_finder_expl_occ = torch.optim.Adam(
                        net_sem_finder_expl_occ.parameters(), lr=1e-3
                    )
                    self.net_list_expl_occ.append(net_sem_finder_expl_occ)
                    self.optimizer_list_expl_occ.append(optimizer_sem_finder_expl_occ)

                # Loss function
                self.criterion_expl_occ = nn.CrossEntropyLoss(
                    weight=torch.Tensor([2, 1, 2]).to(device)
                )

                # MLP Reader
                self.fc_reader = nn.Sequential(nn.Linear(576, 256), nn.ReLU())
                self.global_reader = MLPReader(device=device).to(device)
                global_reader_state_dict = torch.load(
                    implicit_config["global_reader_path"]
                )

                # Removing decoder (used only in Global Reader pre-training)
                keys_to_remove = []
                for k in global_reader_state_dict.keys():
                    if "decoder_conv" in k:
                        keys_to_remove.append(k)
                for k in keys_to_remove:
                    del global_reader_state_dict[k]

                self.global_reader.load_state_dict(global_reader_state_dict)
                for param in self.global_reader.parameters():
                    param.requires_grad = False
                self.global_reader.eval()
            else:
                self.net_list_expl_occ = None

            if self.learn_sem_finder:
                self.replay_buffer = ReplayBuffer(
                    self.batch_size,
                    self.num_envs,
                    self.device,
                    emb_dim=9,
                    learn_impl_net_expl_occ=self.learn_impl_net_expl_occ,
                )

                self.net_sem_finder_list = []
                self.optimizer_sem_finder_list = []
                for _ in range(self.num_envs):
                    net_sem_finder = MLP_sem_finder(
                        D=3, W=512, input_ch=9, output_ch=3
                    ).to(self.device)
                    optimizer_sem_finder = torch.optim.Adam(
                        net_sem_finder.parameters(), lr=1e-3
                    )

                    self.net_sem_finder_list.append(net_sem_finder)
                    self.optimizer_sem_finder_list.append(optimizer_sem_finder)

                # Loss function
                self.criterion_l1 = nn.L1Loss()

                # Projecting coordinates into a 32-dim vector: (x,y,z) -> 32-dim vector
                self.x_embedding = nn.Linear(3, 32)

                # Evaluate
                num_mini_batches = 1
                self.net_sem_finder_list_eval = [
                    [] for _ in range(self.num_envs // num_mini_batches)
                ]
                for i in range(len(self.net_sem_finder_list_eval)):
                    for _ in range(128):
                        self.net_sem_finder_list_eval[i].append(
                            MLP_sem_finder(D=3, W=512, input_ch=9, output_ch=3).to(
                                self.device
                            )
                        )
            else:
                self.net_sem_finder_list = None

        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        if agent_type == "implicit":
            semantic_finder_pred_size = 32
            expl_occ_map_latent_code_size = 576
        else:
            semantic_finder_pred_size = 0
            expl_occ_map_latent_code_size = 0

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size)
                + object_category_embedding_size
                + semantic_finder_pred_size
                + expl_occ_map_latent_code_size
                + previous_action_embedding_size,
                self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size)
                + object_category_embedding_size
                + semantic_finder_pred_size
                + expl_occ_map_latent_code_size,
                self._hidden_size,
            )

        # Auxiliary losses
        if aux_loss_seen_coef is not None:
            self.fc_seen = nn.Linear(self._hidden_size, 2)
            nn.init.orthogonal_(self.fc_seen.weight)
            nn.init.constant_(self.fc_seen.bias, 0)
        else:
            self.fc_seen = None

        if aux_loss_direction_coef is not None:
            self.fc_direction = nn.Linear(self._hidden_size, 12)
            nn.init.orthogonal_(self.fc_direction.weight)
            nn.init.constant_(self.fc_direction.bias, 0)
        else:
            self.fc_direction = None

        if aux_loss_distance_coef is not None:
            self.fc_distance = nn.Linear(self._hidden_size, 35)
            nn.init.orthogonal_(self.fc_distance.weight)
            nn.init.constant_(self.fc_distance.bias, 0)
        else:
            self.fc_distance = None

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        get_value=False,
        aux_loss=False,
        saved_impl_queries_res_batch=None,
        saved_map_latent_codes_batch=None,
    ):
        if not (
            self.agent_type == "implicit" and saved_impl_queries_res_batch is not None
        ):
            target_encoding = self.get_target_encoding(observations)
            x = [
                self.goal_embedding(
                    (target_encoding).type(torch.LongTensor).to(self.device)
                ).squeeze(1)
            ]
            bs = target_encoding.shape[0]

            if not self.is_blind:
                if (
                    self.agent_type == "implicit"
                    and self.learn_sem_finder
                    and saved_impl_queries_res_batch is None
                ):
                    # Encoding observation to add it to the replay buffer
                    # of the Semantic Finder
                    sample_to_add = self.sem_finder_visual_encoder(observations)
                    _, sample_to_add = self.sem_finder_seg_head(sample_to_add)
                    sample_to_add = F.softmax(sample_to_add, dim=1)
                    sample_to_add = sample_to_add.detach()

                # Policy visual input
                perception_embed = self.visual_encoder(observations)
                perception_embed = self.image_features_linear(
                    self.flatten(perception_embed)
                )
                x = [perception_embed] + x

        if (
            self.agent_type == "implicit"
            and self.learn_sem_finder
            and saved_impl_queries_res_batch is None
        ):  # act or get_value
            for idx in range(masks.shape[0]):
                if masks[idx] == 0:
                    # Resetting implicit representations and
                    # associated optimizers
                    self.net_sem_finder_list[idx].reset_parameters()
                    self.optimizer_sem_finder_list[idx] = torch.optim.Adam(
                        self.net_sem_finder_list[idx].parameters(), lr=1e-3
                    )

            # Adding data to the replay buffer of the Semantic Finder
            depth = observations["depth"]
            gps = observations["gps"]
            heading = observations["heading"]
            if get_value == False:
                self.replay_buffer.add_sample(sample_to_add, depth, gps, heading, masks)

            # Training the Semantic Finder
            for _ in range(self.batches):
                if get_value == False:
                    dt = self.replay_buffer.get_batch()
                else:
                    dt = self.replay_buffer.get_batch_get_value(
                        sample_to_add, depth, gps, heading, masks
                    )

                features = dt["rgb"]
                xyz = dt["pos"]
                features = features.unsqueeze(3).repeat(1, 1, 1, xyz.shape[3], 1)
                with torch.enable_grad():
                    run_epoch(
                        criterion_l1=self.criterion_l1,
                        net_sem_finder=self.net_sem_finder_list,
                        optimizer_sem_finder=self.optimizer_sem_finder_list,
                        features=features,
                        xyz=xyz,
                    )

            # Exploration and Occupancy Implicit Representation
            if self.learn_impl_net_expl_occ:
                for idx in range(masks.shape[0]):
                    if masks[idx] == 0:
                        self.net_list_expl_occ[idx].reset_parameters()
                        self.optimizer_list_expl_occ[idx] = torch.optim.Adam(
                            self.net_list_expl_occ[idx].parameters(), lr=1e-3
                        )
                        self.init_rot_egomap[idx] = observations["heading"][idx]

                to_train = np.ones(self.num_envs)
                nb_batches = 0
                while nb_batches < 20 and to_train.sum() > 0:
                    nb_batches += 1
                    # Get a batch
                    if get_value == False:
                        dt = self.replay_buffer.get_batch_expl_occ()
                    else:
                        dt = self.replay_buffer.get_batch_expl_occ_get_value(
                            depth, gps, heading, masks
                        )

                    xyz = dt["pos"].to(self.device)
                    labels = dt["labels"].to(self.device)

                    # Training implicit representation
                    with torch.enable_grad():
                        losses = run_epoch_expl_occ(
                            criterion=self.criterion_expl_occ,
                            net_expl_occ=self.net_list_expl_occ,
                            optimizer_expl_occ=self.optimizer_list_expl_occ,
                            xyz=xyz,
                            labels=labels,
                            to_train=to_train,
                        )
                        to_train = np.array(losses) > self.expl_occ_loss_threshold

                # Reading net_sem_finder weights
                w_1_batch, w_2_batch, w_3_batch = reading_net_expl_occ_weights(
                    self.net_list_expl_occ, self.device
                )

                # Computing angle class
                curr_heading = self.init_rot_egomap - observations["heading"]
                for idx in range(self.num_envs):
                    curr_heading_ = curr_heading[idx].item()
                    curr_heading_ = (180 - math.degrees(curr_heading_)) % 360
                    curr_heading_ = int(round(curr_heading_ / 30.0) * 30.0)
                    assert curr_heading_ % 30 == 0
                    if curr_heading_ == 360:
                        curr_heading_ = 0
                    curr_heading[idx] = curr_heading_ // 30
                curr_heading = curr_heading.type(torch.LongTensor).to(self.device)

                # Global reader prediction
                gps = observations["gps"]
                map_latent_codes = self.global_reader(
                    w_1=w_1_batch,
                    w_2=w_2_batch,
                    w_3=w_3_batch,
                    gps=gps,
                    curr_heading=curr_heading,
                )
                map_latent_codes = self.fc_reader(map_latent_codes)
                map_latent_codes = torch.cat(
                    (
                        map_latent_codes,
                        torch.zeros(
                            map_latent_codes.shape[0], 576 - map_latent_codes.shape[1]
                        ).to(self.device),
                    ),
                    dim=1,
                )
                x = [map_latent_codes] + x

                # Saving the implicit net latent code
                saved_map_latent_codes = []
                for i in range(self.num_envs):
                    saved_map_latent_codes.append(
                        copy.deepcopy(map_latent_codes[i].clone().detach())
                    )
            else:
                # Adding vector of zeroes to x
                map_latent_codes = torch.zeros((self.num_envs, 576)).to(self.device)
                x = [map_latent_codes] + x

                # Saving map latent code
                saved_map_latent_codes = []
                for i in range(self.num_envs):
                    saved_map_latent_codes.append(
                        copy.deepcopy(map_latent_codes[i].clone().detach())
                    )

            query_implicit_rep = torch.zeros((1, bs, 9)).to(self.device)
            for i in range(bs):
                tgt_enc = target_encoding[i].item()
                query_implicit_rep[0, i, target_id_map[tgt_enc]] = 1

            # Euclidian distance similarity
            if not get_value:
                replay_buffer_rgb = []

            uncertainty_euclidian_dist = []
            closest_vectors_in_replay_buffer = []
            for i in range(self.num_envs):
                rgb_curr, nb_samples_curr = (
                    self.replay_buffer.rgb[i],
                    self.replay_buffer.nb_samples_in_ep[i].item(),
                )
                rgb_curr = rgb_curr[: int(nb_samples_curr)]
                rgb_curr = rgb_curr.view(-1, rgb_curr.shape[2])
                if len(rgb_curr) == 0:
                    assert get_value is True
                    rgb_curr = sample_to_add[i]
                    rgb_curr = rgb_curr.permute((1, 2, 0))
                    rgb_curr = rgb_curr.view(-1, rgb_curr.shape[-1])
                res = (rgb_curr - query_implicit_rep[:, i]).pow(2).sum(dim=-1).sqrt()
                uncertainty_euclidian_dist.append(torch.min(res).item())
                if not get_value:
                    replay_buffer_rgb.append(rgb_curr)
                closest_vector_in_replay_buffer = rgb_curr[torch.argmin(res).item()]
                closest_vectors_in_replay_buffer.append(closest_vector_in_replay_buffer)

            # Semantic Finder prediction
            impl_queries_res = []
            for i in range(self.num_envs):
                out = self.net_sem_finder_list[i](query_implicit_rep[:, i])
                impl_queries_res.append(out)
            impl_queries_res = torch.cat(impl_queries_res, dim=0)

            # Converting Semantic Finder prediction to PointGoal format
            impl_queries_res[:, 0] *= (
                self.replay_buffer.max_x - self.replay_buffer.min_x
            )
            impl_queries_res[:, 0] += self.replay_buffer.min_x
            impl_queries_res[:, 1] *= (
                self.replay_buffer.max_y - self.replay_buffer.min_y
            )
            impl_queries_res[:, 1] += self.replay_buffer.min_y
            impl_queries_res[:, 2] *= (
                self.replay_buffer.max_z - self.replay_buffer.min_z
            )
            impl_queries_res[:, 2] += self.replay_buffer.min_z

            agent_position = torch.cat(
                (
                    observations["gps"][:, 1].unsqueeze(-1),
                    torch.zeros_like(observations["gps"][:, 1].unsqueeze(-1)),
                    -observations["gps"][:, 0].unsqueeze(-1),
                ),
                dim=1,
            )
            compass = observations["compass"].squeeze(-1)
            rotation_world_agent = torch.stack(
                (
                    torch.cos(0.5 * compass),
                    torch.zeros_like(compass),
                    torch.sin(0.5 * compass),
                    torch.zeros_like(compass),
                ),
                -1,
            ).to(self.device)
            dist_query_agent = torch.cat(
                (
                    torch.zeros(impl_queries_res.shape[0], 1).to(self.device),
                    impl_queries_res - agent_position,
                ),
                dim=1,
            )

            agent_to_queries = q_prod(q_conj(rotation_world_agent), dist_query_agent)
            agent_to_queries = q_prod(agent_to_queries, rotation_world_agent)[:, 1:]

            impl_queries_res = torch.cat(
                (
                    -agent_to_queries[:, 2].unsqueeze(-1),
                    agent_to_queries[:, 0].unsqueeze(-1),
                ),
                dim=1,
            )

            impl_queries_res = torch.cat(
                (
                    impl_queries_res,
                    torch.zeros(impl_queries_res.shape[0], 1).to(self.device),
                ),
                dim=1,
            )

            for i in range(impl_queries_res.shape[0]):
                impl_queries_res[i, 2] = uncertainty_euclidian_dist[i]

            saved_impl_queries_res = []
            for i in range(self.num_envs):
                saved_impl_queries_res.append(
                    copy.deepcopy(impl_queries_res[i].clone().detach())
                )

            impl_queries_res = self.x_embedding(impl_queries_res)

            # Adding implicit representation prediction as another input to the agent
            x = [impl_queries_res] + x
        elif (
            self.agent_type == "implicit"
            and not self.learn_sem_finder
            and saved_impl_queries_res_batch is None
        ):
            # Adding vector of zeroes to x
            bs = x[0].shape[0]
            map_latent_codes = torch.zeros((bs, 576)).to(self.device)
            x = [map_latent_codes] + x

            # Saving map latent code
            saved_map_latent_codes = []
            for i in range(bs):
                saved_map_latent_codes.append(
                    copy.deepcopy(map_latent_codes[i].clone().detach())
                )

            impl_queries_res = torch.zeros((bs, 32)).to(self.device)
            x = [impl_queries_res] + x

            saved_impl_queries_res = []
            for i in range(bs):
                saved_impl_queries_res.append(
                    copy.deepcopy(impl_queries_res[i].clone().detach())
                )

        loss_seen = None
        pred_softmax = None
        loss_directions = None
        loss_distances = None
        seen_labels = None

        if aux_loss and (
            (self.fc_seen is not None)
            or (self.fc_direction is not None)
            or (self.fc_distance is not None)
        ):
            # Get positions of target objs
            mean_i_obj, mean_j_obj, gt_seen, not_visible_goals = get_obj_poses(
                observations
            )

            # Compute euclidian distance
            if self.fc_distance is not None:
                distance_labels = compute_distance_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

            if self.fc_seen is not None:
                seen_labels = np.array(gt_seen).astype(np.int_)
                seen_labels = torch.from_numpy(seen_labels).to(self.device)

            # Compute directions
            if self.fc_direction is not None:
                direction_labels = compute_direction_labels(
                    mean_i_obj, mean_j_obj, not_visible_goals, self.device
                )

        if self.agent_type != "no-map" and self.agent_type != "implicit":
            global_map_embedding = []
            global_map = observations["semMap"]
            if self.agent_type == "oracle":
                global_map_embedding.append(
                    self.occupancy_embedding(
                        global_map[:, :, :, 0]
                        .type(torch.LongTensor)
                        .to(self.device)
                        .view(-1)
                    ).view(bs, 50, 50, -1)
                )

            global_map_embedding.append(
                self.object_embedding(
                    global_map[:, :, :, 1]
                    .type(torch.LongTensor)
                    .to(self.device)
                    .view(-1)
                ).view(bs, 50, 50, -1)
            )
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x

        if self.agent_type != "implicit" or (
            self.agent_type == "implicit" and saved_impl_queries_res_batch is None
        ):
            if self.use_previous_action:
                x = torch.cat(
                    x + [self.action_embedding(prev_actions).squeeze(1)], dim=1
                )
            else:
                x = torch.cat(x, dim=1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        if (
            self.agent_type == "implicit" and saved_impl_queries_res_batch is not None
        ):  # evaluate
            masks = masks.detach()
            rnn_hidden_states = rnn_hidden_states.detach()
            prev_actions = prev_actions.detach()
            for k in observations.keys():
                observations[k] = observations[k].detach()

            target_encoding = self.get_target_encoding(observations)
            x = [
                self.goal_embedding(
                    (target_encoding).type(torch.LongTensor).to(self.device)
                ).squeeze(1)
            ]
            if not self.is_blind:
                perception_embed = self.visual_encoder(observations)
                perception_embed = self.image_features_linear(
                    self.flatten(perception_embed)
                )
                x = [perception_embed] + x

            map_latent_codes = []
            for saved_codes in saved_map_latent_codes_batch:
                curr_map_latent_codes = []
                for saved_code in saved_codes:
                    curr_map_latent_codes.append(saved_code.unsqueeze(0))
                curr_map_latent_codes = torch.cat(curr_map_latent_codes, dim=0)
                map_latent_codes.append(curr_map_latent_codes.unsqueeze(0))
            map_latent_codes = torch.cat(map_latent_codes, dim=0)
            map_latent_codes = map_latent_codes.permute((1, 0, 2))
            map_latent_codes = map_latent_codes.reshape(-1, map_latent_codes.shape[-1])
            x = [map_latent_codes] + x

            impl_queries_res = []
            for saved_queries_res in saved_impl_queries_res_batch:
                curr_impl_queries_res = []
                for saved_query_res in saved_queries_res:
                    curr_impl_queries_res.append(saved_query_res.unsqueeze(0))
                curr_impl_queries_res = torch.cat(curr_impl_queries_res, dim=0)
                impl_queries_res.append(curr_impl_queries_res.unsqueeze(0))
            impl_queries_res = torch.cat(impl_queries_res, dim=0)
            impl_queries_res = impl_queries_res.permute((1, 0, 2))
            impl_queries_res = impl_queries_res.reshape(-1, impl_queries_res.shape[-1])

            if self.learn_sem_finder:
                impl_queries_res = self.x_embedding(impl_queries_res)

            x = [impl_queries_res] + x

            if self.use_previous_action:
                x = torch.cat(
                    x + [self.action_embedding(prev_actions).squeeze(1)], dim=1
                )
            else:
                x = torch.cat(x, dim=1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

            # No saving when evaluating
            saved_impl_queries_res = None
            saved_map_latent_codes = None

        # Compute auxiliary losses
        if aux_loss and (
            (self.fc_seen is not None)
            or (self.fc_direction is not None)
            or (self.fc_distance is not None)
        ):
            if self.fc_seen is not None:
                pred_seen = self.fc_seen(x)
                loss_seen = F.cross_entropy(pred_seen, seen_labels)
                pred_softmax = F.softmax(pred_seen, dim=1)

            indices_to_keep = (seen_labels == 1).nonzero()[:, 0]
            if len(indices_to_keep) > 0:
                if self.fc_direction is not None:
                    pred_directions = self.fc_direction(x)
                    loss_directions = F.cross_entropy(
                        pred_directions[indices_to_keep],
                        direction_labels[indices_to_keep],
                    )

                if self.fc_distance is not None:
                    pred_distances = self.fc_distance(x)
                    loss_distances = F.cross_entropy(
                        pred_distances[indices_to_keep],
                        distance_labels[indices_to_keep],
                    )
            else:
                if self.fc_direction is not None:
                    loss_directions = torch.zeros(1).squeeze(0).to(self.device)
                if self.fc_distance is not None:
                    loss_distances = torch.zeros(1).squeeze(0).to(self.device)

        return (
            x,
            rnn_hidden_states,
            saved_impl_queries_res,  # For implicit agent only
            saved_map_latent_codes,  # For implicit agent only
            loss_seen,
            loss_directions,
            loss_distances,
            pred_softmax,
            seen_labels,
        )
