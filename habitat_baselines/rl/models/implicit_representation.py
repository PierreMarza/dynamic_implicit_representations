import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.models.implicit_projection import reproject

target_id_map = {
    6: 1,
    4: 2,
    0: 3,
    1: 4,
    2: 5,
    7: 6,
    3: 7,
    5: 8,
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# From https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=10, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": False,
        "input_dims": 2,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class MLP_sem_finder(nn.Module):
    def __init__(self, D=3, W=1000, input_ch=512, output_ch=3):
        super().__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D - 2)]
        )
        self.out = nn.Linear(W, output_ch)

    def forward(self, h):
        for i in range(len(self.pts_linears)):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        return F.sigmoid(self.out(h))

    def reset_parameters(self):
        for i in range(len(self.pts_linears)):
            self.pts_linears[i].reset_parameters()
        self.out.reset_parameters()


class ReplayBuffer:
    def __init__(
        self,
        batch_size,
        num_envs,
        device,
        emb_dim,
        learn_impl_net_expl_occ=False,
        min_xyz=-50,
        max_xyz=50,
        grid_size=28,
        nb_rgb_pos=2500,
        expl_kept_steps=1000,
        expl_kept_points=5000,
        expl_pos_emb_dim=40,
        expl_neg_r=2048,
        expl_neg_c=2048,
        expl_neg_nb_points_per_batch=64,
        expl_labels_thresh=0.001,
        expl_kept_pos_thresh=0.25,
    ):
        # Hyperparameters
        self.min_x, self.max_x = min_xyz, max_xyz
        self.min_y, self.max_y = min_xyz, max_xyz
        self.min_z, self.max_z = min_xyz, max_xyz
        self.emb_dim = emb_dim
        self.grid_size = grid_size
        self.rgb = torch.zeros(
            num_envs, nb_rgb_pos, self.grid_size * self.grid_size, self.emb_dim
        ).to(device)
        self.pos = torch.zeros(
            num_envs, nb_rgb_pos, self.grid_size * self.grid_size, 1, 3
        ).to(device)
        self.nb_samples_in_ep = torch.zeros(num_envs).to(device)
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.device = device
        self.init_rot = None

        # Average Pool Operator
        self.m = nn.AvgPool2d(256 // self.grid_size, stride=256 // self.grid_size).to(
            device
        )

        # Exploration and Occupancy Implicit Representation
        self.learn_impl_net_expl_occ = learn_impl_net_expl_occ
        if learn_impl_net_expl_occ:
            self.expl_kept_steps = expl_kept_steps
            self.expl_kept_points = expl_kept_points
            self.expl_pos_emb_dim = expl_pos_emb_dim
            self.pos_expl = torch.zeros(
                (num_envs, expl_kept_steps, expl_kept_points, expl_pos_emb_dim)
            ).to(device)
            self.labels_expl = torch.zeros(
                (num_envs, expl_kept_steps, expl_kept_points), dtype=torch.long
            ).to(device)
            self.nb_samples_in_ep_expl_occ = torch.zeros(num_envs).to(device)
            self.expl_labels_thresh = expl_labels_thresh
            self.expl_kept_pos_thresh = expl_kept_pos_thresh

            self.negatives = torch.zeros(expl_neg_r, expl_neg_c, 2).to(device)
            step = 1 / expl_neg_r
            for i in range(expl_neg_r):
                for j in range(expl_neg_c):
                    self.negatives[i, j, 0] = i * step
                    self.negatives[i, j, 1] = j * step
            self.negatives = self.negatives.view(-1, 2)

            self.embed_fn, _ = get_embedder()
            self.negatives = self.embed_fn(self.negatives)

            rand_idx = torch.randperm(self.negatives.shape[0])
            self.negatives = self.negatives[rand_idx]
            self.negatives = self.negatives.view(
                -1, self.batch_size, expl_neg_nb_points_per_batch, expl_pos_emb_dim
            )

            self.labels_negatives = torch.zeros(
                (self.batch_size, expl_neg_nb_points_per_batch), dtype=torch.long
            ).to(self.device)

            # to draw ego map
            self.init_rot_to_draw_egomap = torch.zeros(num_envs).to(self.device)
            self.agent_pos = torch.zeros(num_envs, 2).to(self.device)
            self.heading = torch.zeros(num_envs).to(self.device)

    def add_sample(self, input_rgb, input_depth, input_gps, input_heading, masks):
        rgb = input_rgb.permute((0, 2, 3, 1))

        # Re-initialize data for new episodes
        init_rot_to_reinit = []
        for i in range(len(self.rgb)):
            if masks[i] == 0:
                self.rgb[i, : int(self.nb_samples_in_ep[i].item())] = 0
                self.pos[i, : int(self.nb_samples_in_ep[i].item())] = 0
                init_rot_to_reinit.append(i)

                if self.learn_impl_net_expl_occ:
                    self.pos_expl[
                        i, : int(self.nb_samples_in_ep_expl_occ[i].item())
                    ] = 0
                    self.labels_expl[
                        i, : int(self.nb_samples_in_ep_expl_occ[i].item())
                    ] = 0

                    # to draw ego map
                    self.init_rot_to_draw_egomap[i] = input_heading[i].clone()

        self.nb_samples_in_ep *= masks[:, 0]

        if self.learn_impl_net_expl_occ:
            self.nb_samples_in_ep_expl_occ *= masks[:, 0]

        # Project depth pixels into 3D coordinates
        pos, self.init_rot = reproject(
            input_depth, input_gps, input_heading, self.init_rot, init_rot_to_reinit
        )
        self.pos_0 = pos[0]

        # Exploration/Occupancy
        if self.learn_impl_net_expl_occ:
            # to draw ego map
            self.agent_pos[:, 0] = input_gps[:, 1].clone()  # x coord
            self.agent_pos[:, 1] = -input_gps[:, 0].clone()  # z coord
            self.heading = input_heading.clone()

            labels = pos[:, :, :, 1] < self.expl_labels_thresh
            kept_pos = pos[:, :, :, 1] < self.expl_kept_pos_thresh

        # Normalize coordinates between 0 and 1
        pos[:, :, :, 0] = (pos[:, :, :, 0] - self.min_x) / (self.max_x - self.min_x)
        pos[:, :, :, 1] = (pos[:, :, :, 1] - self.min_y) / (self.max_y - self.min_y)
        pos[:, :, :, 2] = (pos[:, :, :, 2] - self.min_z) / (self.max_z - self.min_z)

        if self.learn_impl_net_expl_occ:
            pos_expl = pos.reshape((self.num_envs, -1, 3))
            pos_expl = torch.cat(
                [pos_expl[:, :, 0].unsqueeze(-1), pos_expl[:, :, 2].unsqueeze(-1)],
                dim=-1,
            )
            labels = labels.reshape((self.num_envs, -1))
            kept_pos = kept_pos.reshape((self.num_envs, -1))

            for i in range(self.num_envs):
                # Sampling positive samples
                positive_samples = (labels[i] == 1) * (kept_pos[i] == 1)
                labels_positive = labels[i][positive_samples]
                pos_positive = pos_expl[i][positive_samples]
                nb_positive_samples = min(
                    self.expl_kept_points // 2, len(labels_positive)
                )
                kept_postitive_samples = np.random.randint(
                    len(labels_positive), size=(nb_positive_samples,)
                )
                labels_positive = labels_positive[kept_postitive_samples]
                pos_positive = pos_positive[kept_postitive_samples]

                # Sampling negative samples
                negative_samples = (labels[i] == 0) * (kept_pos[i] == 1)
                labels_negative = labels[i][negative_samples]
                pos_negative = pos_expl[i][negative_samples]
                nb_negative_samples = self.expl_kept_points - nb_positive_samples
                if len(labels_negative) >= nb_negative_samples:
                    kept_negative_samples = np.random.randint(
                        len(labels_negative), size=(nb_negative_samples,)
                    )
                    labels_negative = labels_negative[kept_negative_samples]
                    pos_negative = pos_negative[kept_negative_samples]
                    pos_expl_curr = torch.cat([pos_negative, pos_positive], dim=0)
                    labels_curr = torch.cat([labels_negative, labels_positive], dim=0)
                else:
                    pos_expl_curr = pos_expl[i][: self.expl_kept_points]
                    labels_curr = labels[i][: self.expl_kept_points]

                # Storing only the 'self.expl_kept_steps' last steps
                idx_curr = int(self.nb_samples_in_ep_expl_occ[i].item())
                if idx_curr == self.expl_kept_steps:
                    self.pos_expl[i] = torch.roll(self.pos_expl[i], -1, 0)
                    self.labels_expl[i] = torch.roll(self.labels_expl[i], -1, 0)
                    idx_curr -= 1

                pos_expl_curr = self.embed_fn(pos_expl_curr)
                self.pos_expl[i, idx_curr, :] = pos_expl_curr
                self.labels_expl[i, idx_curr, :] = labels_curr
                self.labels_expl[i, idx_curr, :] += 1

                if int(self.nb_samples_in_ep_expl_occ[i].item()) < self.expl_kept_steps:
                    self.nb_samples_in_ep_expl_occ[i] += 1

        bs = pos.shape[0]
        pos = self.m(pos.permute((0, 3, 1, 2)))
        pos = pos.permute((0, 2, 3, 1)).unsqueeze(3).unsqueeze(3)

        # Saving rgb and pos samples
        rgb = rgb.view(bs, -1, self.emb_dim)
        pos = pos.view(bs, -1, 1, 1, 3)
        pos = pos.reshape((pos.shape[0], pos.shape[1], -1, pos.shape[-1]))
        for i in range(pos.shape[0]):
            idx = int(self.nb_samples_in_ep[i].item())
            self.rgb[i, idx] = rgb[i]
            self.pos[i, idx] = pos[i]
            self.nb_samples_in_ep[i] += 1
        return

    def get_batch(self):
        rgb = []
        pos = []

        for i in range(len(self.rgb)):
            rgb_env = []
            pos_env = []

            # Sampling indices to retrieve
            nb_samples = int(self.nb_samples_in_ep[i].item())
            if nb_samples <= self.batch_size:
                idxs = random.choices(np.arange(nb_samples), k=self.batch_size)
            else:
                idxs = random.sample(
                    range(nb_samples - self.batch_size // 4),
                    self.batch_size - self.batch_size // 4,
                )
                for j in range(self.batch_size // 4):
                    idxs.append(nb_samples - ((self.batch_size // 4) - j))

            # Retrieving training samples
            rgb_env = self.rgb[i][idxs]
            pos_env = self.pos[i][idxs]

            # Sampling crops
            kept_crops = np.random.randint(
                rgb_env.shape[1], size=(rgb_env.shape[1] // 10,)
            )
            rgb_env = rgb_env[:, kept_crops]
            pos_env = pos_env[:, kept_crops]

            pos.append(pos_env.unsqueeze(0))
            rgb.append(rgb_env.unsqueeze(0))

        pos = torch.cat(pos, dim=0)
        rgb = torch.cat(rgb, dim=0)

        return {"rgb": rgb, "pos": pos}

    def get_batch_get_value(
        self, input_rgb, input_depth, input_gps, input_heading, masks
    ):
        # Building new sample without adding it to the
        # replay buffer (as it is only a value estimation step)
        rgb = input_rgb.permute((0, 2, 3, 1))

        # Re-initialize data for new episodes
        init_rot_to_reinit = []
        for i in range(len(self.rgb)):
            if masks[i] == 0:
                init_rot_to_reinit.append(i)
        self.nb_samples_in_ep *= masks[:, 0]

        # Project depth pixels into 3D coordinates
        pos, _ = reproject(
            input_depth, input_gps, input_heading, self.init_rot, init_rot_to_reinit
        )
        self.pos_0 = None

        # Normalize coordinates between 0 and 1
        pos[:, :, :, 0] = (pos[:, :, :, 0] - self.min_x) / (self.max_x - self.min_x)
        pos[:, :, :, 1] = (pos[:, :, :, 1] - self.min_y) / (self.max_y - self.min_y)
        pos[:, :, :, 2] = (pos[:, :, :, 2] - self.min_z) / (self.max_z - self.min_z)

        bs = pos.shape[0]
        pos = self.m(pos.permute((0, 3, 1, 2)))
        pos = pos.permute((0, 2, 3, 1)).unsqueeze(3).unsqueeze(3)

        rgb_current_obs = rgb.view(bs, -1, self.emb_dim)
        pos = pos.view(bs, -1, 1, 1, 3)
        pos_current_obs = pos.reshape((pos.shape[0], pos.shape[1], -1, pos.shape[-1]))

        # Sampling batch (from replay buffer and current observation)
        rgb = []
        pos = []

        for i in range(len(self.rgb)):
            rgb_env = []
            pos_env = []

            # Sampling indices to retrieve
            if masks[i] == 0:
                rgb_env = rgb_current_obs[i].unsqueeze(0).repeat(self.batch_size, 1, 1)
                pos_env = (
                    pos_current_obs[i].unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                )
            else:
                nb_samples = int(self.nb_samples_in_ep[i].item() + 1)
                if nb_samples <= self.batch_size:
                    idxs = random.choices(np.arange(nb_samples), k=self.batch_size)
                else:
                    idxs = random.sample(
                        range(nb_samples - self.batch_size // 4),
                        self.batch_size - self.batch_size // 4,
                    )
                    for j in range(self.batch_size // 4):
                        idxs.append(nb_samples - ((self.batch_size // 4) - j))

                for idx in idxs:
                    if idx == (nb_samples - 1):
                        # Adding current observation (not stored in the replay buffer)
                        rgb_env.append(rgb_current_obs[i].unsqueeze(0))
                        pos_env.append(pos_current_obs[i].unsqueeze(0))
                    else:
                        # Adding sample from the replay buffer
                        rgb_env.append(self.rgb[i][idx].unsqueeze(0))
                        pos_env.append(self.pos[i][idx].unsqueeze(0))
                rgb_env = torch.cat(rgb_env, dim=0)
                pos_env = torch.cat(pos_env, dim=0)

            # Sampling crops
            kept_crops = np.random.randint(
                rgb_env.shape[1], size=(rgb_env.shape[1] // 10,)
            )
            rgb_env = rgb_env[:, kept_crops]
            pos_env = pos_env[:, kept_crops]

            pos.append(pos_env.unsqueeze(0))
            rgb.append(rgb_env.unsqueeze(0))

        pos = torch.cat(pos, dim=0)
        rgb = torch.cat(rgb, dim=0)

        return {"rgb": rgb, "pos": pos}

    def get_batch_expl_occ(self):
        pos = []
        labels = []

        for i in range(len(self.labels_expl)):
            labels_env = []
            pos_env = []

            # Sampling positive sample indices to retrieve
            nb_samples = int(self.nb_samples_in_ep_expl_occ[i].item())
            if nb_samples <= self.batch_size:
                idxs = random.choices(np.arange(nb_samples), k=self.batch_size)
            else:
                idxs = random.sample(
                    range(nb_samples - self.batch_size // 4),
                    self.batch_size - self.batch_size // 4,
                )
                for j in range(self.batch_size // 4):
                    idxs.append(nb_samples - ((self.batch_size // 4) - j))

            # Retrieving positive training samples
            labels_env = self.labels_expl[i][idxs]
            pos_env = self.pos_expl[i][idxs]

            # Sampling crops
            kept_crops = np.random.randint(
                labels_env.shape[1], size=(labels_env.shape[1] // 50,)
            )
            labels_env = labels_env[:, kept_crops]
            pos_env = pos_env[:, kept_crops]

            labels_env = labels_env.view(-1)
            pos_env = pos_env.view(-1, self.expl_pos_emb_dim)

            # Samling negative samples
            kept_negatives = random.randint(0, self.negatives.shape[0] - 1)
            negatives = self.negatives[kept_negatives]
            negatives = negatives.view(-1, self.expl_pos_emb_dim)

            labels_env = torch.cat((labels_env, self.labels_negatives.view(-1)), dim=0)
            pos_env = torch.cat((pos_env, negatives), dim=0)

            pos.append(pos_env.unsqueeze(0))
            labels.append(labels_env.unsqueeze(0))

        pos = torch.cat(pos, dim=0)
        labels = torch.cat(labels, dim=0)
        return {
            "pos": pos,
            "labels": labels,
        }

    def get_batch_expl_occ_get_value(
        self, input_depth, input_gps, input_heading, masks
    ):
        # Building new sample without adding it to the
        # replay buffer (as it is only a value estimation step)

        # Re-initialize data for new episodes
        init_rot_to_reinit = []
        for i in range(self.num_envs):
            if masks[i] == 0:
                init_rot_to_reinit.append(i)
        self.nb_samples_in_ep_expl_occ *= masks[:, 0]

        # Project depth pixels into 3D coordinates
        pos, _ = reproject(
            input_depth, input_gps, input_heading, self.init_rot, init_rot_to_reinit
        )

        labels = pos[:, :, :, 1] < self.expl_labels_thresh
        kept_pos = pos[:, :, :, 1] < self.expl_kept_pos_thresh

        # Normalize coordinates between 0 and 1
        pos[:, :, :, 0] = (pos[:, :, :, 0] - self.min_x) / (self.max_x - self.min_x)
        pos[:, :, :, 1] = (pos[:, :, :, 1] - self.min_y) / (self.max_y - self.min_y)
        pos[:, :, :, 2] = (pos[:, :, :, 2] - self.min_z) / (self.max_z - self.min_z)

        pos_expl = pos.reshape((self.num_envs, -1, 3))
        pos_expl = torch.cat(
            [pos_expl[:, :, 0].unsqueeze(-1), pos_expl[:, :, 2].unsqueeze(-1)], dim=-1
        )
        labels = labels.reshape((self.num_envs, -1))
        kept_pos = kept_pos.reshape((self.num_envs, -1))

        pos_expl_current_obs = []
        labels_expl_current_obs = []
        for i in range(self.num_envs):
            # Sampling positive samples
            positive_samples = (labels[i] == 1) * (kept_pos[i] == 1)
            labels_positive = labels[i][positive_samples]
            pos_positive = pos_expl[i][positive_samples]
            nb_positive_samples = min(self.expl_kept_points // 2, len(labels_positive))
            kept_postitive_samples = np.random.randint(
                len(labels_positive), size=(nb_positive_samples,)
            )
            labels_positive = labels_positive[kept_postitive_samples]
            pos_positive = pos_positive[kept_postitive_samples]

            # Sampling negative samples
            negative_samples = (labels[i] == 0) * (kept_pos[i] == 1)
            labels_negative = labels[i][negative_samples]
            pos_negative = pos_expl[i][negative_samples]
            nb_negative_samples = self.expl_kept_points - nb_positive_samples
            if len(labels_negative) >= nb_negative_samples:
                kept_negative_samples = np.random.randint(
                    len(labels_negative), size=(nb_negative_samples,)
                )
                labels_negative = labels_negative[kept_negative_samples]
                pos_negative = pos_negative[kept_negative_samples]
                pos_expl_curr = torch.cat([pos_negative, pos_positive], dim=0)
                labels_curr = torch.cat([labels_negative, labels_positive], dim=0)
            else:
                pos_expl_curr = pos_expl[i][: self.expl_kept_points]
                labels_curr = labels[i][: self.expl_kept_points]
            pos_expl_curr = self.embed_fn(pos_expl_curr)
            pos_expl_current_obs.append(pos_expl_curr.unsqueeze(0))
            labels_expl_current_obs.append((labels_curr + 1).unsqueeze(0))

        pos_expl_current_obs = torch.cat(pos_expl_current_obs, dim=0)
        labels_expl_current_obs = torch.cat(labels_expl_current_obs, dim=0)

        # Sampling batch (from replay buffer and current observation)
        pos = []
        labels = []

        for i in range(len(self.rgb)):
            labels_env = []
            pos_env = []

            # Sampling positive sample indices to retrieve
            if masks[i] == 0:
                labels_env = (
                    labels_expl_current_obs[i].unsqueeze(0).repeat(self.batch_size, 1)
                )
                pos_env = (
                    pos_expl_current_obs[i].unsqueeze(0).repeat(self.batch_size, 1, 1)
                )
            else:
                nb_samples = int(self.nb_samples_in_ep_expl_occ[i].item() + 1)
                if nb_samples <= self.batch_size:
                    idxs = random.choices(np.arange(nb_samples), k=self.batch_size)
                else:
                    idxs = random.sample(
                        range(nb_samples - self.batch_size // 4),
                        self.batch_size - self.batch_size // 4,
                    )
                    for j in range(self.batch_size // 4):
                        idxs.append(nb_samples - ((self.batch_size // 4) - j))

                for idx in idxs:
                    if idx == (nb_samples - 1):
                        # Adding current observation (not stored in the replay buffer)
                        labels_env.append(labels_expl_current_obs[i].unsqueeze(0))
                        pos_env.append(pos_expl_current_obs[i].unsqueeze(0))
                    else:
                        # Adding sample from the replay buffer
                        labels_env.append(self.labels_expl[i][idx].unsqueeze(0))
                        pos_env.append(self.pos_expl[i][idx].unsqueeze(0))
                labels_env = torch.cat(labels_env, dim=0)
                pos_env = torch.cat(pos_env, dim=0)

            # Sampling crops
            kept_crops = np.random.randint(
                labels_env.shape[1], size=(labels_env.shape[1] // 50,)
            )
            labels_env = labels_env[:, kept_crops]
            pos_env = pos_env[:, kept_crops]

            labels_env = labels_env.view(-1)
            pos_env = pos_env.view(-1, self.expl_pos_emb_dim)

            # Samling negative samples
            kept_negatives = random.randint(0, self.negatives.shape[0] - 1)
            negatives = self.negatives[kept_negatives]
            negatives = negatives.view(-1, self.expl_pos_emb_dim)

            labels_env = torch.cat((labels_env, self.labels_negatives.view(-1)), dim=0)
            pos_env = torch.cat((pos_env, negatives), dim=0)

            pos.append(pos_env.unsqueeze(0))
            labels.append(labels_env.unsqueeze(0))

        pos = torch.cat(pos, dim=0)
        labels = torch.cat(labels, dim=0)
        return {"pos": pos, "labels": labels}


def run_epoch(criterion_l1, net_sem_finder, optimizer_sem_finder, features, xyz):
    xyz_pred = []
    for i in range(len(features)):
        if len(features[i]) > 0:
            xyz_pred_ = net_sem_finder[i](features[i])
            xyz_pred.append(xyz_pred_)
        else:
            xyz_pred.append(None)

    for i in range(len(features)):
        if xyz_pred[i] is not None:
            loss_sem_finder_l1 = criterion_l1(xyz_pred[i], xyz[i])
            optimizer_sem_finder[i].zero_grad()
            loss_sem_finder_l1.backward()
            optimizer_sem_finder[i].step()

    return
