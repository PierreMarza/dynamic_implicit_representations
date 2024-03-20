###############################################################################
# Base code from:                                                             #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

import math
import numpy as np
import torch


def get_obj_poses(observations):
    mean_i_obj = []
    mean_j_obj = []
    gt_seen = []
    not_visible_goals = []

    mapCache = observations["semMap"]
    for batch in range(mapCache.shape[0]):
        indices = (
            mapCache[batch, :, :, 1] == observations["multiobjectgoal"][batch] + 2
        ).nonzero()
        i_obj = indices[:, 0].type(torch.FloatTensor)
        j_obj = indices[:, 1].type(torch.FloatTensor)

        if len(i_obj) == 0 or len(j_obj) == 0:
            assert len(i_obj) == 0 and len(j_obj) == 0
            mean_i_obj.append(-1)
            mean_j_obj.append(-1)
            not_visible_goals.append(batch)
            gt_seen.append(0)
        else:
            mean_i_obj.append(torch.mean(i_obj).item())
            mean_j_obj.append(torch.mean(j_obj).item())
            gt_seen.append(1)

    # Convert lists to arrays
    mean_i_obj = np.array(mean_i_obj)
    mean_j_obj = np.array(mean_j_obj)

    return mean_i_obj, mean_j_obj, gt_seen, not_visible_goals


def compute_distance_labels(mean_i_obj, mean_j_obj, not_visible_goals, device):
    euclidian_distance = np.sqrt(
        ((25 - mean_i_obj).astype(np.float64)) ** 2
        + ((25 - mean_j_obj).astype(np.float64)) ** 2
    )
    euclidian_distance = np.floor(euclidian_distance)
    distance_labels = euclidian_distance.astype(np.int_)

    if len(not_visible_goals) > 0:
        distance_labels[not_visible_goals] = -1
    distance_labels = torch.from_numpy(distance_labels).to(device)

    return distance_labels


def compute_direction_labels(mean_i_obj, mean_j_obj, not_visible_goals, device):
    bin_size = 360 / 12
    x_diffs = (mean_i_obj - 25).astype(np.float64)
    y_diffs = (mean_j_obj - 25).astype(np.float64)

    mask = 1 - (np.where((x_diffs == 0.0), 1, 0)) * (np.where((y_diffs == 0.0), 1, 0))
    atans = np.arctan2(x_diffs, y_diffs)
    atans = (-atans) % (2 * math.pi)

    angle_diff = np.degrees(atans)
    angle_diff *= mask
    angle_diff = np.floor(angle_diff)

    direction_labels = (angle_diff - (bin_size / 2)) % 360 // bin_size
    direction_labels = direction_labels.astype(np.int_)
    if len(not_visible_goals) > 0:
        direction_labels[not_visible_goals] = -1
    direction_labels = torch.from_numpy(direction_labels).to(device)

    return direction_labels
