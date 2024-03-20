import torch


Q_TO_MAT = torch.Tensor(
    [
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
        [[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]],
        [[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    ]
)

Q_TO_Q_CONJ = torch.Tensor([1, -1, -1, -1])


def q_conj(q):
    """
    Input        q: float tensor of size * x 4
    Output conj(q): float tensor of size * x 4
    """
    return q * Q_TO_Q_CONJ.to("cuda:0")


def q_prod(q1, q2):
    """
    Input       q1: float tensor of size * x 4
                q2: float tensor of size * x 4
    Output   q1.q2: float tensor of size * x 4
    """
    mat1 = torch.tensordot(q1, Q_TO_MAT.to("cuda:0"), 1).to("cuda:0")
    res = torch.matmul(mat1, q2.unsqueeze(-1))
    return res.squeeze(-1)


def reproject(depth, gps, head, init_rot=None, init_rot_to_reinit=[]):
    h = 256
    w = 256
    hfov = torch.Tensor([79.0]).to("cuda:0")
    min_depth = 0
    max_depth = 10
    sensor_pos = torch.Tensor([0, 0, 0.88, 0]).to("cuda:0")
    f = 0.5 * w / torch.tan(0.5 * torch.deg2rad(hfov))

    x_rect = (torch.arange(w).view(1, 1, w) - 0.5 * w).to("cuda:0") / f
    y_rect = (torch.arange(h).view(1, h, 1) - 0.5 * h).to("cuda:0") / f

    depth = min_depth + (max_depth - min_depth) * depth.squeeze(-1)
    gps = gps.view(-1, 1, 1, 2)
    head = head.view(-1, 1, 1)

    q = torch.stack(
        (torch.zeros_like(depth), depth * x_rect, -depth * y_rect, -depth), -1
    )
    q += sensor_pos

    rot = torch.stack(
        (
            torch.cos(0.5 * head),
            torch.zeros_like(head),
            torch.sin(0.5 * head),
            torch.zeros_like(head),
        ),
        -1,
    )

    if init_rot is None:
        init_rot = rot
        rot = torch.zeros_like(init_rot)
        rot[..., 0] = 1
    elif len(init_rot_to_reinit) > 0:
        for i in range(len(init_rot)):
            if i in init_rot_to_reinit:
                init_rot[i] = rot[i]
                rot[i] = 0
                rot[i, :, :, 0] = 1
            else:
                rot[i] = q_prod(rot[i].unsqueeze(0), q_conj(init_rot[i].unsqueeze(0)))
    else:
        rot = q_prod(rot, q_conj(init_rot))

    pts = q_prod(q_prod(rot, q), q_conj(rot))[..., 1:]
    pts[..., 0] += gps[..., 1]
    pts[..., 2] -= gps[..., 0]
    return pts, init_rot
