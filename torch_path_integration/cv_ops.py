import numpy as np
import torch


def make_homogenous(tmf):  # (..., 6)
    shape = tmf.shape[:-1] + (2, 3)
    tmf = tmf.view(*shape)
    zeros = torch.zeros_like(tmf[..., :1, 0])
    last = torch.stack([zeros, zeros, zeros + 1], -1)
    tm = torch.cat([tmf, last], -2)  # (..., 3, 3)
    return tm


def remove_homogenous(tm):  # (..., 3, 3)
    return tm[..., :2, :].view(*tm.shape[:-2], 6)  # (..., 6)


def affine_transform_2d(rotation=None,
                        scale_x=None,
                        scale_y=None,
                        shear=None,
                        trans_x=None,
                        trans_y=None):
    existings = [x for x in (scale_x, scale_y, rotation, shear, trans_x, trans_y)
                 if x is not None]
    device = existings[0].device
    batch_shape = existings[0].shape

    if scale_x is None:
        scale_x = torch.ones(*batch_shape, device=device)
    if scale_y is None:
        scale_y = torch.ones(*batch_shape, device=device)
    if rotation is None:
        rotation = torch.zeros(*batch_shape, device=device)
    if shear is None:
        shear = torch.zeros(*batch_shape, device=device)
    if trans_x is None:
        trans_x = torch.zeros(*batch_shape, device=device)
    if trans_y is None:
        trans_y = torch.zeros(*batch_shape, device=device)

    c, s = torch.cos(rotation), torch.sin(rotation)

    tmf = [
        scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c, trans_x,
        scale_y * s, scale_y * c, trans_y
    ]
    tmf = torch.stack(tmf, -1)

    return make_homogenous(tmf)


def decompose_transformation_matrix(T, eps=1e-8):
    tx = T[..., 0, 2]
    ty = T[..., 1, 2]

    rot = torch.atan2(T[..., 1, 0], T[..., 1, 1])

    sy_sq = T[..., 1, 0].pow(2) + T[..., 1, 1].pow(2)
    sy = sy_sq.sqrt()

    det = T[..., 0, 0] * T[..., 1, 1] - T[..., 0, 1] * T[..., 1, 0]  # sx * sy
    sx = det / (sy + eps)

    sh = (T[..., 0, 0].pow(2) + T[..., 0, 1].pow(2) - sx.pow(2)) / (sy_sq + eps)

    return rot, sx, sy, sh, tx, ty


def apply_nonlinear(transformation_params):
    rot, sx, sy, sh, tx, ty = transformation_params.split(1, -1)
    sx, sy = (torch.sigmoid(t) + 1e-2 for t in (sx, sy))

    tx, ty, sh = (torch.tanh(t * 5.) for t in (tx, ty, sh))
    rot *= np.pi

    return torch.cat([rot, sx, sy, sh, tx, ty], -1)
