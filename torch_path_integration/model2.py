import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_path_integration import cv_ops
from torch_path_integration.cv_ops import affine_transform_2d


class ActionConverter(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.linear = nn.Linear(action_dim, 3)

    def forward(self, action):
        delta = self.linear(action)
        drot_logit, dx, dy = delta[..., 0], delta[..., 1], delta[..., 2]
        drot = drot_logit * np.pi
        return drot, dx, dy


class ContextAwareActionConverter(nn.Module):
    def __init__(self, action_dim, context_dim):
        super().__init__()
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.linear = nn.Linear(action_dim + context_dim, 3)

    def forward(self, action, context):
        delta = self.linear(torch.cat([action, context], -1))
        drot_logit, dx, dy = delta[..., 0], delta[..., 1], delta[..., 2]
        drot = torch.tanh(drot_logit) * np.pi
        return drot, dx, dy


class StepIntegrator(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_converter = ActionConverter(action_dim)

    def forward(self, t_in, action):
        drot, dx, dy = self.action_converter(action)
        t_act = cv_ops.affine_transform_2d(rotation=drot, trans_x=dx, trans_y=dy)
        return t_in @ t_act

    def loss(self, t_out, target_location, target_orientation):
        rot = target_orientation.squeeze(-1)
        tx, ty = target_location[..., 0], target_location[..., 1]
        t_gt = affine_transform_2d(rotation=rot,
                                   trans_x=tx,
                                   trans_y=ty)
        return F.mse_loss(t_out, t_gt)


class ContextAwareStepIntegrator(nn.Module):
    def __init__(self, action_dim, other_context_dim=0):
        super().__init__()
        self.action_converter = ActionConverter(action_dim)
        self.linear = nn.Linear(6 + 6 + other_context_dim, 6)

    def forward(self, t_in, action, other_context=None):
        drot, dx, dy = self.action_converter(action)
        t_act = cv_ops.affine_transform_2d(rotation=drot, trans_x=dx, trans_y=dy)

        pose_in = cv_ops.remove_homogenous(t_in)
        pose_act = cv_ops.remove_homogenous(t_act)
        pose = torch.cat([pose_in, pose_act], -1)
        if other_context is not None:
            pose = torch.cat([pose, other_context], -1)

        pose_final = self.linear(pose)
        t_final = cv_ops.make_homogenous(pose_final)
        return t_in @ t_final

    def loss(self, t_out, target_location, target_orientation):
        rot = target_orientation.squeeze(-1)
        tx, ty = target_location[..., 0], target_location[..., 1]
        t_gt = affine_transform_2d(rotation=rot,
                                   trans_x=tx,
                                   trans_y=ty)
        return F.mse_loss(t_out, t_gt)


class PathIntegrator(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_converter = ActionConverter(action_dim)

    def forward(self, t_in, action):
        B, T = t_in.shape[:2]

        ct = t_in[:, 0, ...]

        t_outs = []
        for i in range(T):
            if np.random.rand(1) < 0.2:
                ct = t_in[:, i]
            drot, dx, dy = self.action_converter(action[:, i])
            t_d = cv_ops.affine_transform_2d(rotation=drot, trans_x=dx, trans_y=dy)
            ct = ct @ t_d
            t_outs.append(ct)

        return torch.stack(t_outs, 1)

    def loss(self, t_out, target_location, target_orientation):
        rot = target_orientation.squeeze(-1)
        tx, ty = target_location[..., 0], target_location[..., 1]
        t_gt = affine_transform_2d(rotation=rot,
                                   trans_x=tx,
                                   trans_y=ty)
        return F.mse_loss(t_out, t_gt)


class ContextAwarePathIntegrator(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_converter = ActionConverter(action_dim)
        self.linear = nn.Sequential(
            nn.Linear(6 + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, t_in, action):
        B, T = t_in.shape[:2]

        ct = t_in[:, 0]

        t_outs = []
        for i in range(T):
            drot, dx, dy = self.action_converter(action[:, i, :])

            if np.random.rand(1) < 0.2:
                ct = t_in[:, i]
                t_d = cv_ops.affine_transform_2d(rotation=drot, trans_x=dx, trans_y=dy)
            else:
                pose_act = torch.stack([drot, dx, dy], -1)
                pose_in = cv_ops.remove_homogenous(ct)
                pose = torch.cat([pose_in, pose_act], -1)

                rot, tx, ty = self.linear(pose).split(1, -1)
                sx, sy, sh = torch.ones_like(rot), torch.ones_like(rot), torch.zeros_like(rot)
                tr_params = torch.cat([rot, sx, sy, sh, tx, ty], -1)
                # tr_params = cv_ops.apply_nonlinear(tr_params)
                rot_ctx, tx_ctx, ty_ctx = tr_params[..., 0], tr_params[..., -2], tr_params[..., -1]

                rot = rot_ctx
                tx = tx_ctx
                ty = ty_ctx

                t_d = cv_ops.affine_transform_2d(
                    rotation=rot,
                    trans_x=tx,
                    trans_y=ty
                )

            t_out = ct @ t_d
            t_outs.append(t_out)
            ct = t_out

        return torch.stack(t_outs, 1)

    def loss(self, t_out, target_location, target_orientation):
        rot = target_orientation.squeeze(-1)
        tx, ty = target_location[..., 0], target_location[..., 1]
        t_gt = affine_transform_2d(rotation=rot,
                                   trans_x=tx,
                                   trans_y=ty)
        return F.mse_loss(t_out, t_gt)


