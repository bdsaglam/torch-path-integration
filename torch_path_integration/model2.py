import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_path_integration import cv_ops


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
        t_gt = cv_ops.affine_transform_2d(rotation=rot, trans_x=tx, trans_y=ty)
        return F.mse_loss(t_out, t_gt)


class ContextAwarePathIntegrator(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_converter = ActionConverter(action_dim)
        self.linear = nn.Sequential(
            nn.Linear(6 + 6, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, t_init, action, t_direct=None, mask=None):
        """
        Performs path-integration starting from initial transformation matrix,
        t_init, and applying actions successively.
        The errors occur at any time step accumulates over whole sequence.
        Therefore, the network needs a correction signal to perform ,
        path-integration module looks at 'true' transformation state, t_direct,
        at randomly selected time steps. The 'true' transformation state can be
        an inference by another network using other modalities such as vision
        or it can be derived from ground truth location and orientation.
        The mask is used to determine at which time steps to look at t_direct.

        For any time step,
            `T_new = T_old @ T_diff`

        :param t_init: torch.tensor, [B, 1, 3, 3], initial transformation matrix
        :param action: torch.tensor, [B, T, A], encoded action
        :param t_direct: torch.tensor, [B, T, 3, 3], direct transformation matrix
        :param mask: torch.tensor, [B, T], mask
        :return: torch.tensor, [B, T, 3, 3], target transformation matrix
        """

        B, T = action.shape[:2]

        if mask is not None:
            mask = mask[:, :, None, None].repeat(1, 1, 3, 3)  # (B, T, 3, 3)

        t_ctx = t_init[:, 0]  # (B, 3, 3)

        t_outs = []
        for i in range(T):
            pose_ctx = cv_ops.remove_homogenous(t_ctx)  # (B, 6)

            drot, dx, dy = self.action_converter(action[:, i, :])  # (B, ), (B, ), (B, )
            t_a = cv_ops.affine_transform_2d(
                rotation=drot, trans_x=dx, trans_y=dy)  # (B, 3, 3)
            pose_act = cv_ops.remove_homogenous(t_a)  # (B, 6)

            pose = torch.cat([pose_ctx, pose_act], -1)  # (B, 12)

            rot, tx, ty = [t.squeeze(-1) for t in self.linear(pose).split(1, -1)]  # (B, ), (B, ), (B, )

            t_d = cv_ops.affine_transform_2d(
                rotation=rot, trans_x=tx, trans_y=ty)  # (B, 3, 3)

            t_out = t_ctx @ t_d  # (B, 3, 3)
            t_outs.append(t_out)

            if mask is not None and t_direct is not None:
                t_ctx = torch.where(mask[:, i], t_direct[:, i], t_out)
            else:
                t_ctx = t_out

        return torch.stack(t_outs, 1)  # (B, T, 3, 3)

    def loss(self, t_out, target_location, target_orientation):
        rot = target_orientation.squeeze(-1)
        tx, ty = target_location[..., 0], target_location[..., 1]
        t_gt = cv_ops.affine_transform_2d(
            rotation=rot, trans_x=tx, trans_y=ty)
        return F.mse_loss(t_out, t_gt)
