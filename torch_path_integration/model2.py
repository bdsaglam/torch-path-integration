import numpy as np
import torch
import torch.nn as nn

from torch_path_integration import cv_ops
from torch_path_integration.nn_ext import MLP


class ActionPoseEstimator(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.linear = nn.Linear(action_dim, 3)

    def forward(self, action):
        pose = self.linear(action)
        rot, tx, ty = pose.split(1, -1)
        return rot * np.pi, tx, ty


class ContextAwareStepIntegrator(nn.Module):
    def __init__(self, action_dim, hidden_dims=(32,)):
        super().__init__()
        self.action_pose_estimator = ActionPoseEstimator(action_dim)

        self.context_pose_estimator = MLP(
            sizes=(6 + 6, *hidden_dims, 3),
            activate_final=False
        )

    def forward(self, t_in, action):
        """
        Estimates transformation matrix from action and input transformation matrix.

        :param t_in: torch.tensor, [B, 3, 3], input transformation matrix
        :param action: torch.tensor, [B, A], encoded action
        :return: torch.tensor, [B, 3, 3], estimated transformation matrix
        """

        drot, dtx, dty = self.action_pose_estimator(action)  # (B, 1), (B, 1), (B, 1)
        t_a = cv_ops.affine_transform_2d(rotation=drot, trans_x=dtx, trans_y=dty)  # (B, 3, 3)
        pose_act = cv_ops.remove_homogeneous(t_a)  # (B, 6)

        pose_in = cv_ops.remove_homogeneous(t_in)  # (B, 6)

        pose = torch.cat([pose_in, pose_act], -1)  # (B, 12)
        rot, tx, ty = self.context_pose_estimator(pose).split(1, -1)  # (B, 1), (B, 1), (B, 1)

        t_d = cv_ops.affine_transform_2d(rotation=rot, trans_x=tx, trans_y=ty)  # (B, 3, 3)

        t_out = t_in @ t_d  # (B, 3, 3)
        return t_out


class ContextAwarePathIntegrator(nn.Module):
    def __init__(self, action_dim, hidden_dims=(32,)):
        super().__init__()
        self.step_integrator = ContextAwareStepIntegrator(action_dim, hidden_dims)

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
        :return: torch.tensor, [B, T, 3, 3], estimated transformation matrix
        """

        B, T = action.shape[:2]

        if mask is not None:
            mask = mask[:, :, None, None].repeat(1, 1, 3, 3)  # (B, T, 3, 3)

        t_in = t_init[:, 0]  # (B, 3, 3)

        t_outs = []
        for i in range(T):
            t_out = self.step_integrator(t_in, action[:, i])
            t_outs.append(t_out)

            if mask is not None and t_direct is not None:
                t_in = torch.where(mask[:, i], t_direct[:, i], t_out)
            else:
                t_in = t_out

        return torch.stack(t_outs, 1)  # (B, T, 3, 3)
