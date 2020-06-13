import numpy as np

import torch
import torch.nn.functional as F


class PlaceCellEnsemble:
    def __init__(self, n_cells, std):
        super().__init__()
        self.n_cells = n_cells
        self.centers = 2 * torch.rand(n_cells, 2) - 1  # (N, 2)
        self.var = std ** 2

    def encode(self, location):  # (B, 2)
        sq_distance = (location.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(-1)
        unnormalized_log_prob = -sq_distance / (2 * self.var)
        prob = F.softmax(unnormalized_log_prob, dim=-1)  # (B, N)
        return prob

    def decode(self, prob, strategy='soft'):  # (B, N)
        if strategy == 'soft':
            return prob @ self.centers
        if strategy == 'hard':
            return self.centers[prob.argmax(-1)]

        raise ValueError("Unknown strategy.")


class HeadDirectionCellEnsemble:
    def __init__(self, n_cells, kappa):
        super().__init__()
        self.n_cells = n_cells
        self.centers = np.pi * (2 * torch.rand(n_cells, 1) - 1)
        self.kappa = kappa

    def encode(self, angle):  # (B, 1)
        unnormalized_log_prob = self.kappa * torch.cos(angle - self.centers.T)
        prob = F.softmax(unnormalized_log_prob, dim=-1)  # (B, N)
        return prob

    def decode(self, prob, strategy='soft'):  # (B, N)
        if strategy == 'soft':
            return prob @ self.centers
        if strategy == 'hard':
            return self.centers[prob.argmax(-1)]

        raise ValueError("Unknown strategy.")
