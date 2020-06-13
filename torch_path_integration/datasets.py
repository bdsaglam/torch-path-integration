import json
import pathlib
import random

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MouseDataset(Dataset):
    def __init__(self, root, env_size, load_from_disk_period=256):
        self.root = pathlib.Path(root)
        self.file_list = list(self.root.glob('*'))
        self.num_records_per_file = 10_000
        self.load_from_disk_period = load_from_disk_period
        self.env_size = float(env_size)
        self.data = None
        self.counter = 0

    def __len__(self):
        return self.num_records_per_file * len(self.file_list)

    def __getitem__(self, index):
        # Load data and get label
        index = index % self.num_records_per_file

        bulk = self.loaded_data()
        initial_location = self.normalize_location(bulk['init_pos'][index].unsqueeze(0))  # (1, 2)
        initial_orientation = bulk['init_hd'][index].unsqueeze(0)  # (1, 1)
        # (T, 3), (u., sin(φ.), cos(φ.))
        velocity = self.normalize_velocity(bulk['ego_vel'][index])

        target_location = self.normalize_location(bulk['target_pos'][index])  # (T, 2)
        target_orientation = bulk['target_hd'][index]  # (T, 1), [-π, π]

        return (initial_location, initial_orientation, velocity), (target_location, target_orientation)

    def normalize_location(self, x):
        return x / self.env_size

    def normalize_velocity(self, v):
        return v / torch.tensor([self.env_size, 1, 1])

    def loaded_data(self):
        if self.data is None or self.counter % self.load_from_disk_period == 0:
            self.data = torch.load(random.choice(self.file_list))

        self.counter += 1
        return self.data


class WestWorldDataset(Dataset):
    def __init__(self, root):
        self.root = pathlib.Path(root)

        config_file = self.root / 'config.json'
        config = json.load(config_file.open('r'))
        self.env_height = config['height']
        self.env_width = config['width']
        self.action_space_dim = config['action_space_dim']

        self.action_files = sorted(list((self.root / 'actions').glob('*.csv')))
        self.pose_files = sorted(list((self.root / 'poses').glob('*.csv')))
        assert len(self.action_files) == len(self.pose_files)

    def __len__(self):
        return len(self.action_files)

    def __getitem__(self, index):
        action_file = self.action_files[index]
        pose_file = self.pose_files[index]
        assert action_file.stem == pose_file.stem

        # action
        action = [int(line) for line in action_file.read_text().splitlines()[1:]]
        action = torch.tensor(action, dtype=torch.int64)
        action = F.one_hot(action, self.action_space_dim).float()  # (T + 1, A)

        # pose
        df = pd.read_csv(pose_file)
        x, z, phi = df['x'].values, df['z'].values, df['phi'].values
        x = 2 * torch.tensor(x, dtype=torch.float32) / self.env_width - 1
        z = 2 * torch.tensor(z, dtype=torch.float32) / self.env_height - 1
        location = torch.stack([x, z], 1)  # (T + 1, 2)

        orientation = torch.tensor(phi, dtype=torch.float32).unsqueeze(-1)  # (T + 1, 1)
        mask = orientation > np.pi
        orientation[mask] = orientation[mask] - 2 * np.pi

        initial_location = location[:1]  # (1, 2)
        initial_orientation = orientation[:1]  # (1, 1)
        velocity = action[:-1]  # (T, A)

        target_location = location[1:]  # (T, 2)
        target_orientation = orientation[1:]  # (T, 1)

        return (initial_location, initial_orientation, velocity), (target_location, target_orientation)
