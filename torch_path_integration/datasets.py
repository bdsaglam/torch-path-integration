import json
import pathlib
import random

import matplotlib

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

        loaded_data = self.loaded_data()
        initial_location = self.normalize_location(torch.tensor(loaded_data['init_pos'][index]).unsqueeze(0))  # (1, 2)
        initial_orientation = torch.tensor(loaded_data['init_hd'][index]).unsqueeze(0)  # (1, 1)
        # (T, 3), (u., sin(φ.), cos(φ.))
        velocity = self.normalize_velocity(torch.tensor(loaded_data['ego_vel'][index]))

        target_location = self.normalize_location(torch.tensor(loaded_data['target_pos'][index]))  # (T, 2)
        target_orientation = torch.tensor(loaded_data['target_hd'][index])  # (T, 1), [-π, π]

        return (initial_location, initial_orientation, velocity), (target_location, target_orientation)

    def normalize_location(self, x):
        return (x / (self.env_size / 2) + 1) / 2

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

        self.episodes = sorted([p.stem for p in (self.root / 'images').glob('*') if p.is_dir()])

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = self.episodes[index]

        # action
        action_dir = self.root / 'actions' / episode
        action_filepaths = list(action_dir.glob("*.txt"))
        actions = [torch.tensor(int(fp.read_text()), dtype=torch.int64)
                   for fp in action_filepaths]
        action = F.one_hot(torch.stack(actions, 0), self.action_space_dim).float()  # (T + 1, A)

        # pose
        pose_dir = self.root / 'poses' / episode
        pose_filepaths = [pose_dir / (fp.stem + '.txt') for fp in action_filepaths]
        poses = [[float(field) for field in fp.read_text().split(' ')]
                 for fp in pose_filepaths]

        x, _, z, o = list(zip(*poses))
        x = torch.tensor(x) / self.env_width
        z = torch.tensor(z) / self.env_height
        location = torch.stack([x, z], 1)  # (T + 1, 2)

        orientation = torch.tensor(o).unsqueeze(-1)  # (T + 1, 1)
        mask = orientation > np.pi
        orientation[mask] = orientation[mask] - 2 * np.pi

        initial_location = location[:1]  # (1, 2)
        initial_orientation = orientation[:1]  # (1, 1)
        velocity = action[:-1]  # (T, A)

        target_location = location[1:]  # (T, 2)
        target_orientation = orientation[1:]  # (T, 1)

        return (initial_location, initial_orientation, velocity), (target_location, target_orientation)
