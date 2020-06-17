import pathlib
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from monty.collections import AttrDict
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.backends import cudnn
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch_path_integration import cv_ops, visualization
from torch_path_integration.datasets import WestWorldDataset, MouseDataset
from torch_path_integration.model2 import ContextAwarePathIntegrator, PathIntegrator
from torch_path_integration.optimizers import RAdam, LookAhead
from torch_path_integration.visualization import PathVisualizer

Tensor = torch.Tensor


class PIMExperiment2(LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams
        self.model = PathIntegrator(**hparams.model)
        self.path_vis = self.make_path_visualizer()

    def make_path_visualizer(self):
        bg_image = None

        root = pathlib.Path(self.hparams.data_dir)
        fp = root / 'top_view.png'
        if fp.exists():
            from PIL import Image

            image = Image.open(fp)
            img = np.asarray(image)

            pad = 36
            crop_img = img[pad:-pad, pad:-pad]

            h, w, c = crop_img.shape
            alpha = np.full((h, w, 1), fill_value=100)
            bg_image = np.concatenate([crop_img, alpha], -1)

        pv = PathVisualizer(background_image=bg_image, figsize_per_example=(6, 6))
        return pv

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        # optimizer args
        parser.add_argument('--optimizer_type', type=str, default='RMSprop')
        parser.add_argument('--learning_rate', type=float, default=3e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--look_ahead', action='store_true')
        parser.add_argument('--look_ahead_k', type=int, default=5)
        parser.add_argument('--look_ahead_alpha', type=float, default=0.5)
        parser.add_argument('--use_lr_scheduler', action='store_true')
        parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.95)
        return parser

    def forward(self, action_embedding, hidden_state) -> Tensor:
        return self.model(action_embedding, hidden_state)

    def configure_optimizers(self):
        eps = 1e-2 / float(self.hparams.batch_size) ** 2
        if self.hparams.optimizer_type == "RMSprop":
            optimizer = RMSprop(self.parameters(),
                                lr=self.hparams.learning_rate,
                                momentum=0.9,
                                eps=eps,
                                weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type == "RAdam":
            optimizer = RAdam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              eps=eps,
                              weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type == "Adam":
            optimizer = Adam(self.parameters(),
                             lr=self.hparams.learning_rate,
                             eps=eps,
                             weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Unknown optimizer type.")

        if self.hparams.look_ahead:
            optimizer = LookAhead(optimizer,
                                  k=self.hparams.look_ahead_k,
                                  alpha=self.hparams.look_ahead_alpha)

        if not self.hparams.use_lr_scheduler:
            return optimizer

        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.hparams.lr_scheduler_decay_rate)

        return [optimizer], [scheduler]

    def prepare_data(self):
        if self.hparams.dataset_type == 'westworld':
            ds = WestWorldDataset(self.hparams.data_dir)
        elif self.hparams.dataset_type == 'mouse':
            ds = MouseDataset(self.hparams.data_dir, self.hparams.env_size)
        else:
            raise ValueError(f"Unknown dataset type {self.hparams.dataset_type}")

        n = len(ds)
        tn = int(n * 0.8)
        vn = n - tn
        self.train_dataset, self.val_dataset = random_split(ds, [tn, vn])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def inference(self, batch):
        # (B, 1, 2), (B, 1, 1), (B, T, A), (B, T, 2), (B, T, 1)
        (initial_location, initial_orientation, action), (target_location, target_orientation) = batch
        B, T = target_location.shape[:2]

        rot = torch.cat([initial_orientation, target_orientation], -2).squeeze(-1)[:, :-1]
        location = torch.cat([initial_location, target_location], -2)[:, :-1, :]
        tx, ty = location[..., 0], location[..., 1]
        t_in = cv_ops.affine_transform_2d(rotation=rot, trans_x=tx, trans_y=ty)

        t_out = self.model(t_in, action)

        return AttrDict(t_out=t_out,
                        target_location=target_location,
                        target_orientation=target_orientation)

    def training_step(self, batch, batch_idx):
        res = self.inference(batch)
        loss = self.model.loss(res.t_out, res.target_location, res.target_orientation)
        log = dict(
            loss=loss.detach(),
        )
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        res = self.inference(batch)
        loss = self.model.loss(res.t_out, res.target_location, res.target_orientation)
        out = dict(val_loss=loss)
        if batch_idx == 0:
            out.update(batch=batch)
            out.update(res=res)
        return out

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_val_loss}

        # log predictions
        batch = outputs[0]['batch']
        res = outputs[0]['res']
        self.visualize_episode(batch, res)

        return {'val_loss': avg_val_loss, 'log': log}

    def visualize_episode(self, batch, res):
        # (B, 1, 2), (B, 1, 1), (B, T, A), (B, T, 2), (B, T, 1)
        (initial_location, initial_orientation, velocity), (target_location, target_orientation) = batch
        B = min(8, target_location.shape[0])
        T = target_location.shape[1]

        rot, sx, sy, sh, tx, ty = cv_ops.decompose_transformation_matrix(res.t_out[:B])
        pred_path = torch.stack([tx, ty], -1).detach().numpy()

        gt_path = torch.cat([initial_location[:B], target_location[:B]], 1).numpy()

        loc_fig = self.path_vis.plot(gt_path, pred_path)
        loc_vis = visualization.fig_to_tensor(loc_fig)
        plt.close(loc_fig)
        self.logger.experiment.add_image('paths', loc_vis, self.current_epoch)


if __name__ == '__main__':
    # For reproducibility
    seed_everything(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    parser = ArgumentParser()
    parser.add_argument('--hparams_file', type=str, default=None)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare hparams
    hparams_file = pathlib.Path(args.hparams_file)
    hparams = yaml.safe_load(hparams_file.read_text())

    experiment = PIMExperiment2(
        hparams=Namespace(**hparams),
    )

    # prepare trainer params
    trainer_params = hparams['trainer']
    trainer = Trainer(**trainer_params)
    trainer.fit(experiment)
