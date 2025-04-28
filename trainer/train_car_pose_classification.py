from pathlib import Path
import os

import torch
import torch.nn as nn
import numpy as np
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from dataset.image_pose import ImagePoseDataset
from trainer import create_trainer
from model.discriminator import PoseClassifier

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class StyleGAN2Trainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.config.batch_size = 64

        self.train_set = ImagePoseDataset(config, id_range=(0, 1300))
        self.val_set = ImagePoseDataset(config, id_range=(1300, 2000))

        self.classifier = PoseClassifier(out_dim_azimuth=8, out_dim_elevation=0)

        self.loss = nn.CrossEntropyLoss(reduction='none')

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt = torch.optim.Adam([
            {'params': list(self.classifier.parameters()), 'lr': self.config.lr, 'eps': 1e-8, 'betas': (0.0, 0.99)}
        ])
        return opt

    # training step with pose cross-entropy loss (only azimuth for cars)
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        opt.zero_grad(set_to_none=True)

        images = batch[0]
        azimuth_gt_classes = batch[2]
        elevation_gt_classes = batch[3]

        azimuth_logits, elevation_logits = self.classifier(images)

        azimuth_loss = self.loss(azimuth_logits, azimuth_gt_classes).mean()

        loss = azimuth_loss
        self.manual_backward(loss)

        opt.step()

        self.log("azimuth_loss", azimuth_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    # validation step with pose cross-entropy loss
    def validation_step(self, batch, batch_idx):
        images = batch[0]
        azimuth_gt_classes = batch[2]
        elevation_gt_classes = batch[3]

        azimuth_logits, elevation_logits = self.classifier(images)

        azimuth_loss = self.loss(azimuth_logits, azimuth_gt_classes).mean()

        loss = azimuth_loss

        self.log("val_azimuth_loss", azimuth_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    # placeholder
    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        (Path("runs") / self.config.experiment / "checkpoints").mkdir(exist_ok=True)

    # create dataloader
    def train_dataloader(self):
        return DataLoader(self.train_set, self.config.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    # create dataloader
    def val_dataloader(self):
        return DataLoader(self.val_set, self.config.batch_size, shuffle=False, drop_last=True, num_workers=self.config.num_workers)


@hydra.main(config_path='../config', config_name='car_pose')
def main(config):
    trainer = create_trainer("ImagePose", config)
    model = StyleGAN2Trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()