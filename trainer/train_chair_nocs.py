import math
import shutil
from pathlib import Path
import os

import torch
import torch.nn as nn
import numpy as np
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset.image_nocs import ImageNocsDataset
from dataset.image_nocs_scannet import ImageNocsScannetDataset
from trainer import create_trainer
from util.timer import Timer

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
        self.config.batch_size = 32

        self.train_set = ImageNocsScannetDataset(config, scene_id_range=(0, 480))
        self.val_set = ImageNocsScannetDataset(config, scene_id_range=(480, 800))

        self.model = smp.create_model(
            arch="Unet",
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=4,
            classes=3
        )

        self.loss = nn.MSELoss(reduction='none')

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt = torch.optim.Adam([
            {'params': list(self.model.parameters()), 'lr': self.config.lr, 'eps': 1e-8, 'betas': (0.0, 0.99)}
        ])
        return opt

    # training step with NOCs MSE loss
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        opt.zero_grad(set_to_none=True)

        images = batch[0]
        nocs = batch[2]

        pred_nocs = self.model(images)

        loss = self.loss(pred_nocs, nocs).mean()
        self.manual_backward(loss)

        opt.step()

        self.log("loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    # validation step with NOCs MSE loss
    def validation_step(self, batch, batch_idx):
        images = batch[0]
        nocs = batch[2]

        pred_nocs = self.model(images)

        loss = self.loss(pred_nocs, nocs).mean()

        self.log("val_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    # placeholder
    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        (Path("runs") / self.config.experiment / "checkpoints").mkdir(exist_ok=True)

    # create dataloader
    def train_dataloader(self):
        return DataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    # create dataloader
    def val_dataloader(self):
        return DataLoader(self.val_set, self.config.batch_size, shuffle=False, drop_last=True, num_workers=self.config.num_workers)

    # inference loop to store predicted NOCs
    def inference(self, save_dir=None):
        if save_dir is None:
            SAVE_DIR = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/08021853_ImageLatent_chairs_shapenet_nocs_effnetb4_lr0.0003/images'
        else:
            SAVE_DIR = save_dir
        os.makedirs(SAVE_DIR, exist_ok=True)
        for iter_idx, batch in enumerate(self.val_dataloader()):
            images = batch[0]
            nocs = batch[2]
            pred_nocs = self.model(images)

            save_image(images[:, :3, :, :].cpu().detach(),
                       os.path.join(SAVE_DIR, f"input_{iter_idx}.jpg"), value_range=(0, 1), normalize=True)
            save_image(nocs.cpu().detach(),
                       os.path.join(SAVE_DIR, f"gt_{iter_idx}.jpg"), value_range=(-1, 1), normalize=True)
            save_image(pred_nocs.cpu().detach(),
                       os.path.join(SAVE_DIR, f"pred_{iter_idx}.jpg"), value_range=(-1, 1), normalize=True)

    # inference loop for images from folder
    def inference_for_gt_prediction(self, input_dir=None, save_dir=None):

        from torchvision import transforms
        from PIL import Image
        from tqdm import tqdm

        if input_dir is None:
            INPUT_DIR = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/temp_chair_test_scannet_store_nocs/images/000000'
        else:
            INPUT_DIR = input_dir
        gt_files = [x for x in os.listdir(INPUT_DIR) if x.startswith('gt_')]
        if save_dir is None:
            SAVE_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/chair_scannet_aligned_EVAL'
        else:
            SAVE_DIR = save_dir
        os.makedirs(SAVE_DIR, exist_ok=True)

        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()
        ])

        for image_name in tqdm(gt_files):
            img_id = image_name.split('.')[0]
            gt_image = Image.open(os.path.join(INPUT_DIR, image_name))
            # gt_image = Image.open(os.path.join(INPUT_DIR, image_name)).crop((2, 2, 514, 514))
            # gt_image = Image.open(os.path.join(INPUT_DIR, image_name)).crop((2058, 2, 2570, 514))

            gt_image = preprocess(gt_image)
            mask = torch.where((gt_image[0] >= 0.92) & (gt_image[1] >= 0.92) & (gt_image[2] >= 0.92), 0, 1)
            mask_coords = torch.where(mask == 0)
            gt_image = torch.cat([gt_image, mask[None, ...]], dim=0)
            gt_image = gt_image[None, ...]
            gt_nocs = self.model(gt_image)
            gt_nocs[:, :, mask_coords[0], mask_coords[1]] = 0

            save_image(gt_image[:, :3, :, :].cpu().detach(),
                       os.path.join(SAVE_DIR, f"input_{img_id}.jpg"), value_range=(0, 1), normalize=True)
            save_image(gt_image[:, 3:, :, :].cpu().detach(),
                       os.path.join(SAVE_DIR, f"mask_{img_id}.jpg"), value_range=(0, 1), normalize=True)
            save_image(gt_nocs.cpu().detach(),
                       os.path.join(SAVE_DIR, f"nocs_{img_id}.jpg"), value_range=(-1, 1), normalize=True)
            np.save(os.path.join(SAVE_DIR, f"nocs_{img_id}.npy"),
                    gt_nocs.cpu().detach().numpy())


@hydra.main(config_path='../config', config_name='chair_nocs')
def main(config):
    if config.mode == 'training':
        trainer = create_trainer("ImageNOCs", config)
        model = StyleGAN2Trainer(config)
        trainer.fit(model)

    elif config.mode == 'store':
        CHECKPOINT = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/18022129_ImageLatent_chairs_scannet_nocs_effnetb4_lr0.0003/checkpoints/_epoch=6.ckpt'
        config.batch_size = 8

        model = StyleGAN2Trainer(config).load_from_checkpoint(CHECKPOINT)

        # model.inference(save_dir=config.save_dir)
        model.inference_for_gt_prediction(save_dir=config.save_dir, input_dir=config.input_dir)

if __name__ == '__main__':
    main()