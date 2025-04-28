import math
import shutil
from pathlib import Path
import os
import json
import warnings
warnings.filterwarnings('ignore')
import trimesh
from scipy import ndimage

import torch
from torch import nn as nn
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
import sys
sys.path.append('/rhome/abokhovkin/projects/clean-fid')
from cleanfid import fid

from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset, get_car_views_nonoise
from dataset import to_vertex_colors_scatter, to_vertex_features_scatter, GraphDataLoader, to_device
from dataset import Collater
from model.augment import AugmentPipe
from model.differentiable_renderer import DifferentiableRenderer, DifferentiableMLPRenderer, DifferentiableP3DRenderer, ShapeNetCarsRenderer
from model.graph import TwinGraphEncoder
from model.graph_generator_u_deep_features_tuned import Generator
from model.discriminator import DiscriminatorProgressive
from model.loss import PathLengthPenalty, compute_gradient_penalty
from trainer import create_trainer
from util.timer import Timer
from trainer.fid_score_per_image_multiview import compute_multiview_score
from trainer.fid_score_per_image import compute_score
import time
import json

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

torch.autograd.set_detect_anomaly(True)


class StyleGAN2Trainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.config.batch_size = 1
        config.batch_size = 1
        self.config.views_per_sample = 6
        config.views_per_sample = 6
        self.config.num_eval_images = 2048
        config.num_eval_images = 2048
        config.random_bg = 'white'
        # config.texopt = True
        # self.config.texopt = True

        self.train_set = FaceGraphMeshDataset(config)
        self.val_set = FaceGraphMeshDataset(config, config.num_eval_images)

        # ONLY FOR DEBUG, created limited-size dataset
        # self.train_set = FaceGraphMeshDataset(config, 12)
        # self.val_set = FaceGraphMeshDataset(config, 12)

        # allocate modules
        self.E = TwinGraphEncoder(self.train_set.num_feats, 1, layer_dims=(64, 128, 128, 256, 256, 256, 384, 384))
        # self.E = TwinGraphEncoder(self.train_set.num_feats, 1)
        self.G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max)
        self.D = DiscriminatorProgressive(256, 3, w_num_layers=config.num_mapping_layers, mbstd_on=config.mbstd_on, channel_base=self.config.d_channel_base)
        self.D_faces = DiscriminatorProgressive(256, 3, w_num_layers=config.num_mapping_layers, mbstd_on=config.mbstd_on, channel_base=self.config.d_channel_base)

        self.R = DifferentiableMLPRenderer(self.config.render_size, small_shader=self.config.small_shader,
                                           residual=self.config.residual, tanh=self.config.tanh_render,
                                           use_act=self.config.use_act_render, mode='car')
        self.R_faces = None
        self.R_color = DifferentiableP3DRenderer(self.config.render_size)
        self.R_shapenet = ShapeNetCarsRenderer(self.config.render_size)
        self.register_buffer("state", torch.zeros(1).long())

        # create ADA augmentation
        self.augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size, config.views_per_sample, config.colorspace)
        # self.augment_pipe = AugmentPipe(0.15, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size, config.views_per_sample, config.colorspace)

        # set additional hyperparameters
        self.grid_z = torch.randn(config.num_eval_images, self.config.latent_dim)
        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.ema = None
        self.ema_r = None

        self.stable_global_step = 0


        # self.config.resume_ema = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/18021548_StyleGAN23D-CompCars_mlpbig_bigdisc_twodisc_lrr5-4_lrmd55-5_lrd1-3_lrg1-3_lre1-4_mlp2x_2gpu_noglobfeat_norm/checkpoints/ema_000023864.pth'
        # self.config.resume_ema_r = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/18021548_StyleGAN23D-CompCars_mlpbig_bigdisc_twodisc_lrr5-4_lrmd55-5_lrd1-3_lrg1-3_lre1-4_mlp2x_2gpu_noglobfeat_norm/checkpoints/ema_r_000023864.pth'

        # if config.load_checkpoint:
            # orig Texturify checkpoint
            # pretrained_dict = torch.load(
            #     '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/05041847_StyleGAN23D-CompCars_progressive128x2_clip_fg3bgg-lrd1g14-v8m8-1K_256/checkpoints/_epoch=409.ckpt'
            # )
            # Texturify with new diff. renderer
            # pretrained_dict = torch.load(
            #     '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/14102322_StyleGAN23D-CompCars_mlpshader_res_imgsize256-256_bigdisc_shapeposenc_twodiscload_lrr0_lrd1-3_lrg1-3_lre1-4/checkpoints/_epoch=33.ckpt'
            # )
            # Texturify MLP, 2x and detached
            # pretrained_dict = torch.load(
            #     '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/25101834_StyleGAN23D-CompCars_mlpbig_bigdisc_twodisc_lrr4-4_lrmd5-4_lrd1-3_lrg1-3_lre1-4_noact_stablegs_mlp2x_detachmlp_2gpu/checkpoints/_epoch=17.ckpt'
            # )

            # E_dict = self.E.state_dict()
            # E_dict_update = {k[2:]: v for k, v in pretrained_dict['state_dict'].items()
            #                        if k[2:] in E_dict and
            #                        pretrained_dict['state_dict'][k].shape == E_dict[k[2:]].shape}
            # E_dict.update(E_dict_update)
            # print('Updated keys (encoder):', len(E_dict_update))
            # print('Not loaded keys:', list(set(E_dict.keys()) - set(E_dict_update.keys())))
            # self.E.load_state_dict(E_dict)
            #
            # G_dict = self.G.state_dict()
            # G_dict_update = {k[2:]: v for k, v in pretrained_dict['state_dict'].items()
            #                  if k[2:] in G_dict and
            #                  pretrained_dict['state_dict'][k].shape == G_dict[k[2:]].shape}
            # G_dict.update(G_dict_update)
            # print('Updated keys (generator):', len(G_dict_update))
            # print('Not loaded keys:', list(set(G_dict.keys()) - set(G_dict_update.keys())))
            # self.G.load_state_dict(G_dict)
            #
            # D_dict = self.D.state_dict()
            # D_dict_update = {k[2:]: v for k, v in pretrained_dict['state_dict'].items()
            #                  if k[2:] in D_dict and
            #                  pretrained_dict['state_dict'][k].shape == D_dict[k[2:]].shape}
            # D_dict.update(D_dict_update)
            # print('Updated keys (discriminator):', len(D_dict_update))
            # print('Not loaded keys:', list(set(D_dict.keys()) - set(D_dict_update.keys())))
            # self.D.load_state_dict(D_dict)
            #
            # D_faces_dict = self.D_faces.state_dict()
            # D_faces_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
            #                  if k[8:] in D_faces_dict and
            #                  pretrained_dict['state_dict'][k].shape == D_faces_dict[k[8:]].shape}
            # D_faces_dict.update(D_faces_dict_update)
            # print('Updated keys (discriminator):', len(D_faces_dict_update))
            # print('Not loaded keys:', list(set(D_faces_dict.keys()) - set(D_faces_dict_update.keys())))
            # self.D_faces.load_state_dict(D_faces_dict)
            #
            # R_dict = self.R.state_dict()
            # R_dict_update = {k[2:]: v for k, v in pretrained_dict['state_dict'].items()
            #                  if k[2:] in R_dict and
            #                  pretrained_dict['state_dict'][k].shape == R_dict[k[2:]].shape}
            # R_dict.update(R_dict_update)
            # print('Updated keys (renderer):', len(R_dict_update))
            # print('Not loaded keys:', list(set(R_dict.keys()) - set(R_dict_update.keys())))
            # self.R.load_state_dict(R_dict)


    def configure_optimizers(self):

        g_opt = torch.optim.Adam([
            {'params': list(self.G.parameters()), 'lr': self.config.lr_g, 'betas': (0.0, 0.99), 'eps': 1e-8},
            {'params': list(self.E.parameters()), 'lr': self.config.lr_e, 'eps': 1e-8, 'weight_decay': 1e-4}
        ])
        r_opt = torch.optim.AdamW([
            {'params': list(self.R.parameters()), 'lr': self.config.lr_r, 'betas': (0.5, 0.99), 'eps': 1e-8, 'weight_decay': 1e-2},
            {'params': [self.R.aux_latent], 'lr': 1e-3, 'betas': (0.0, 0.99), 'eps': 1e-8}
        ])
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_mlp_d, betas=(0.0, 0.99), eps=1e-8)
        d_faces_opt = torch.optim.Adam(self.D_faces.parameters(), lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8)

        return g_opt, d_faces_opt, r_opt, d_opt

    # synthesis forward step
    def forward(self, batch, limit_batch_size=False):
        z = self.latent(limit_batch_size)
        w, z_out = self.get_mapped_latent(z, 0.9)
        fake, all_face_colors, feature, tuned_feature = self.G.synthesis(batch['graph_data'], w, batch['shape'])
        return fake, w, feature, tuned_feature, z_out

    # Generator step
    def g_step(self, batch, batch_idx, resolution, alpha):
        g_opt = self.optimizers()[0]
        g_opt.zero_grad(set_to_none=True)
        r_opt = self.optimizers()[2]
        r_opt.zero_grad(set_to_none=True)
        fake, w, feature, tuned_feature, z_out = self.forward(batch)

        rendered_faces_fake = self.render_color(batch,
                                                face_color=fake,
                                                resolution=resolution)[:, :3, :, :]
        p_faces_fake = self.D_faces(self.augment_pipe(rendered_faces_fake), alpha)
        gen_faces_loss = torch.nn.functional.softplus(-p_faces_fake).mean()

        rendered_fake, _, _ = self.render(tuned_feature, batch,
                                    face_color=fake,
                                    use_shape_features=self.config.render_shape_features,
                                    resolution=2 * resolution,
                                    style_latent=z_out)
        rendered_fake = rendered_fake[:, :3, :, :]
        p_fake = self.D(self.augment_pipe(rendered_fake), alpha)
        gen_loss = torch.nn.functional.softplus(-p_fake).mean()

        self.manual_backward(gen_loss)
        # self.manual_backward(gen_loss + gen_faces_loss)

        step(g_opt, self.G)
        step(r_opt, self.R)

        log_gen_loss = gen_loss.item()
        log_gen_faces_loss = gen_faces_loss.item()

        self.log("G", log_gen_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("G_faces", log_gen_faces_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    # Generator regularization step (PLP)
    def g_regularizer(self, batch, batch_idx, resolution):
        g_opt = self.optimizers()[0]
        for idx in range(len(batch['shape'])):
            batch['shape'][idx] = batch['shape'][idx].detach()
        g_opt.zero_grad(set_to_none=True)
        fake, w, feature, tuned_feature, z_out = self.forward(batch)

        rendered_fake, _, _ = self.render(tuned_feature, batch,
                                        face_color=fake,
                                        use_shape_features=self.config.render_shape_features,
                                        resolution=resolution,
                                        style_latent=z_out)
        rendered_fake = rendered_fake[:, :3, :, :]
        plp = self.path_length_penalty(rendered_fake, w)

        if not torch.isnan(plp):
            gen_loss = self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
            self.log("rPLP", plp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            
            self.manual_backward(gen_loss)
            # self.manual_backward(gen_faces_loss + gen_loss)
            
            step(g_opt, self.G)

    # Discriminator step
    def d_step(self, batch, batch_idx, resolution, alpha):
        d_faces_opt = self.optimizers()[1]
        d_faces_opt.zero_grad(set_to_none=True)
        d_opt = self.optimizers()[3]
        d_opt.zero_grad(set_to_none=True)

        fake, _, feature, tuned_feature, z_out = self.forward(batch)

        rendered_faces_fake = self.render_color(batch,
                                                face_color=fake.detach(),
                                                resolution=resolution)[:, :3, :, :]
        p_faces_fake = self.D_faces(self.augment_pipe(rendered_faces_fake), alpha)
        fake_faces_loss = torch.nn.functional.softplus(p_faces_fake).mean()
        # self.manual_backward(fake_faces_loss)

        p_faces_real = self.D_faces(self.augment_pipe(self.train_set.get_color_bg_real(batch)), alpha)
        self.augment_pipe.accumulate_real_sign(p_faces_real.sign().detach())
        real_faces_loss = torch.nn.functional.softplus(-p_faces_real).mean()
        # self.manual_backward(real_faces_loss)

        rendered_fake, _, _ = self.render(tuned_feature.detach(), batch,
                                    face_color=fake.detach(),
                                    use_shape_features=self.config.render_shape_features,
                                    detach=True,
                                    resolution=2 * resolution,
                                    style_latent=z_out)
        rendered_fake = rendered_fake[:, :3, :, :]
        p_fake = self.D(self.augment_pipe(rendered_fake), alpha)
        fake_loss = torch.nn.functional.softplus(p_fake).mean()

        self.manual_backward(fake_loss)
        # self.manual_backward(fake_loss + fake_faces_loss)

        p_real = self.D(self.augment_pipe(self.train_set.get_color_bg_real(batch, highres=True)), alpha)
        self.augment_pipe.accumulate_real_sign(p_real.sign().detach())
        real_loss = torch.nn.functional.softplus(-p_real).mean()
        
        self.manual_backward(real_loss)
        # self.manual_backward(real_loss + real_faces_loss)

        step(d_faces_opt, self.D_faces)
        step(d_opt, self.D)

        p_fake_log = p_faces_fake.mean()
        p_real_log = p_faces_real.mean()

        self.log("D_real", real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("D_fake", fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        disc_loss = real_loss + fake_loss
        self.log("D", disc_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        self.log("D_faces_real", real_faces_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("D_faces_fake", fake_faces_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        disc_faces_loss = real_faces_loss + fake_faces_loss
        self.log("D_faces", disc_faces_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("p_fake", p_fake_log, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("p_real", p_real_log, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    # Discriminator regularization step (gradient penalty)
    def d_regularizer(self, batch, batch_idx, alpha):
        d_faces_opt = self.optimizers()[1]
        d_faces_opt.zero_grad(set_to_none=True)
        d_opt = self.optimizers()[3]
        d_opt.zero_grad(set_to_none=True)
        image = self.train_set.get_color_bg_real(batch)
        image.requires_grad_()
        image2 = image.clone().detach().requires_grad_()

        p_faces_real = self.D_faces(self.augment_pipe(image, True), alpha)
        gp_faces = compute_gradient_penalty(image, p_faces_real)
        disc_faces_loss = self.config.lambda_gp * gp_faces * self.config.lazy_gradient_penalty_interval

        p_real = self.D(self.augment_pipe(image2, True), alpha)
        gp = compute_gradient_penalty(image2, p_real)
        disc_loss = self.config.lambda_gp * gp * self.config.lazy_gradient_penalty_interval
        
        self.manual_backward(disc_loss)
        # self.manual_backward(disc_loss + disc_faces_loss)
        
        step(d_faces_opt, self.D_faces)
        step(d_opt, self.D)

        self.log("rGP", gp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("rGP_faces", gp_faces, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    # general diff. rendering using PyTorch3D (Neural Field)
    def render(self, face_feature, batch, face_color=None, use_shape_features=False, style_latent=None,
               detach=False, use_color_bg=True, resolution=None, norm_params=None, camera_delta_angle=None,
               transform_matrices=None, camera_delta_angle_array=None, render_nocs=False, render_sil=False,
               only_first_view=False, camera_delta_elevation_array=None):
        if use_shape_features:
            if detach:
                shape_features = to_vertex_features_scatter(batch['shape'][0].detach(), batch)
            else:
                shape_features = to_vertex_features_scatter(batch['shape'][0], batch)
        if face_color is None:
            rendered_color, image_nocs, sil = self.R.render(batch['vertices'],
                                           batch['indices'],
                                           to_vertex_features_scatter(face_feature, batch),
                                           batch["ranges"].cpu(),
                                           style_latent=style_latent,
                                           color_bg=batch['bg'] if use_color_bg else None,
                                           shape=shape_features if use_shape_features else None,
                                           resolution=self.config.render_size if resolution is None else resolution,
                                           views_per_shape=self.config.views_per_sample,
                                           norm_params=norm_params,
                                           camera_delta_angle=camera_delta_angle,
                                           transform_matrices=transform_matrices,
                                           camera_delta_angle_array=camera_delta_angle_array,
                                           camera_delta_elevation_array=camera_delta_elevation_array,
                                           render_nocs=render_nocs,
                                           render_sil=render_sil,
                                           only_first_view=only_first_view
                                           )
        else:
            rendered_color, image_nocs, sil = self.R.render(batch['vertices'],
                                           batch['indices'],
                                           to_vertex_features_scatter(face_feature, batch),
                                           batch["ranges"].cpu(),
                                           to_vertex_colors_scatter(face_color, batch),
                                           style_latent=style_latent,
                                           color_bg=batch['bg'] if use_color_bg else None,
                                           shape=shape_features if use_shape_features else None,
                                           resolution=self.config.render_size if resolution is None else resolution,
                                           views_per_shape=self.config.views_per_sample,
                                           norm_params=norm_params,
                                           camera_delta_angle=camera_delta_angle,
                                           transform_matrices=transform_matrices,
                                           camera_delta_angle_array=camera_delta_angle_array,
                                           camera_delta_elevation_array=camera_delta_elevation_array,
                                           render_nocs=render_nocs,
                                           render_sil=render_sil,
                                           only_first_view=only_first_view
                                           )
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if resolution is None:
            resolution = self.config.image_size
        if self.config.render_size != resolution:
            ret_val = torch.nn.functional.interpolate(ret_val, (resolution, resolution), mode='bilinear', align_corners=True)
        return ret_val, image_nocs, sil

    # general diff. rendering using PyTorch3D (per-face colors)
    def render_color(self, batch, face_color, use_color_bg=True, resolution=None, camera_delta_angle_array=None, camera_delta_elevation_array=None,
                     norm_params=None):
        rendered_color = self.R_color.render(batch['vertices'],
                                             batch['indices'],
                                             batch["ranges"].cpu(),
                                             to_vertex_colors_scatter(face_color, batch),
                                             color_bg=batch['bg'] if use_color_bg else None,
                                             resolution=self.config.render_size,
                                             camera_delta_angle_array=camera_delta_angle_array,
                                             camera_delta_elevation_array=camera_delta_elevation_array,
                                             norm_params=norm_params)
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if resolution is None:
            resolution = self.config.image_size
        if self.config.render_size != resolution:
            ret_val = torch.nn.functional.interpolate(ret_val, (resolution, resolution), mode='bilinear',
                                                      align_corners=True)
        return ret_val

    # old NVDiffRast renderer for per-face colors
    def render_faces(self, face_colors, batch, resolution=None, use_bg_color=True):
        rendered_color = self.R_faces.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), batch['bg'] if use_bg_color else None, resolution=self.config.render_size)
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if resolution is None:
            resolution = self.config.image_size
        if self.config.render_size != resolution:
            ret_val = torch.nn.functional.interpolate(ret_val, (resolution, resolution), mode='bilinear', align_corners=True)
        return ret_val

    # ShapeNet renderer
    def render_shapenet(self, obj_id=None, resolution=None, camera_delta_angle=None, camera_elevation_angle=None,
                        camera_dist=None):
        rendered_color = self.R_shapenet.render(obj_id=obj_id,
                                                camera_delta_angle=camera_delta_angle,
                                                camera_elevation_angle=camera_elevation_angle,
                                                camera_dist=camera_dist)
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if resolution is None:
            resolution = self.config.image_size
        if self.config.render_size != resolution:
            ret_val = torch.nn.functional.interpolate(ret_val, (resolution, resolution), mode='bilinear',
                                                      align_corners=True)
        ret_val = ret_val[:, :3, :, :]

        return ret_val

    # full training step for Generator, Discriminator and Renderer
    def training_step(self, batch, batch_idx):
        self.set_shape_codes(batch)
        resolution, alpha = self.get_resolution_and_alpha()
        set_real_sample_resolution(batch, resolution)

        # optimize generator
        self.g_step(batch, batch_idx, resolution, alpha)

        if not self.config.freeze_texturify:
            if self.config.lazy_path_penalty_interval != -1 and self.stable_global_step > self.config.lazy_path_penalty_after and (self.stable_global_step + 1) % self.config.lazy_path_penalty_interval == 0:
                self.g_regularizer(batch, batch_idx, resolution)

        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.R.parameters(), max_norm=1.0)

        self.ema.update(self.G.parameters())
        self.ema_r.update(self.R.parameters())

        # optimize discriminator
        self.d_step(batch, batch_idx, resolution, alpha)

        if (self.stable_global_step + 1) % self.config.lazy_gradient_penalty_interval == 0:
            self.d_regularizer(batch, batch_idx, alpha)

        self.execute_ada_heuristics()
        self.update_state()

        self.stable_global_step += 1

        self.log("stable_global_step", self.stable_global_step, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    # update ADA augmentation parameters
    def execute_ada_heuristics(self):
        if (self.stable_global_step + 1) % self.config.ada_interval == 0:
            self.augment_pipe.heuristic_update()
        self.log("aug_p", self.augment_pipe.p.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    # placeholder
    def validation_step(self, batch, batch_idx):
        pass

    # epoch end full validation check (save EMA, update EMA, save generated images (FID / store), compute FID / KID)
    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        (Path("runs") / self.config.experiment / "checkpoints").mkdir(exist_ok=True)
        torch.save(self.ema, Path("runs") / self.config.experiment / "checkpoints" / f"ema_{self.stable_global_step:09d}.pth")
        torch.save(self.ema_r, Path("runs") / self.config.experiment / "checkpoints" / f"ema_r_{self.stable_global_step:09d}.pth")
        with Timer("export_grid"):
            odir_real, odir_fake, odir_samples, odir_grid, odir_meshes, odir_faces_fake = self.create_directories()
            # self.export_grid("", odir_grid, None)
            self.ema.store(self.G.parameters())
            self.ema_r.store(self.R.parameters())
            self.ema.copy_to([p for p in self.G.parameters() if p.requires_grad])
            self.ema_r.copy_to([p for p in self.R.parameters() if p.requires_grad])
            self.export_grid("ema_", odir_grid, odir_fake)
            # self.export_faces_grid("ema_", odir_grid, odir_faces_fake)
            # self.export_mesh(odir_meshes)

        len_loader = len(self.val_dataloader())
        with Timer("export_samples"):
            latents = self.grid_z.split(self.config.batch_size)
            for iter_idx, batch in enumerate(self.val_dataloader()):
                if iter_idx > len_loader - 1:
                    continue
                batch = to_device(batch, self.device)
                self.set_shape_codes(batch)
                shape = batch['shape']
                real_render = batch['real'].cpu()
                pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(batch['graph_data'],
                                                                                     latents[iter_idx % len(latents)].to(self.device),
                                                                                     shape,
                                                                                     noise_mode='const')
                fake_faces_render = self.render_faces(pred_colors, batch, use_bg_color=True).cpu()
                fake_faces_render = self.train_set.cspace_convert_back(fake_faces_render)
                save_image(fake_faces_render, odir_samples / f"faces_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

                fake_render, _, _ = self.render(tuned_feature, batch,
                                          face_color=pred_colors,
                                          use_shape_features=self.config.render_shape_features,
                                          use_color_bg=True,
                                          style_latent=z_out)
                fake_render = fake_render.cpu()
                real_render = self.train_set.cspace_convert_back(real_render)
                real_render = real_render.detach()

                fake_render = self.train_set.cspace_convert_back(fake_render[:, :3, ...])
                save_image(real_render, odir_samples / f"real_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                save_image(fake_render, odir_samples / f"fake_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                for batch_idx in range(real_render.shape[0]):
                    save_image(real_render[batch_idx], odir_real / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)

                color_render = self.render_color(batch,
                                                 face_color=pred_colors,
                                                 use_color_bg=True)[:, :3, :, :]
                color_render = self.train_set.cspace_convert_back(color_render[:, :3, ...])
                save_image(color_render, odir_samples / f"facescolor_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

        self.ema.restore([p for p in self.G.parameters() if p.requires_grad])
        self.ema_r.restore([p for p in self.R.parameters() if p.requires_grad])
        fid_score = fid.compute_fid(str(odir_real), str(odir_fake), device=self.device, num_workers=0)
        print(f'FID: {fid_score:.5f}')
        # fid_faces_score = fid.compute_fid(str(odir_real), str(odir_faces_fake), device=self.device, num_workers=0)
        # print(f'FID (faces): {fid_faces_score:.5f}')
        kid_score = fid.compute_kid(str(odir_real), str(odir_fake), device=self.device, num_workers=0)
        print(f'KID: {kid_score:.5f}')
        # kid_faces_score = fid.compute_kid(str(odir_real), str(odir_faces_fake), device=self.device, num_workers=0)
        # print(f'KID (faces): {kid_faces_score:.5f}')
        self.log(f"fid", fid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True)
        self.log(f"kid", kid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True)
        # self.log(f"fid (faces)", fid_faces_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True)
        # self.log(f"kid (faces)", kid_faces_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True)
        shutil.rmtree(odir_real.parent)

    # store images for sanity check (generated images, gt images, augmented images)
    @rank_zero_only
    def sanity_check_on_step(self):
        odir_sanity_dump = self.create_sanity_directories()
        with torch.no_grad():
            with Timer("sanity_check_on_step"):
                latents = self.grid_z.split(self.config.batch_size)
                for iter_idx, batch in enumerate(self.val_dataloader()):
                    if iter_idx > 8:
                        break
                    batch = to_device(batch, self.device)
                    self.set_shape_codes(batch)
                    shape = batch['shape']
                    pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(batch['graph_data'],
                                                                                         latents[iter_idx % len(latents)].to(self.device),
                                                                                         shape,
                                                                                         noise_mode='const')

                    fake_render = self.render(tuned_feature, batch,
                                              face_color=pred_colors,
                                              use_shape_features=self.config.render_shape_features,
                                              use_color_bg=False,
                                              style_latent=z_out)
                    fake_render_aug = self.augment_pipe(fake_render[:, :3, ...])
                    fake_render = self.train_set.cspace_convert_back(fake_render[:, :3, ...])
                    fake_render_aug = self.train_set.cspace_convert_back(fake_render_aug[:, :3, ...])
                    save_image(fake_render, odir_sanity_dump / f"fake_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                    save_image(fake_render_aug, odir_sanity_dump / f"fake_aug_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

                    color_render = self.render_color(batch,
                                                     face_color=pred_colors,
                                                     use_color_bg=False)[:, :3, :, :]
                    color_render_aug = self.augment_pipe(color_render[:, :3, ...])
                    color_render = self.train_set.cspace_convert_back(color_render[:, :3, ...])
                    color_render_aug = self.train_set.cspace_convert_back(color_render_aug[:, :3, ...])
                    save_image(color_render, odir_sanity_dump / f"facescolor_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                    save_image(color_render_aug, odir_sanity_dump / f"facescolor_aug_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

    # StyleGAN mapping latent -> w style code
    def get_mapped_latent(self, z, style_mixing_prob):
        if torch.rand(()).item() < style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.G.mapping.num_ws)
            w1, z_out_1 = self.G.mapping(z[0])
            w2, z_out_2 = self.G.mapping(z[1], skip_w_avg_update=True)
            w1 = w1[:, :cross_over_point, :]
            w2 = w2[:, cross_over_point:, :]
            return torch.cat((w1, w2), dim=1), z_out_1
        else:
            w, z_out = self.G.mapping(z[0])
            return w, z_out

    # generate latent parameters
    def latent(self, limit_batch_size=False):
        batch_size = self.config.batch_size if not limit_batch_size else self.config.batch_size // self.path_length_penalty.pl_batch_shrink
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    # Encoder step
    def set_shape_codes(self, batch):
        code = self.E(batch['x'], batch['graph_data']['ff2_maps'][0], batch['graph_data'])
        batch['shape'] = code

    # set training dataloader
    def train_dataloader(self):
        return GraphDataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True,
                               num_workers=1)

    # set validation dataloader
    def val_dataloader(self):
        return GraphDataLoader(self.val_set, self.config.batch_size, shuffle=False, drop_last=True,
                               num_workers=1, texopt=self.config.texopt)

    # save generated images (Neural Field) into output_dir_fid folder for FID evaluation
    def export_grid(self, prefix, output_dir_vis, output_dir_fid):
        vis_generated_images = []
        grid_loader = iter(GraphDataLoader(self.train_set, batch_size=self.config.batch_size, drop_last=True))
        len_loader = len(grid_loader)
        for iter_idx, z in enumerate(self.grid_z.split(self.config.batch_size)):
            if iter_idx > len_loader - 1:
                continue
            z = z.to(self.device)
            eval_batch = to_device(next(grid_loader), self.device)
            self.set_shape_codes(eval_batch)
            pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(eval_batch['graph_data'], z, eval_batch['shape'], noise_mode='const')
            fake, _, _ = self.render(tuned_feature,
                               eval_batch,
                               face_color=pred_colors,
                               use_shape_features=self.config.render_shape_features,
                               use_color_bg=False,
                               style_latent=z_out)
            fake = fake.cpu()[:, :3, ...]
            fake = self.train_set.cspace_convert_back(fake)
            if output_dir_fid is not None:
                for batch_idx in range(fake.shape[0]):
                    save_image(fake[batch_idx], output_dir_fid / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
            # if iter_idx < self.config.num_vis_images // self.config.batch_size:
            #     vis_generated_images.append(fake)
        torch.cuda.empty_cache()
        # vis_generated_images = torch.cat(vis_generated_images, dim=0)
        # save_image(vis_generated_images, output_dir_vis / f"{prefix}{self.global_step:06d}.png", nrow=int(math.sqrt(vis_generated_images.shape[0])), value_range=(-1, 1), normalize=True)

    # save generated images (per-face colors) into output_dir_fid folder for FID evaluation
    def export_faces_grid(self, prefix, output_dir_vis, output_dir_fid):
        grid_loader = iter(GraphDataLoader(self.train_set, batch_size=self.config.batch_size, drop_last=True))
        len_loader = len(grid_loader)
        for iter_idx, z in enumerate(self.grid_z.split(self.config.batch_size)):
            if iter_idx > len_loader - 1:
                continue
            z = z.to(self.device)
            eval_batch = to_device(next(grid_loader), self.device)
            self.set_shape_codes(eval_batch)
            pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(eval_batch['graph_data'], z, eval_batch['shape'], noise_mode='const')
            fake = self.render_color(eval_batch,
                                     face_color=pred_colors,
                                     use_color_bg=False).cpu()[:, :3, :, :]
            fake = self.train_set.cspace_convert_back(fake)
            if output_dir_fid is not None:
                for batch_idx in range(fake.shape[0]):
                    save_image(fake[batch_idx], output_dir_fid / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
        torch.cuda.empty_cache()

    # export colored meshes (per-face colors only)
    def export_mesh(self, outdir):
        grid_loader = iter(GraphDataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=True))
        for iter_idx, z in enumerate(self.grid_z.split(self.config.batch_size)):
            if iter_idx < self.config.num_vis_meshes // self.config.batch_size:
                z = z.to(self.device)
                eval_batch = to_device(next(grid_loader), self.device)
                self.set_shape_codes(eval_batch)
                generated_colors, all_face_colors = self.G(eval_batch['graph_data'], z, eval_batch['shape'], noise_mode='const')
                generated_colors = self.train_set.cspace_convert_back(generated_colors) * 0.5 + 0.5
                for bidx in range(generated_colors.shape[0] // self.config.num_faces[0]):
                    self.train_set.export_mesh(eval_batch['name'][bidx],
                                               generated_colors[self.config.num_faces[0] * bidx: self.config.num_faces[0] * (bidx + 1)], outdir / f"{eval_batch['name'][bidx]}.obj")

    # create all necessary directories for FID, storing, meshes
    def create_directories(self):
        output_dir_fid_real = Path(f'runs/{self.config.experiment}/fid/real')
        output_dir_fid_fake = Path(f'runs/{self.config.experiment}/fid/fake')
        output_dir_fid_faces_fake = Path(f'runs/{self.config.experiment}/fid/faces_fake')
        output_dir_samples = Path(f'runs/{self.config.experiment}/images/{self.stable_global_step:06d}')
        output_dir_textures = Path(f'runs/{self.config.experiment}/textures/')
        output_dir_meshes = Path(f'runs/{self.config.experiment}/meshes/{self.stable_global_step:06d}')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes, output_dir_fid_faces_fake]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes, output_dir_fid_faces_fake

    # create all necessary directories for inference
    def create_store_directories(self, prefix='0'):
        DIR = Path('/cluster/valinor/abokhovkin/scannet-texturing/data/final_prediction')
        output_dir_fid_real = DIR / Path(f'{prefix}/fid/real')
        output_dir_fid_fake = DIR / Path(f'{prefix}/fid/fake')
        output_dir_samples = DIR / Path(f'{prefix}/images/{self.stable_global_step:06d}')
        output_dir_samples_levels = DIR / Path(f'{prefix}/images_levels/{self.stable_global_step:06d}')
        output_dir_textures = DIR / Path(f'{prefix}/textures/')
        output_dir_meshes = DIR / Path(f'{prefix}/meshes/{self.stable_global_step:06d}')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures,
                     output_dir_meshes, output_dir_samples_levels]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes, output_dir_samples_levels

    # create all necessary temporary directories for debugging
    def create_temp_directories(self, prefix='0'):
        output_dir_fid_real = Path(f'runs/temp_{prefix}/fid/real')
        output_dir_fid_fake = Path(f'runs/temp_{prefix}/fid/fake')
        output_dir_samples = Path(f'runs/temp_{prefix}/images/{self.global_step:06d}')
        output_dir_samples_levels = Path(f'runs/temp_{prefix}/images_levels/{self.global_step:06d}')
        output_dir_textures = Path(f'runs/temp_{prefix}/textures/')
        output_dir_meshes = Path(f'runs/temp_{prefix}/meshes/{self.global_step:06d}')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes, output_dir_samples_levels]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes, output_dir_samples_levels

    # create sanity check directories
    def create_sanity_directories(self):
        output_dir_sanity_dump = Path(f'runs/{self.config.experiment}/sanity/images/{self.stable_global_step:06d}')
        for odir in [output_dir_sanity_dump]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_sanity_dump

    # allocate EMA and face renderer before training start
    def on_train_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
            self.ema_r = ExponentialMovingAverage(self.R.parameters(), 0.995)
        if self.R_faces is None:
            self.R_faces = DifferentiableRenderer(self.config.image_size, "bounds", self.config.colorspace)
        if self.config.resume_ema is not None:
            self.ema = torch.load(self.config.resume_ema, map_location=self.device)
            self.ema_r = torch.load(self.config.resume_ema_r, map_location=self.device)

    # check if EMA and face renderer was not allocated
    def on_validation_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
            self.ema_r = ExponentialMovingAverage(self.R.parameters(), 0.995)
        if self.R_faces is None:
            self.R_faces = DifferentiableRenderer(self.config.image_size, "bounds", self.config.colorspace)

    # update state for progressive resolution
    def update_state(self):
        if not self.state.item() == (len(self.config.progressive_switch) - 1):
            if self.stable_global_step > self.config.progressive_switch[self.state.item()]:
                self.state += 1
        self.log(f"d_state", float(self.state.item()), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    # get current resolution and alpha according to the state
    def get_resolution_and_alpha(self):
        last_update = 0
        if self.state.item() != 0:
            last_update = self.config.progressive_switch[self.state.item() - 1]
        alpha = max(0, min(1, (self.stable_global_step - last_update) / self.config.alpha_switch[self.state.item()]))
        resolution = 2 ** (self.state.item() + self.config.progressive_start_res)
        self.log(f"d_alpha", float(alpha), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"d_resolution", float(resolution), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return resolution, alpha

    # crop texture randomly from an input image, biased towards image mask center
    def crop_texture(self, image, background_value=1):

        bsize, num_channels, height, width = image.shape

        image_mask = (image[:, 0, ...] != background_value) | (image[:, 1, ...] != background_value) | (
                image[:, 2, ...] != background_value)
        image_mask_coords = torch.stack(torch.where(image_mask)[1:]).T.float()
        bck_mask = (image[:, 0, ...] == background_value) & (image[:, 1, ...] == background_value) & (
                image[:, 2, ...] == background_value)
        bck_mask_coords = torch.stack(torch.where(bck_mask)[1:]).T
        image_center = image_mask_coords.mean(axis=0)
        random_shift = (torch.rand((bsize, 2)) * 0.1 * height).to(image_center.device)
        image_center = image_center + random_shift
        image_center = image_center.int()

        min_x_side = min(image_center[:, 1], width - image_center[:, 1])
        min_y_side = min(image_center[:, 0], height - image_center[:, 0])
        min_side = min(min_x_side, min_y_side)
        texture_side = int(min(min_side, 175 / 1024.0 * height))

        if texture_side < 10:
            texture_side = int(175 / 1024.0 * height)
            image_center = torch.LongTensor([[128, 128]]).to(image.device)

        texture_y_bot = image_center[:, 0] + texture_side
        texture_y_top = image_center[:, 0] - texture_side
        texture_x_left = image_center[:, 1] - texture_side
        texture_x_right = image_center[:, 1] + texture_side

        texture_crop = image[:, :, texture_y_top:texture_y_bot, texture_x_left:texture_x_right]
        return texture_crop

    # crop texture based on the closest NOC values
    def crop_texture_cover(self, image, background_value=1, coords=None,
                           use_nocs=False, nocs_map=None, nocs_value=None, target_texture_side=None):

        bsize, num_channels, height, width = image.shape

        image_mask = (image[:, 0, ...] != background_value) | (image[:, 1, ...] != background_value) | (
                    image[:, 2, ...] != background_value)
        image_mask = image_mask.cpu().detach().numpy().astype('uint8')
        # image_mask = ndimage.binary_erosion(image_mask[0], structure=np.ones((40, 40)))
        image_mask = image_mask[0]
        image_mask = torch.LongTensor(image_mask)

        if coords is None:

            image_mask_coords = torch.stack(torch.where(image_mask)).T.int()
            if len(image_mask_coords) == 0:
                image_center = torch.LongTensor([128, 128]).to(image.device)
            else:
                image_center = image_mask_coords[np.random.choice(len(image_mask_coords))]

            min_x_side = min(image_center[1], width - image_center[1])
            min_y_side = min(image_center[0], height - image_center[0])
            min_side = min(min_x_side, min_y_side)
            texture_side = int(min(min_side, 64 / 1024.0 * height))

            if texture_side < 10:
                texture_side = int(100 / 1024.0 * height)
                image_center = torch.LongTensor([128, 128]).to(image.device)

            if use_nocs:    
                nocs_value = nocs_map[0, :3, image_center[0], image_center[1]]

            texture_y_bot = image_center[0] + texture_side
            texture_y_top = image_center[0] - texture_side
            texture_x_left = image_center[1] - texture_side
            texture_x_right = image_center[1] + texture_side
        else:
            if not use_nocs:
                texture_y_top = coords[0]
                texture_y_bot = coords[1]
                texture_x_left = coords[2]
                texture_x_right = coords[3]
            else:
                nocs_mask = nocs_map[:, 3:]
                nocs_mask_coords = torch.where(nocs_mask)
                if len(nocs_mask_coords[2]) != 0:
                    nocs_map_values = nocs_map[:, :3][:, :, nocs_mask_coords[2], nocs_mask_coords[3]]
                    min_idx = torch.argmin(torch.sum(((nocs_map_values - nocs_value[None, :, None]) ** 2)[0], dim=0))
                    nocs_map_min_value = nocs_map_values[0, :, min_idx]
                    # int_coord_noise = (torch.rand(2) * 20 - 10).int()
                    int_coord_noise = (torch.rand(2) * 0 - 0).int()
                    nocs_map_coord = [nocs_mask_coords[2][min_idx] + int_coord_noise[0],
                                      nocs_mask_coords[3][min_idx] + int_coord_noise[1]]
                    nocs_value = nocs_map_min_value
                else:
                    nocs_map_coord = torch.LongTensor([128, 128]).to(image.device)

                if nocs_map_coord[0] < target_texture_side:
                    nocs_map_coord[0] = torch.LongTensor([target_texture_side])
                if nocs_map_coord[1] < target_texture_side:
                    nocs_map_coord[1] = torch.LongTensor([target_texture_side])
                if nocs_map_coord[0] > height - target_texture_side:
                    nocs_map_coord[0] = torch.LongTensor([height - target_texture_side])
                if nocs_map_coord[1] > width - target_texture_side:
                    nocs_map_coord[1] = torch.LongTensor([width - target_texture_side])
                texture_y_bot = nocs_map_coord[0] + target_texture_side
                texture_y_top = nocs_map_coord[0] - target_texture_side
                texture_x_left = nocs_map_coord[1] - target_texture_side
                texture_x_right = nocs_map_coord[1] + target_texture_side

            texture_side = target_texture_side

        if not use_nocs:
            nocs_value = -2.0

        texture_crop = image[:, :, texture_y_top:texture_y_bot, texture_x_left:texture_x_right]
        image_mask = image_mask[None, None, ...]
        mask_crop = image_mask[:, :, texture_y_top:texture_y_bot, texture_x_left:texture_x_right]
        coords = (texture_y_top, texture_y_bot, texture_x_left, texture_x_right)

        return texture_crop, mask_crop, coords, nocs_value, texture_side

    # store ShapeNet renders (car category) for NOCs training
    def store_shapenet_fornocs(self, savedir=None, shapenetdir=None):

        from model.differentiable_renderer import ChairRenderer
        from pytorch3d.io import load_objs_as_meshes

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/cars_shapenet_fornocs'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)

        if shapenetdir is None:
            SHAPENETDIR = '/canis/ShapeNet/ShapeNetCore.v2/02958343'
        else:
            SHAPENETDIR = shapenetdir

        jsonfiles = sorted([x for x in os.listdir(SAVEDIR) if x.endswith('.json')])

        for jsonfile in jsonfiles:

            img_id = jsonfile.split('.')[0]

            meta_data = json.loads(
                Path(os.path.join(SAVEDIR, jsonfile)).read_text())

            obj_id = meta_data['obj_id']
            intrinsics = None
            camera_pose = torch.eye(4).cuda()
            model_to_scan = np.array(meta_data['transform'])
            model_to_scan = torch.FloatTensor(model_to_scan).cuda()
            transform_matrices = (model_to_scan, camera_pose)

            mesh_shapenet = load_objs_as_meshes(
                [f'{SHAPENETDIR}/{obj_id}/models/model_normalized.obj'],
                load_textures=True,
                create_texture_atlas=True,
                texture_atlas_size=4,
                device='cuda'
            )
            if len(mesh_shapenet._verts_list[0]) > 40000:
                continue

            renderer = ChairRenderer(img_size=(968, 1296))
            images = renderer.render(mesh_shapenet, transform_matrices, intrinsics)
            images = images[..., :3]
            images = torch.permute(images, (0, 3, 1, 2))
            images_crop = images[:, :, 100:900, 250:1050]

            save_image(images.cpu().detach(),
                       os.path.join(SAVEDIR, f"{img_id}_render.jpg"),
                       value_range=(0, 1), normalize=True)
            save_image(images_crop.cpu().detach(),
                       os.path.join(SAVEDIR, f"{img_id}_rendercrop.jpg"),
                       value_range=(0, 1), normalize=True)

    # store ShapeNet renders (car category)
    def store_shapenet_textures(self, savedir=None, shapenetdir=None):

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/cars_shapenet'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)

        if shapenetdir is None:
            SHAPENETDIR = '/canis/ShapeNet/ShapeNetCore.v2/02958343'
        else:
            SHAPENETDIR = shapenetdir

        offset = 0

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)
            for batch_idx, batch in enumerate(self.train_dataloader()):

                if batch_idx < 0:
                    continue
                if batch_idx > 32767:
                    break

                obj_id = batch['name'][0]

                # ShapeNet
                # front, back, right, left, front_right, front_left, back_right, back_left
                azimuths_avail = [math.pi / 2, 3 * math.pi / 2,
                                  0, math.pi,
                                  math.pi / 2 - math.pi / 6, math.pi / 2 + math.pi / 6,
                                  0 - math.pi / 3, math.pi + math.pi / 3]
                azimuths_avail = np.array(azimuths_avail)

                azimuth_angle_indices = np.arange(8)
                camera_delta_angle_array_gt = azimuths_avail[azimuth_angle_indices]
                camera_elevation_array_gt = np.ones((8)) * (math.pi / 2 - math.pi / 48)
                camera_dist_array_gt = np.ones((8)) * 1.1
                camera_dist_array_gt[0] = 0.8
                camera_dist_array_gt[1] = 0.8
                mesh_shapenet = trimesh.load(f'{SHAPENETDIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                if len(mesh_shapenet.vertices) > 40000:
                    continue
                mesh_shapenet_mean = np.zeros(4)
                mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                norm_params = (mesh_shapenet_mean, mesh_shapenet_max)
                images_target = []
                for view_id in range(len(camera_delta_angle_array_gt)):
                    image_target = self.render_shapenet(obj_id,
                                                        camera_delta_angle=camera_delta_angle_array_gt[view_id] - np.pi / 2,
                                                        camera_elevation_angle=camera_elevation_array_gt[view_id] - np.pi / 2,
                                                        camera_dist=camera_dist_array_gt[view_id])
                    images_target += [image_target]
                images_target = torch.cat(images_target, dim=0)

                metadata = {}
                metadata['obj_id'] = obj_id
                metadata['shapenet_mean'] = list(mesh_shapenet_mean.cpu().numpy().astype('float'))
                metadata['shapenet_max'] = mesh_shapenet_max
                with open(os.path.join(SAVEDIR, f'obj_id_{batch_idx+offset}.json'), 'w') as fout:
                    json.dump(metadata, fout)

                for i in range(len(images_target)):
                    save_image(images_target.cpu().detach()[i][None, ...],
                               os.path.join(SAVEDIR, f"clip_{batch_idx+offset}_{i}.jpg"),
                               value_range=(0, 1), normalize=True)
                    params = {'azimuth': float(camera_delta_angle_array_gt[i]),
                              'elevation': float(camera_elevation_array_gt[i]),
                              'camera_id': i}
                    with open(os.path.join(SAVEDIR, f"clip_{batch_idx+offset}_{i}.json"), 'w') as fout:
                        json.dump(params, fout)

    # store CompCars GT images
    def store_compcars_textures(self, savedir=None):

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/cars_compcars'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)

        offset = 0

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)
            for batch_idx, batch in enumerate(self.val_dataloader()):

                if batch_idx > 32767:
                    break

                set_real_sample_resolution(batch, 256)
                obj_id = batch['name'][0]

                # CompCars
                image_target = self.train_set.get_color_bg_real(batch, highres=True)
                image_target = torch.nn.functional.interpolate(image_target, (512, 512), mode='bilinear',
                                                               align_corners=True)
                images_target = (image_target + 1.0) / 2.0

                metadata = {}
                metadata['obj_id'] = obj_id
                with open(os.path.join(SAVEDIR, f'obj_id_{batch_idx + offset}.json'), 'w') as fout:
                    json.dump(metadata, fout)

                for i in range(len(images_target)):
                    save_image(images_target.cpu().detach()[i][None, ...],
                               os.path.join(SAVEDIR, f"clip_{batch_idx + offset}_{i}.jpg"),
                               value_range=(0, 1), normalize=True)

    # inference loop, store synthesized renders
    def store_rendered_images(self, savedir=None):
        torch.autograd.set_detect_anomaly(True)
        offset = 0
        SETUP = 'random'
        if savedir is None:
            BASE_SAVE_DIR = "/cluster/valinor/abokhovkin/scannet-texturing/data/Projections/22111349_83_cars"
        else:
            BASE_SAVE_DIR = savedir
        setup_tokens = SETUP.split('+')
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
        if self.R is None:
            self.R = DifferentiableRenderer(self.config.image_size, "bounds", self.config.colorspace)

        self.grid_z = torch.randn(8192, self.config.latent_dim)
        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)
            latents = self.grid_z.split(1)

            self.G = self.G.to('cuda')
            self.E = self.E.to('cuda')
            self.R = self.R.to('cuda')

            print('len dataloader', len(self.train_dataloader()))
            for iter_idx, _batch in enumerate(self.train_dataloader()):
                # if iter_idx > 63:
                #     break
                if setup_tokens[0] == 'random':
                    input_batch = _batch
                    input_batch = to_device(input_batch, 'cuda')
                    self.set_shape_codes(input_batch)
                    input_latents = latents[iter_idx]

                if 'fixed_cam' not in setup_tokens:
                    _batch = to_device(_batch, 'cuda')
                    self.set_shape_codes(_batch)
                    shape = _batch['shape']
                    input_batch['shape'] = shape
                    input_batch['vertices'] = _batch['vertices']

                meta_data = {}
                meta_data['name'] = _batch['name']

                pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(input_batch['graph_data'],
                                                                                     input_latents.to('cuda'),
                                                                                     shape,
                                                                                     noise_mode='const')
                fake_render = self.render(tuned_feature, input_batch,
                                          face_color=pred_colors,
                                          use_shape_features=self.config.render_shape_features,
                                          use_color_bg=False,
                                          style_latent=z_out)[:, :3, :, :]
                for k in range(len(fake_render)):
                    save_image(fake_render[k][None, ...],
                               os.path.join(BASE_SAVE_DIR, f"{iter_idx + offset}_{k}_{SETUP}.jpg"), value_range=(-1, 1), normalize=True)
                torch.save(input_latents.cpu().detach(), os.path.join(BASE_SAVE_DIR, f"globlat_{iter_idx + offset}_{SETUP}.pth"))

                with open(os.path.join(BASE_SAVE_DIR, f"meta_{iter_idx + offset}_{SETUP}.json"), 'w') as fout:
                    json.dump(meta_data, fout)

    # load model to classify azimuth, elevation angles
    def load_pose_model(self, ckpt_path=None):

        from model.discriminator import PoseClassifier

        pose_classifier = PoseClassifier(out_dim_azimuth=8, out_dim_elevation=0)
        checkpoint = torch.load(ckpt_path)['state_dict']
        checkpoint = {k[11:]: v for k, v in checkpoint.items()}

        pose_classifier.load_state_dict(checkpoint)

        return pose_classifier

    # optimize texture for aligned image-shapes, two modes (compcars / shapenet)
    def optimize_stylemesh_image_vgg_mlp_aligned(
            self, 
            ckpt_ema=None, 
            ckpt_ema_r=None, 
            mode='compcars',
            task='aligned', 
            vgg_path=None, 
            pose_ckpt_path=None,
            nocs_dir=None,
            shapenet_dir=None,
            dataset_path=None,
            prefix=None
        ):

        from model.differentiable_renderer import CarsRenderer
        from util.stylemesh import image_pyramid, ContentAndStyleLoss, pre, post
        from pytorch3d.io import load_objs_as_meshes
        from copy import deepcopy

        device = torch.device("cuda:0")

        # loss weights (probably need to adjust) and other hyperparams
        base_style_weight = 1e6 # 1e6 / 1.0 
        base_content_weight = 1.0 # 1.0
        base_view_weight = 1.0
        w_patch_loss = 1.0
        w_global_base_loss = 1.0
        num_levels = 3
        num_iterations = 801

        # load VGG
        if vgg_path is None:
            VGG_PATH = '/cluster/valinor/abokhovkin/StyleMesh/VGG/vgg_conv.pth'
        else:
            VGG_PATH = vgg_path
        StyleLoss = ContentAndStyleLoss(VGG_PATH)
        StylePatchLoss = ContentAndStyleLoss(VGG_PATH)

        self.G = self.G.to('cuda')
        self.R = self.R.to('cuda')
        ema = torch.load(ckpt_ema, map_location=device)
        ema.copy_to([p for p in self.G.parameters() if p.requires_grad])
        ema_r = torch.load(ckpt_ema_r, map_location=device)
        ema_r.copy_to([p for p in self.R.parameters() if p.requires_grad])
        initial_G_parameters = deepcopy(self.G.synthesis.tuner.state_dict())
        # initial_R_parameters = deepcopy(self.R.state_dict())


        print('Mode', mode)
        if mode not in ['compcars', 'shapenet']:
            raise NotImplementedError(f'Mode {mode} is not implemented')

        if prefix is None:
            PREFIX = 'car_shapenet_500iter_stylemeshvgg_nocspatch64_unalignedtrans'
        else:
            PREFIX = prefix
        odir_real, odir_fake, odir_samples, odir_grid, odir_meshes, output_dir_samples_levels = self.create_store_directories(
            prefix=PREFIX
        )

        len_loader = len(self.val_dataloader())
        collater = Collater([], [], True)
        shapenet_renderer = CarsRenderer(img_size=(512, 512))

        if pose_ckpt_path is None:
            POSE_CKPT_PATH = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/08021506_ImageLatent_cars_shapenet_pose_resnet18_lr0.00002/checkpoints/_epoch=119.ckpt'
        else:
            POSE_CKPT_PATH = pose_ckpt_path
        pose_model = self.load_pose_model(POSE_CKPT_PATH)
        pose_model = pose_model.cuda()
        pose_model.eval()

        if mode == 'compcars':
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/car_compcars_aligned_EVAL'
            else:
                NOCS_DIR = nocs_dir
        else:
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/car_shapenet_aligned_EVAL'
            else:
                NOCS_DIR = nocs_dir

        if shapenet_dir is None:
            SHAPENET_DIR = '/canis/ShapeNet/ShapeNetCore.v2/02958343'
        else:
            SHAPENET_DIR = shapenet_dir

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)

            all_optimized_images = []
            all_optimized_faces_images = []
            for batch_idx, batch in enumerate(self.val_dataloader()):

                # if batch_idx > 512:
                #     break

                obj_id = batch['name'][0]

                print(f'Image #{batch_idx}')
                set_real_sample_resolution(batch, 256)
                if task == 'transfer':
                    used_obj_ids = [x.split('.')[0] for x in os.listdir(dataset_path)]
                    obj_id = used_obj_ids[(batch_idx + 5) % 256]
                else:
                    obj_id = batch['name'][0]
                view_indices = batch['view_indices']

                filenames = batch['image_filename']
                available_views = get_car_views_nonoise()
                sampled_views = [available_views[vidx] for vidx in view_indices]
                norm_params = None
                camera_delta_angle = None
                transform_matrices = None

                if mode == 'shapenet':
                    camera_delta_angle_array_gt = [2 * np.pi - sampled_views[i]['azimuth'] - np.pi / 2 for i in range(len(sampled_views))]
                    camera_delta_elevation_array_gt = [np.pi - np.pi / 24] * len(sampled_views)
                    camera_delta_angle = camera_delta_angle_array_gt[4]

                    mesh_shapenet = load_objs_as_meshes(
                        [f'{SHAPENET_DIR}/{obj_id}/models/model_normalized.obj'],
                        load_textures=True,
                        create_texture_atlas=True,
                        texture_atlas_size=4,
                        device='cuda'
                    )
                    if len(mesh_shapenet._verts_list[0]) > 40000:
                        continue
                    mesh_vertices = mesh_shapenet._verts_list[0]
                    mesh_shapenet_mean = (mesh_vertices.amax(dim=0) + mesh_vertices.amin(dim=0)) / 2
                    mesh_shapenet._verts_list[0] = mesh_shapenet._verts_list[0] - mesh_shapenet_mean
                    mesh_vertices = mesh_shapenet._verts_list[0]
                    mesh_shapenet_max = mesh_vertices[:, 1].max()
                    mesh_shapenet._verts_list[0] = mesh_shapenet._verts_list[0] / mesh_shapenet_max * 0.16

                    mesh_shapenet_max = 0.16
                    norm_params = (0.0, mesh_shapenet_max)

                    images_target = []
                    for view_id in range(len(sampled_views)):
                        image_target = shapenet_renderer.render(mesh_shapenet,
                                                                camera_delta_elevation_array_gt[view_id],
                                                                camera_delta_angle_array_gt[view_id])
                        images_target += [image_target]
                    images_target = torch.cat(images_target, dim=0)
                    images_target = torch.permute(images_target, (0, 3, 1, 2))[:, :3, ...]

                    image_target_opt = images_target[4].to('cuda')
                    image_target_opt = pre()(image_target_opt)
                    image_target_opt = image_target_opt[None, ...]

                elif mode == 'compcars':
                    mesh_shapenet = trimesh.load(f'{SHAPENET_DIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                    mesh_shapenet_mean = np.zeros(4)
                    mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                    mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda() * 0.0
                    mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                    mesh_shapenet_max = 0.16
                    norm_params = (mesh_shapenet_mean, mesh_shapenet_max)
                    images_target = self.train_set.get_color_bg_real(batch, highres=True)
                    images_target = torch.nn.functional.interpolate(images_target, (512, 512), mode='bilinear',
                                                                   align_corners=True)
                    images_target = (images_target + 1.0) / 2.0
                    image_target_opt = images_target[4:5].to('cuda')
                    camera_delta_angle = 2 * np.pi - sampled_views[4]['azimuth'] - np.pi / 2
                    camera_delta_angle_array_gt = [2 * np.pi - sampled_views[i]['azimuth'] - np.pi / 2 for i in range(len(sampled_views))]
                    camera_delta_elevation_array_gt = [np.pi - np.pi / 24] * len(sampled_views)

                try:
                    gt_nocs = np.load(os.path.join(NOCS_DIR, f'nocs_gt_{batch_idx}.npy'))
                    gt_nocs = torch.FloatTensor(gt_nocs).cuda()
                except FileNotFoundError:
                    print(f'NOCs image {os.path.join(NOCS_DIR, f'nocs_gt_{batch_idx}.npy')} not found')
                    continue

                meta_data = {
                    'shape_id': batch['name'][0],
                    'azimuth': [float(x) for x in camera_delta_angle_array_gt],
                    'elevation': [float(x) for x in camera_delta_elevation_array_gt],
                    'image_filenames': filenames
                }

                azimuth_logits, elevation_logits = pose_model(image_target_opt) # unaligned and texture transfer
                camera_delta_angle = 2 * np.pi - available_views[int(torch.argmax(azimuth_logits).cpu())]['azimuth'] - np.pi / 2

                # IF YOU NEED TO FIND PARTICULAR SHAPENET OBJECT ID
                # try:
                #     batch = self.train_set.get_by_shapenet_id(obj_id)
                #     batch = collater([batch])
                # except NotImplementedError:
                #     print(f'The sample with shapenet id {obj_id} was not found!')
                #     continue

                batch = to_device(batch, 'cuda')
                self.E = self.E.to('cuda')
                self.D = self.D.to('cuda')
                self.set_shape_codes(batch)
                shape = batch['shape']

                image_target_mask = torch.where((image_target_opt[:, 0] > 0.99) & (image_target_opt[:, 1] > 0.99) & (
                            image_target_opt[:, 2] > 0.99), 0, 1)
                image_target_mask = image_target_mask[:, None, ...]
                image_target_mask = image_target_mask.float()
                gt_nocs_mask = torch.cat([gt_nocs, image_target_mask], dim=1)

                # StyleLoss.set_style_image(image_target_opt, num_levels)
                StyleLoss.set_style_image(image_target_opt, num_levels, image_target_mask)

                self.R.create_angles()

                # define optimizable parameters
                init_rand_face_feature = torch.randn((24576, 256)).cuda() / 25.
                init_rand_face_feature.requires_grad = True
                latent = torch.randn((1, 512)).cuda() / 3. # random latent
                latent.requires_grad = True
                init_shape = [shape[i].clone().detach()[:].cuda() for i in range(len(shape))]
                for i in range(len(init_shape)):
                    init_shape[i].requires_grad = True

                lr_latent = 1e-2
                lr_mlp = 1e-3

                optimizer_latent = torch.optim.Adam([
                    {'params': [latent], 'lr': lr_latent, 'betas': (0.9, 0.99), 'eps': 1e-8}
                ])
                optimizer = torch.optim.Adam([
                    {'params': [init_rand_face_feature], 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': list(self.G.synthesis.tuner.parameters()), 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    # {'params': list(self.R.parameters()), 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    # {'params': init_shape, 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8}
                ])
                # list(self.G.synthesis.tuner.parameters()) + [init_rand_face_feature]

                all_images = []
                all_faces_images = []
                all_content_losses = []
                all_style_losses = []
                all_latent_norms = []
                target_crops = []
                target_mask_crops = []
                fake_crops = []
                fake_mask_crops = []
                target_render_crop = None
                target_render_crop_save = None
                for iter_idx in range(num_iterations):

                    shape = init_shape

                    def closure():
                        nonlocal all_images
                        nonlocal all_faces_images
                        nonlocal all_content_losses
                        nonlocal all_style_losses
                        nonlocal all_latent_norms
                        nonlocal all_optimized_images
                        nonlocal all_optimized_faces_images
                        nonlocal target_render_crop
                        nonlocal target_render_crop_save
                        nonlocal target_crops
                        nonlocal target_mask_crops
                        nonlocal fake_crops
                        nonlocal fake_mask_crops
                        nonlocal gt_nocs
                        nonlocal transform_matrices

                        t = 1000 * time.time()
                        torch.manual_seed(int(t) % 2 ** 32)
                        pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(batch['graph_data'],
                                                                                             latent, shape, noise_mode='const')

                        np.random.seed(15)
                        camera_delta_angle_array = np.random.uniform(-np.pi / 2, np.pi / 2, self.config.views_per_sample)
                        camera_delta_angle_array[0] = camera_delta_angle
                        camera_delta_elevation_array = [np.pi - np.pi / 24] * self.config.views_per_sample
                        # camera_delta_angle_array = None
                        t = 1000 * time.time()
                        np.random.seed(int(t) % 2 ** 32)

                        # render faces (not useful)
                        fake_render_faces = self.render_color(
                            batch,
                            face_color=pred_colors,
                            use_color_bg=False,
                            camera_delta_angle_array=camera_delta_angle_array,
                            camera_delta_elevation_array=camera_delta_elevation_array,
                            norm_params=norm_params
                        )
                        fake_render_faces_mask = fake_render_faces[:, 3:4, :, :]
                        fake_render_faces = fake_render_faces[:, :3, :, :]
                        fake_render_faces_reshaped = torch.nn.functional.interpolate(fake_render_faces, (512, 512),
                                                                                     mode='bilinear',
                                                                                     align_corners=True)
                        fake_render_faces_reshaped = (fake_render_faces_reshaped + 1.0) / 2.0

                        if iter_idx < 200:
                            opt_features = tuned_feature
                            w_global_loss = w_global_base_loss
                        else:
                            opt_features = tuned_feature + init_rand_face_feature
                            w_global_loss = 0.0

                        if iter_idx in [0, 100, 150, 200, 250, 300, 350, 400, 600, 800]:
                            only_first_view = False
                        else:
                            only_first_view = True

                        fake_render, image_ref, _ = self.render(
                            opt_features, 
                            batch,
                            face_color=pred_colors,
                            use_shape_features=True,
                            use_color_bg=False,
                            style_latent=z_out,
                            only_first_view=only_first_view,
                            render_nocs=True,
                            render_sil=False,
                            camera_delta_angle_array=camera_delta_angle_array,
                            camera_delta_angle=camera_delta_angle,
                            camera_delta_elevation_array=camera_delta_elevation_array,
                            norm_params=norm_params
                        )
                        fake_render_mask = fake_render[:, 3:4, :, :]
                        fake_render = fake_render[:, :3, :, :]
                        fake_render_reshaped = torch.nn.functional.interpolate(fake_render, (512, 512),
                                                                               mode='bilinear', align_corners=True)
                        fake_render_reshaped = (fake_render_reshaped + 1.0) / 2.0

                        image_ref = torch.permute(image_ref, (0, 3, 1, 2))
                        image_ref_mask = image_ref[:, 3:4, ...]
                        image_ref = image_ref[:, :3, ...]
                        image_ref = torch.nn.functional.interpolate(image_ref, (512, 512),
                                                                    mode='bilinear', align_corners=True)
                        image_ref_mask = torch.nn.functional.interpolate(image_ref_mask, (512, 512),
                                                                         mode='nearest')
                        
                        image_ref_mask = torch.ones_like(image_ref_mask).float() # ablation

                        pred_nocs = image_ref
                        scale = gt_nocs.max() / pred_nocs.max()
                        pred_nocs = pred_nocs * scale
                        pred_nocs_mask = torch.cat([image_ref, image_ref_mask[0:1]], dim=1)

                        preprocessed = []
                        for k_view in range(len(fake_render_reshaped)):
                            fake_render_reshaped_view = pre()(fake_render_reshaped[k_view])
                            preprocessed += [fake_render_reshaped_view[None, ...]]
                        fake_render_reshaped = torch.cat(preprocessed, dim=0)


                        # Global loss branch
                        global_loss = 0.0
                        for k_view in range(len(fake_render_reshaped)):

                            fake_pyramid = image_pyramid(fake_render_reshaped[k_view][None, ...], list(np.arange(num_levels)), reverse=True)
                            fake_mask_pyramid = image_pyramid(fake_render_mask[k_view][None, ...], list(np.arange(num_levels)), reverse=True, mode='nearest')

                            style_loss, content_loss, pyramid = StyleLoss(fake_pyramid, image_target_opt, fake_mask_pyramid)

                            style_weight = 1.0
                            view_weight = base_view_weight if k_view == 0 else 0.0
                            content_weight = base_content_weight

                            global_loss = global_loss + (view_weight * style_weight * style_loss + view_weight * content_weight * content_loss)

                            break # only first view


                        # Patch loss branch
                        # multi-patch from gt to prediction
                        fake_render_crops_save = []
                        fake_render_mask_crops_save = []
                        fake_render_crops = []
                        fake_render_mask_crops = []
                        gt_render_crops = []
                        gt_render_mask_crops = []
                        gt_render_crops_save = []
                        gt_render_mask_crops_save = []

                        patch_loss = 0.0
                        num_patches_per_view = 4  # 1 / 4

                        for k_patch in range(num_patches_per_view):
                            target_render_crop_, target_mask_crop_, target_coords, target_nocs_value, target_texture_side = self.crop_texture_cover(
                                image_target_opt,
                                1,
                                coords=None,
                                nocs_map=gt_nocs_mask,
                                use_nocs=True)
                            
                            target_mask_crop_ = target_mask_crop_.float()
                            target_render_crop = torch.nn.functional.interpolate(target_render_crop_, (512, 512),
                                                                                 mode='bilinear',
                                                                                 align_corners=True)
                            target_mask_crop = torch.nn.functional.interpolate(target_mask_crop_, (512, 512),
                                                                               mode='nearest')
                            target_render_crop_save = torch.nn.functional.interpolate(target_render_crop_,
                                                                                      (128, 128),
                                                                                      mode='bilinear',
                                                                                      align_corners=True)
                            target_mask_crop_save = torch.nn.functional.interpolate(target_mask_crop_, (128, 128),
                                                                                    mode='nearest')
                            gt_render_crops += [target_render_crop]
                            gt_render_mask_crops += [target_mask_crop]
                            gt_render_crops_save += [target_render_crop_save]
                            gt_render_mask_crops_save += [target_mask_crop_save]

                            fake_render_crop_, fake_render_mask_crop_, coords, found_nocs_value, _ = self.crop_texture_cover(
                                fake_render_reshaped[0][None, ...],
                                1,
                                coords=target_coords,
                                use_nocs=True,
                                nocs_map=pred_nocs_mask,
                                nocs_value=target_nocs_value,
                                target_texture_side=target_texture_side)

                            fake_render_mask_crop_ = fake_render_mask_crop_.float()
                            fake_render_crop = torch.nn.functional.interpolate(fake_render_crop_, (512, 512),
                                                                               mode='bilinear', align_corners=True)
                            fake_render_crop_save = torch.nn.functional.interpolate(fake_render_crop_, (128, 128),
                                                                                    mode='bilinear',
                                                                                    align_corners=True)
                            fake_render_mask_crop = torch.nn.functional.interpolate(fake_render_mask_crop_,
                                                                                    (512, 512),
                                                                                    mode='nearest')
                            fake_render_mask_crop_save = torch.nn.functional.interpolate(fake_render_mask_crop_,
                                                                                         (128, 128),
                                                                                         mode='nearest')

                            fake_render_crops_save += [fake_render_crop_save]
                            fake_render_crops += [fake_render_crop]
                            fake_render_mask_crops_save += [fake_render_mask_crop_save]
                            fake_render_mask_crops += [fake_render_mask_crop]

                        fake_patch_pyramids = []
                        fake_render_patch_mask_pyramids = []
                        for k in range(len(fake_render_crops)):
                            fake_patch_pyramid = image_pyramid(fake_render_crops[k], np.arange(num_levels),
                                                               reverse=True)
                            fake_render_patch_mask_pyramid = image_pyramid(fake_render_mask_crops[k],
                                                                           np.arange(num_levels),
                                                                           reverse=True, mode='nearest')
                            fake_patch_pyramids += [fake_patch_pyramid]
                            fake_render_patch_mask_pyramids += [fake_render_patch_mask_pyramid]

                        StylePatchLoss.set_style_image_multiview(torch.cat(gt_render_crops), num_levels,
                                                                 torch.cat(gt_render_mask_crops))

                        style_loss, content_loss = StylePatchLoss.forward_multiview(fake_patch_pyramids,
                                                                                    gt_render_crops,
                                                                                    fake_render_patch_mask_pyramids)

                        # style_weight = base_style_weight if k_view != 0 else 0.0
                        style_weight = base_style_weight
                        # content_weight = base_content_weight if k_view == 0 else 0.0
                        content_weight = base_content_weight

                        patch_loss = patch_loss + (style_weight * style_loss + content_weight * content_loss)







                        if iter_idx % 10 == 0:
                            print(f'LOSSES: {global_loss} (global), {patch_loss} (patch)')

                        loss = (w_global_loss * global_loss + w_patch_loss * patch_loss) / 1.0
                        loss.backward(retain_graph=True)

                        all_content_losses += [content_loss.cpu().detach()]
                        all_style_losses += [style_loss.cpu().detach()]
                        all_latent_norms += [torch.sum(latent ** 2).cpu().detach()]
                        
                        if iter_idx in [0, 100, 150, 200, 250, 300, 350, 400, 600, 800]:

                            with open(str(odir_samples / f"metadata_{batch_idx}.json"), 'w') as fout:
                                json.dump(meta_data, fout)

                            save_image(fake_render_reshaped.cpu().detach(),
                                       odir_samples / f"fake_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)
                            # save_image(pred_nocs.cpu().detach(),
                            #            odir_samples / f"fakenocs_{batch_idx}_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)


                            camera_delta_elevation_array = [np.pi - np.pi / 12] * 8

                            fake_render, _, _ = self.render(opt_features, batch,
                                                            face_color=pred_colors,
                                                            use_shape_features=True,
                                                            use_color_bg=False,
                                                            style_latent=z_out,
                                                            only_first_view=False,
                                                            render_nocs=False,
                                                            render_sil=False,
                                                            camera_delta_angle_array=camera_delta_angle_array_gt,
                                                            camera_delta_angle=camera_delta_angle,
                                                            camera_delta_elevation_array=camera_delta_elevation_array,
                                                            norm_params=norm_params)
                            fake_render_mask = fake_render[:, 3:4, :, :]
                            fake_render = fake_render[:, :3, :, :]
                            fake_render_reshaped = torch.nn.functional.interpolate(fake_render, (512, 512),
                                                                                   mode='bilinear', align_corners=True)
                            fake_render_reshaped = (fake_render_reshaped + 1.0) / 2.0
                            save_image(fake_render_reshaped.cpu().detach(),
                                       odir_samples / f"fakeasgt_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)

                            # Additional outputs in case you need them
                            # save_image(fake_render_faces_reshaped.cpu().detach(),
                            #            odir_samples / f"fakefaces_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)
                            # fake_render_faces = self.render_color(batch,
                            #                                       face_color=pred_colors,
                            #                                       use_color_bg=False,
                            #                                       camera_delta_angle_array=camera_delta_angle_array_gt,
                            #                                       camera_delta_elevation_array=camera_delta_elevation_array,
                            #                                       norm_params=norm_params
                            #                                       )
                            # fake_render_faces_mask = fake_render_faces[:, 3:4, :, :]
                            # fake_render_faces = fake_render_faces[:, :3, :, :]
                            # fake_render_faces_reshaped = torch.nn.functional.interpolate(fake_render_faces, (512, 512),
                            #                                                              mode='bilinear',
                            #                                                              align_corners=True)
                            # fake_render_faces_mask = torch.nn.functional.interpolate(fake_render_faces_mask, (512, 512),
                            #                                                          mode='nearest')
                            # fake_render_faces_reshaped = (fake_render_faces_reshaped + 1.0) / 2.0
                            # save_image(fake_render_faces_reshaped.cpu().detach(),
                            #            odir_samples / f"fakefacesasgt_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)
                            # save_image(fake_render_faces_mask.cpu().detach(),
                            #            odir_samples / f"fakefacesmaskasgt_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1),
                            #            normalize=True)

                        if iter_idx % 10 == 0:
                            print("run {}:".format(iter_idx))
                            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                                style_loss.item(), content_loss.item()))
                            print('Saving to:', odir_samples)
                            print()
                        if iter_idx % 50 == 0:
                            all_images += [fake_render_reshaped.cpu().detach()]
                            # all_faces_images += [fake_render_faces_reshaped.cpu().detach()]
                        if iter_idx % 20 == 0:
                            target_crops += [gt_render_crops_save[0].cpu().detach()]
                            fake_crops += [fake_render_crops_save[0].cpu().detach()]
                            target_mask_crops += [gt_render_mask_crops_save[0].cpu().detach()]
                            fake_mask_crops += [fake_render_mask_crops_save[0].cpu().detach()]

                        # Additional outputs in case you need them
                        # if iter_idx == num_iterations - 1:
                        #     save_image(fake_render_reshaped.cpu().detach(),
                        #                odir_samples / f"fake_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     save_image(fake_render_mask.cpu().detach(),
                        #                odir_samples / f"fakemask_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     # save_image(fake_render_faces_reshaped.cpu().detach(),
                        #     #            odir_samples / f"fake_faces_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     save_image(pred_nocs.cpu().detach(),
                        #                odir_samples / f"fakenocs_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        #     save_image(gt_nocs.cpu().detach(),
                        #                odir_samples / f"gtnocs_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        #     save_image(torch.cat(all_images, dim=0),
                        #                odir_samples / f"opt_fake_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     # save_image(torch.cat(all_faces_images, dim=0),
                        #     #            odir_samples / f"opt_fake_faces_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     all_content_losses = torch.stack(all_content_losses)
                        #     all_style_losses = torch.stack(all_style_losses)
                        #     torch.save(all_content_losses, odir_samples / f"loss_content_{batch_idx}.pth")
                        #     torch.save(all_style_losses, odir_samples / f"loss_style_{batch_idx}.pth")
                        #     torch.save(all_latent_norms, odir_samples / f"latent_norm_{batch_idx}.pth")
                        #     save_image(torch.cat(target_crops, dim=0),
                        #                odir_samples / f"realpatches_{batch_idx}.jpg", value_range=(0, 1),
                        #                normalize=True)
                        #     save_image(torch.cat(target_mask_crops, dim=0),
                        #                odir_samples / f"realpatchesmask_{batch_idx}.jpg", value_range=(0, 1),
                        #                normalize=True)
                        #     save_image(torch.cat(fake_crops, dim=0),
                        #                odir_samples / f"fakepatches_{batch_idx}.jpg", value_range=(0, 1),
                        #                normalize=True)
                        #     save_image(torch.cat(fake_mask_crops, dim=0),
                        #                odir_samples / f"fakepatchesmask_{batch_idx}.jpg", value_range=(0, 1),
                        #                normalize=True)
                            # all_content_losses = []
                            # all_style_losses = []
                            # all_latent_norms = []
                            # target_crops = []
                            # target_mask_crops = []
                            # fake_crops = []
                            # fake_mask_crops = []

                        return 1.0 * (style_loss + content_loss)

                    # optimizer_latent.step(closure)
                    # optimizer_latent.zero_grad()

                    if iter_idx < 200:
                        optimizer_latent.step(closure)
                        optimizer_latent.zero_grad()
                    else:
                        optimizer.step(closure)
                        optimizer.zero_grad()

                    # optimizer_latent.step(closure)
                    # optimizer_latent.zero_grad()

                del norm_params
                del camera_delta_angle
                del transform_matrices

                # save_image(image_target_clip.cpu().detach(),
                #            odir_samples / f"gt_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                # save_image(images_target.cpu().detach(),
                #            odir_samples / f"gt_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                # save_image(image_target_mask.cpu().detach(),
                #            odir_samples / f"gtmask_{batch_idx}.jpg", value_range=(0, 1), normalize=True)

                self.G.synthesis.tuner.load_state_dict(initial_G_parameters)
                # self.R.load_state_dict(initial_R_parameters)


# set GT image resolution
def set_real_sample_resolution(batch, resolution):
    if batch['real'].shape[-1] != resolution:
        batch['real2'] = torch.nn.functional.interpolate(batch['real'], size=(2 * resolution, 2 * resolution), mode='bilinear', align_corners=False)
        batch['mask2'] = torch.nn.functional.interpolate(batch['mask'], size=(2 * resolution, 2 * resolution), mode='nearest')
        batch['real'] = torch.nn.functional.interpolate(batch['real'], size=(resolution, resolution), mode='bilinear', align_corners=False)
        batch['mask'] = torch.nn.functional.interpolate(batch['mask'], size=(resolution, resolution), mode='nearest')
    else:
        batch['real2'] = torch.nn.functional.interpolate(batch['real'], size=(2 * resolution, 2 * resolution), mode='bilinear', align_corners=False)
        batch['mask2'] = torch.nn.functional.interpolate(batch['mask'], size=(2 * resolution, 2 * resolution), mode='nearest')
        pass

# wrapper for step to clip gradients
def step(opt, module):
    for param in module.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
    opt.step()


@hydra.main(config_path='../config', config_name='stylegan2_car')
def main(config):
    trainer = create_trainer("StyleGAN23D-CompCars", config)
    model = StyleGAN2Trainer(config)

    # config.resume = '...'
    # model = StyleGAN2Trainer(config).load_from_checkpoint(config.resume)
    trainer.fit(model)


if __name__ == '__main__':
    main()
