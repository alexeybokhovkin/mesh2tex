import math
import shutil
from pathlib import Path
import os
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
import sys
sys.path.append('/rhome/abokhovkin/projects/clean-fid')
from cleanfid import fid
from torch.nn import functional as F
import time
import json

from dataset.mesh_real_features import FaceGraphMeshDataset
from dataset import to_vertex_features_scatter, to_vertex_colors_scatter, to_vertex_colors_scatter_level, GraphDataLoader, to_device
from dataset import Collater
from model.augment import AugmentPipe
from model.differentiable_renderer import DifferentiableMLPRenderer, DifferentiableRenderer, DifferentiableP3DRenderer, ShapeNetRenderer
from model.graph import TwinGraphEncoder
from model.graph_generator_u_deep_features_tuned import Generator
from model.discriminator import Discriminator
from model.loss import PathLengthPenalty, compute_gradient_penalty
from trainer import create_trainer
from util.timer import Timer
from util import palette
from trainer.fid_score_per_image_multiview import compute_multiview_score
from trainer.fid_score_per_image import compute_score
import trimesh
from collections import defaultdict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import clip

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

        config.mode = 'scannet'
        print('Mode', config.mode)
        if config.mode == 'shapenet':
            config.dataset_path = "/cluster/valinor/abokhovkin/scannet-texturing/data/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color_left"
        elif config.mode == 'scannet':
            config.dataset_path = "/cluster/daidalos/abokhovkin/scannet-texturing/data/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color"  # ScanNet
        config.num_eval_images = 1024

        self.train_set = FaceGraphMeshDataset(config, dump_data=True)
        self.val_set = FaceGraphMeshDataset(config, config.num_eval_images)

        # ONLY FOR DEBUG, created limited-size dataset
        # self.train_set = FaceGraphMeshDataset(config, 24)
        # self.val_set = FaceGraphMeshDataset(config, 24)

        if config.overfit:
            self.train_set = FaceGraphMeshDataset(config, 512, single_mode=True)
            self.val_set = FaceGraphMeshDataset(config, 48, single_mode=True)

        # allocate modules
        self.E = TwinGraphEncoder(self.train_set.num_feats, 1)
        self.G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max)
        self.D_faces = Discriminator(256, 3, w_num_layers=config.num_mapping_layers, mbstd_on=config.mbstd_on, channel_base=self.config.d_channel_base)
        self.D = Discriminator(512, 3, w_num_layers=config.num_mapping_layers, mbstd_on=config.mbstd_on, channel_base=self.config.d_channel_base)


        self.R = DifferentiableMLPRenderer(self.config.render_size, small_shader=self.config.small_shader,
                                           residual=self.config.residual, tanh=self.config.tanh_render,
                                           use_act=self.config.use_act_render, mode='chair')
        self.R_faces = None
        self.R_color = DifferentiableP3DRenderer(self.config.render_size)
        self.R_shapenet = ShapeNetRenderer(self.config.render_size)

        # original StyleGAN2 class for augmentation
        self.augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size, config.views_per_sample, config.colorspace)
        
        # set additional hyperparameters
        self.grid_z = torch.randn(config.num_eval_images, self.config.latent_dim)
        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.ema = None
        self.ema_r = None

        self.stable_global_step = 0


        # self.config.resume_ema = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/16110010_StyleGAN23D_mlpshader_bigdisc_twodisc_lrr5-5_lrd14-4_lrmd10-4_lrg12-4_lre1-4_noact_stablegs_mlp2x_2gpu/checkpoints/ema_000078480.pth'
        # self.config.resume_ema_r = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/16110010_StyleGAN23D_mlpshader_bigdisc_twodisc_lrr5-5_lrd14-4_lrmd10-4_lrg12-4_lre1-4_noact_stablegs_mlp2x_2gpu/checkpoints/ema_r_000078480.pth'

        # if config.load_checkpoint:
            # orig Texturify checkpoint
            # pretrained_dict = torch.load(
            #     '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/09111652_StyleGAN23D_mlpshader_bigdisc_twodisc_lrr1-4_lrd1-3_lrmd1e-4_lrg1-3_lre1-4_noact_stablegs_2gpu_detachmlp/checkpoints/_epoch=109.ckpt'
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
        r_opt = torch.optim.Adam([
            {'params': list(self.R.parameters()), 'lr': self.config.lr_r, 'betas': (0.0, 0.99), 'eps': 1e-8},
            {'params': [self.R.aux_latent], 'lr': self.config.lr_r, 'betas': (0.0, 0.99), 'eps': 1e-8}
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
    def g_step(self, batch, batch_idx):
        g_opt = self.optimizers()[0]
        g_opt.zero_grad(set_to_none=True)
        r_opt = self.optimizers()[2]
        r_opt.zero_grad(set_to_none=True)
        fake, w, feature, tuned_feature, z_out = self.forward(batch)

        rendered_faces_fake = self.render_color(batch,
                                                face_color=fake,
                                                resolution=256)[:, :3, :, :]
        p_faces_fake = self.D_faces(self.augment_pipe(rendered_faces_fake))
        gen_faces_loss = torch.nn.functional.softplus(-p_faces_fake).mean()

        rendered_fake, _, _ = self.render(tuned_feature, batch,
                                    face_color=fake,
                                    use_shape_features=self.config.render_shape_features,
                                    resolution=384,
                                    style_latent=z_out)
        rendered_fake = rendered_fake[:, :3, :, :]
        rendered_fake = torch.nn.functional.interpolate(rendered_fake, (512, 512), mode='bilinear',
                                                        align_corners=True)
        p_fake = self.D(self.augment_pipe(rendered_fake))
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
    def g_regularizer(self, batch, batch_idx):
        g_opt = self.optimizers()[0]
        for idx in range(len(batch['shape'])):
            batch['shape'][idx] = batch['shape'][idx].detach()
        g_opt.zero_grad(set_to_none=True)
        fake, w, feature, tuned_feature, z_out = self.forward(batch)

        rendered_faces_fake = self.render_color(batch,
                                                face_color=fake,
                                                resolution=256)[:, :3, :, :]
        plp_faces = self.path_length_penalty(rendered_faces_fake, w)

        rendered_fake, _, _ = self.render(tuned_feature, batch,
                                    face_color=fake,
                                    use_shape_features=self.config.render_shape_features,
                                    resolution=256,
                                    style_latent=z_out)
        rendered_fake = rendered_fake[:, :3, :, :]
        plp = self.path_length_penalty(rendered_fake, w)

        if not torch.isnan(plp_faces) and not torch.isnan(plp):
            gen_faces_loss = self.config.lambda_plp * plp_faces * self.config.lazy_path_penalty_interval
            gen_loss = self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
            self.log("rPLP_faces", plp_faces, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log("rPLP", plp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

            self.manual_backward(gen_loss)
            # self.manual_backward(gen_faces_loss + gen_loss)

            step(g_opt, self.G)

    # Discriminator step
    def d_step(self, batch, batch_idx):
        d_faces_opt = self.optimizers()[1]
        d_faces_opt.zero_grad(set_to_none=True)
        d_opt = self.optimizers()[3]
        d_opt.zero_grad(set_to_none=True)

        fake, _, feature, tuned_feature, z_out = self.forward(batch)

        rendered_faces_fake = self.render_color(batch,
                                                face_color=fake.detach(),
                                                resolution=256)[:, :3, :, :]
        p_faces_fake = self.D_faces(self.augment_pipe(rendered_faces_fake))
        fake_faces_loss = torch.nn.functional.softplus(p_faces_fake).mean()

        p_faces_real = self.D_faces(self.augment_pipe(self.train_set.get_color_bg_real(batch)))
        self.augment_pipe.accumulate_real_sign(p_faces_real.sign().detach())
        real_faces_loss = torch.nn.functional.softplus(-p_faces_real).mean()

        rendered_fake, _, _ = self.render(tuned_feature.detach(), batch,
                                    face_color=fake.detach(),
                                    use_shape_features=self.config.render_shape_features,
                                    detach=True,
                                    resolution=384,
                                    style_latent=z_out)
        rendered_fake = rendered_fake[:, :3, :, :]
        rendered_fake = torch.nn.functional.interpolate(rendered_fake, (512, 512), mode='bilinear',
                                                        align_corners=True)
        p_fake = self.D(self.augment_pipe(rendered_fake))
        fake_loss = torch.nn.functional.softplus(p_fake).mean()
        self.manual_backward(fake_loss + fake_faces_loss)

        p_real = self.D(self.augment_pipe(self.train_set.get_color_bg_real(batch, highres=True)))
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
    def d_regularizer(self, batch, batch_idx):
        d_faces_opt = self.optimizers()[1]
        d_faces_opt.zero_grad(set_to_none=True)
        d_opt = self.optimizers()[3]
        d_opt.zero_grad(set_to_none=True)
        image = self.train_set.get_color_bg_real(batch)
        image.requires_grad_()
        image2 = self.train_set.get_color_bg_real(batch, highres=True)
        image2.requires_grad_()

        p_faces_real = self.D_faces(self.augment_pipe(image, True))
        gp_faces = compute_gradient_penalty(image, p_faces_real)
        disc_faces_loss = self.config.lambda_gp * gp_faces * self.config.lazy_gradient_penalty_interval

        p_real = self.D(self.augment_pipe(image2, True))
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
               transform_matrices=None, camera_delta_angle_array=None, camera_delta_elevation_array=None,
               only_first_view=False, opt_azimuth=None, opt_elevation=None, render_scannet=False,
               unaligned=False, scannet_mode=False, render_nocs=False, render_sil=False):
        if use_shape_features:
            if detach:
                shape_features = to_vertex_features_scatter(batch['shape'][0].detach(), batch)
            else:
                shape_features = to_vertex_features_scatter(batch['shape'][0], batch)
        if face_color is None:
            rendered_color, nocs_image, sillhouette = self.R.render(batch['vertices'],
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
                                           only_first_view=only_first_view,
                                           opt_azimuth=opt_azimuth,
                                           opt_elevation=opt_elevation,
                                           render_scannet=render_scannet,
                                           unaligned=unaligned,
                                           scannet_mode=scannet_mode,
                                           render_nocs=render_nocs,
                                           render_sil=render_sil)
        else:
            rendered_color, nocs_image, sillhouette = self.R.render(batch['vertices'],
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
                                           only_first_view=only_first_view,
                                           opt_azimuth=opt_azimuth,
                                           opt_elevation=opt_elevation,
                                           render_scannet=render_scannet,
                                           unaligned=unaligned,
                                           scannet_mode=scannet_mode,
                                           render_nocs=render_nocs,
                                           render_sil=render_sil
                                           )
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if not render_scannet:
            if resolution is None:
                resolution = self.config.image_size
            if self.config.render_size != resolution:
                ret_val = torch.nn.functional.interpolate(ret_val, (resolution, resolution), mode='bilinear', align_corners=True)
        return ret_val, nocs_image, sillhouette

    # general diff. rendering using PyTorch3D (per-face colors)
    def render_color(self, batch, face_color, use_color_bg=True, resolution=None, norm_params=None,
                     camera_delta_angle=None, transform_matrices=None, camera_delta_angle_array=None):
        rendered_color = self.R_color.render(batch['vertices'],
                                             batch['indices'],
                                             batch["ranges"].cpu(),
                                             to_vertex_colors_scatter(face_color, batch),
                                             color_bg=batch['bg'] if use_color_bg else None,
                                             resolution=self.config.render_size,
                                             norm_params=norm_params,
                                             camera_delta_angle=camera_delta_angle,
                                             transform_matrices=transform_matrices,
                                             camera_delta_angle_array=camera_delta_angle_array)
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if resolution is None:
            resolution = self.config.image_size
        if self.config.render_size != resolution:
            ret_val = torch.nn.functional.interpolate(ret_val, (resolution, resolution), mode='bilinear',
                                                      align_corners=True)
        return ret_val
    
    # old NVDiffRast renderer for per-face colors
    def render_faces(self, face_colors, batch, use_bg_color=True):
        rendered_color = self.R_faces.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), batch['bg'] if use_bg_color else None,
                                       resolution=self.config.render_size)
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if self.config.render_size != self.config.image_size:
            ret_val = torch.nn.functional.interpolate(ret_val, (self.config.image_size, self.config.image_size), mode='bilinear', align_corners=True)
        return ret_val

    # ShapeNet renderer
    def render_shapenet(self, obj_id=None, resolution=None, camera_delta_angle=None, camera_elevation=None):
        rendered_color = self.R_shapenet.render(obj_id=obj_id,
                                                camera_delta_angle=camera_delta_angle,
                                                camera_elevation=camera_elevation)
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
        set_real_sample_resolution(batch)

        # optimize generator
        self.g_step(batch, batch_idx)

        if not self.config.freeze_texturify:
            if self.stable_global_step > self.config.lazy_path_penalty_after and (self.stable_global_step + 1) % self.config.lazy_path_penalty_interval == 0:
                self.g_regularizer(batch, batch_idx)

        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.R.parameters(), max_norm=1.0)

        self.ema.update(self.G.parameters())
        self.ema_r.update(self.R.parameters())

        # optimize discriminator
        self.d_step(batch, batch_idx)

        if (self.stable_global_step + 1) % self.config.lazy_gradient_penalty_interval == 0:
            self.d_regularizer(batch, batch_idx)

        self.execute_ada_heuristics()

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
            odir_real, odir_fake, odir_samples, odir_grid, odir_meshes, output_dir_samples_levels, odir_faces_fake = self.create_directories()
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
                fake_faces_render = self.render_faces(pred_colors, batch, use_bg_color=False).cpu()
                fake_faces_render = self.train_set.cspace_convert_back(fake_faces_render)
                save_image(fake_faces_render, odir_samples / f"faces_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

                fake_render, _, _ = self.render(tuned_feature, batch,
                                          face_color=pred_colors,
                                          use_shape_features=self.config.render_shape_features,
                                          use_color_bg=False,
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
                                                 use_color_bg=False)[:, :3, :, :]
                color_render = self.train_set.cspace_convert_back(color_render[:, :3, ...])
                save_image(color_render, odir_samples / f"facescolor_{iter_idx}.jpg", value_range=(-1, 1),
                           normalize=True)
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
        return GraphDataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers,
                               texopt=self.config.texopt)

    # set validation dataloader
    def val_dataloader(self):
        return GraphDataLoader(self.val_set, self.config.batch_size, shuffle=False, drop_last=True, num_workers=self.config.num_workers,
                               texopt=self.config.texopt)

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
        #     if iter_idx < self.config.num_vis_images // self.config.batch_size:
        #         vis_generated_images.append(fake)
        torch.cuda.empty_cache()
        # vis_generated_images =** torch.cat(vis_generated_images, dim=0)
        # save_image(vis_generated_images, output_dir_vis / f"{prefix}{self.stable_global_step:06d}.png", nrow=int(math.sqrt(vis_generated_images.shape[0])), value_range=(-1, 1), normalize=True)

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
                generated_colors, all_face_colors, feature = self.G(eval_batch['graph_data'], z, eval_batch['shape'], noise_mode='const')
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
        output_dir_samples_levels = Path(f'runs/{self.config.experiment}/images_levels/{self.stable_global_step:06d}')
        output_dir_textures = Path(f'runs/{self.config.experiment}/textures/')
        output_dir_meshes = Path(f'runs/{self.config.experiment}/meshes/{self.stable_global_step:06d}')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures,
                     output_dir_meshes, output_dir_samples_levels, output_dir_fid_faces_fake]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes, output_dir_samples_levels, \
               output_dir_fid_faces_fake
    
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
        output_dir_samples = Path(f'runs/temp_{prefix}/images/{self.stable_global_step:06d}')
        output_dir_samples_levels = Path(f'runs/temp_{prefix}/images_levels/{self.stable_global_step:06d}')
        output_dir_textures = Path(f'runs/temp_{prefix}/textures/')
        output_dir_meshes = Path(f'runs/temp_{prefix}/meshes/{self.stable_global_step:06d}')
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

    # crop texture randomly from an input image, biased towards image mask center
    def crop_texture(self, image, background_value=1):

        bsize, num_channels, height, width = image.shape

        image_mask = (image[:, 0, ...] != background_value) | (image[:, 1, ...] != background_value) | (image[:, 2, ...] != background_value)
        image_mask_coords = torch.stack(torch.where(image_mask)[1:]).T.float()
        bck_mask = (image[:, 0, ...] == background_value) & (image[:, 1, ...] == background_value) & (image[:, 2, ...] == background_value)
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
                           use_nocs=False, nocs_map=None, nocs_value=None, target_texture_side=None,
                           patch_size=100):

        bsize, num_channels, height, width = image.shape

        image_mask = (image[:, 0, ...] != background_value) | (image[:, 1, ...] != background_value) | (image[:, 2, ...] != background_value)
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
            texture_side = int(min(min_side, patch_size / 1024.0 * height))

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
                # print('coords 1', nocs_map_coord, target_texture_side)
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

    # load the model that predicts poses of chairs
    def load_pose_model(self, ckpt_path=None):

        from model.discriminator import PoseClassifier

        pose_classifier = PoseClassifier(out_dim_azimuth=12, out_dim_elevation=5)
        checkpoint = torch.load(ckpt_path)['state_dict']
        checkpoint = {k[11:]: v for k, v in checkpoint.items()}

        pose_classifier.load_state_dict(checkpoint)

        return pose_classifier

    # load the model that predicts NOCs of chairs
    def load_nocs_model(self, ckpt_path=None):

        import segmentation_models_pytorch as smp

        nocs_predictor = smp.create_model(
            arch="Unet",
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=4,
            classes=3
        )
        checkpoint = torch.load(ckpt_path)['state_dict']
        checkpoint = {k[6:]: v for k, v in checkpoint.items()}

        nocs_predictor.load_state_dict(checkpoint)

        return nocs_predictor

    # inference loop, store synthesized renders
    def store_rendered_images(self, savedir=None):
        torch.autograd.set_detect_anomaly(True)
        SETUP = 'random' # random_1chair
        if savedir is None:
            BASE_SAVE_DIR = "/cluster/valinor/abokhovkin/scannet-texturing/data/Projections/15112113_70_chairs"
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
                fake_render, _, _ = self.render(tuned_feature, input_batch,
                                          face_color=pred_colors,
                                          use_shape_features=self.config.render_shape_features,
                                          use_color_bg=False,
                                          style_latent=z_out)
                fake_render = fake_render[:, :3, :, :]
                for k in range(len(fake_render)):
                    save_image(fake_render[k][None, ...],
                               os.path.join(BASE_SAVE_DIR, f"{iter_idx}_{k}_{SETUP}.jpg"), value_range=(-1, 1), normalize=True)
                torch.save(input_latents.cpu().detach(), os.path.join(BASE_SAVE_DIR, f"globlat_{iter_idx}_{SETUP}.pth"))

                with open(os.path.join(BASE_SAVE_DIR, f"meta_{iter_idx}_{SETUP}.json"), 'w') as fout:
                    json.dump(meta_data, fout)

    def optimize_stylemesh_image_vgg_mlp_aligned(
            self, 
            ckpt_ema=None, 
            ckpt_ema_r=None, 
            mode='shapenet',
            vgg_path=None,
            scannet_crops_dir=None,
            legal_scannet_ids_path=None,
            nocs_dir=None,
            shapenet_dir=None,
            prefix=None
        ):

        import torchvision.transforms as transforms
        from PIL import Image
        from util.stylemesh import image_pyramid, ContentAndStyleLoss, pre, post
        from copy import deepcopy

        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()])

        device = torch.device("cuda:0")

        # loss weights (probably need to adjust) and other hyperparams
        base_style_weight = 1e6 # 1e6 / 1.0 
        base_content_weight = 1.0 # 1.0
        base_view_weight = 1.0
        w_patch_loss = 1.0
        w_global_base_loss = 1.0
        num_levels = 3
        num_iterations = 501

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
        if mode not in ['scannet', 'shapenet']:
            raise NotImplementedError(f'Mode {mode} is not implemented')
        
        if prefix is None:
            PREFIX = 'chair_shapenet_500iter_stylemeshvgg_nocspatch100_delta+tuner+shape_layer3'
        else:
            PREFIX = prefix
        odir_real, odir_fake, odir_samples, odir_grid, odir_meshes, output_dir_samples_levels = self.create_store_directories(
            prefix=PREFIX
        )

        len_loader = len(self.val_dataloader())
        collater = Collater([], [], True)

        # load target ScanNet crops for optimization
        if scannet_crops_dir is None:
            SCANNET_CROPS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ScanNet_chair_masked'
        else:
            SCANNET_CROPS_DIR = scannet_crops_dir
        sceneid_to_imgnames = defaultdict(list)
        all_scannet_imagenames = defaultdict(list)
        scene_ids = sorted(os.listdir(SCANNET_CROPS_DIR))
        for scene_id in scene_ids:
            if len(os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id))) > 0:
                img_names = [x for x in os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id)) if 'scancrop' in x]
                sceneid_to_imgnames[scene_id] = img_names
                all_scannet_imagenames[scene_id] = defaultdict(list)
                for img_name in img_names:
                    all_scannet_imagenames[scene_id][img_name.split('_')[1]] += [img_name]
        for scene_id in all_scannet_imagenames.copy():
            for img_id in all_scannet_imagenames[scene_id].copy():
                if len(all_scannet_imagenames[scene_id][img_id]) < 6:
                    all_scannet_imagenames[scene_id].pop(img_id)
            if len(all_scannet_imagenames) == 0:
                all_scannet_imagenames.pop(scene_id)
        used_scene_ids = []
        for scene_id in all_scannet_imagenames:
            used_scene_ids += [scene_id]
        used_scene_ids = sorted(used_scene_ids)
        all_scannet_imagenames_tuples = []
        for scene_id in used_scene_ids:
            for img_id in all_scannet_imagenames[scene_id]:
                all_scannet_imagenames_tuples += [(scene_id, all_scannet_imagenames[scene_id][img_id])]
        random.seed(13)
        random.shuffle(all_scannet_imagenames_tuples)


        legal_scannet_ids = None
        if legal_scannet_ids_path is not None:
            # LEGAL_SCANNET_IDS_PATH = '/cluster/valinor/abokhovkin/scannet-texturing/data/scannet_legal_ids.json'
            LEGAL_SCANNET_IDS_PATH = legal_scannet_ids_path
            with open(LEGAL_SCANNET_IDS_PATH, 'r') as fin:
                legal_scannet_ids = json.load(fin)['ids']

        if mode == 'scannet':
            patch_size = 50
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/chair_scannet_aligned_EVAL'
            else:
                NOCS_DIR = nocs_dir
        else:
            patch_size = 100
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/chair_shapenet_aligned_EVAL'
            else:
                NOCS_DIR = nocs_dir

        if shapenet_dir is None:
            SHAPENET_DIR = '/canis/ShapeNet/ShapeNetCore.v2/03001627'
        else:
            SHAPENET_DIR = shapenet_dir

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)

            all_optimized_images = []
            all_optimized_faces_images = []
            for batch_idx, batch in enumerate(self.val_dataloader()):

                if legal_scannet_ids is not None:
                    if mode == 'scannet' and batch_idx not in legal_scannet_ids:
                        continue

                # if batch_idx > 512:
                #     break

                print(f'Image #{batch_idx}')
                set_real_sample_resolution(batch)
                obj_id = batch['shapenet_id'][0]

                norm_params = None
                camera_delta_angle = None
                transform_matrices = None

                if mode == 'shapenet':
                    scannet_mode = False
                    np.random.seed(13)
                    camera_delta_angle = float(np.random.uniform(-np.pi / 2, np.pi / 2))

                    np.random.seed(14)
                    camera_delta_angle_array_gt = np.random.uniform(-np.pi / 2, np.pi / 2, size=(6))
                    camera_delta_angle_array_gt[4] = camera_delta_angle
                    # camera_delta_angle_array_gt[4] = camera_delta_angle - np.pi / 6 # deviated

                    mesh_shapenet = trimesh.load(f'{SHAPENET_DIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                    if len(mesh_shapenet.vertices) > 25000:
                        continue
                    mesh_shapenet_mean = np.zeros(4)
                    mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                    mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                    mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                    norm_params = (mesh_shapenet_mean, mesh_shapenet_max)
                    images_target = []
                    for view_id in range(len(camera_delta_angle_array_gt)):
                        image_target = self.render_shapenet(obj_id, camera_delta_angle=camera_delta_angle_array_gt[view_id])
                        images_target += [image_target]
                    images_target = torch.cat(images_target, dim=0)

                    t = 1000 * time.time()
                    np.random.seed(int(t) % 2 ** 32)

                    meta_store = {
                        'shape_id': batch['shapenet_id'][0],
                        'sample_name': batch['name'][0],
                        'azimuth': [float(x) for x in camera_delta_angle_array_gt]
                    }


                    # Photoshape (additional mode if you want to try)
                    # image_target = self.train_set.get_color_bg_real(batch, highres=True)
                    # image_target = torch.nn.functional.interpolate(image_target, (512, 512), mode='bilinear',
                    #                                                align_corners=True)
                    # image_target = image_target[0:1]
                    # image_target = (image_target + 1.0) / 2.0

                elif mode == 'scannet':
                    scannet_mode = True
                    np.random.seed(13)
                    scene_id, instance_img_names = all_scannet_imagenames_tuples[batch_idx]
                    random_img_names = np.random.choice(instance_img_names, 6, replace=False)
                    images_target = []
                    all_meta_datas = []
                    for k, img_name in enumerate(random_img_names):
                        image_target = Image.open(os.path.join(SCANNET_CROPS_DIR, scene_id, img_name))
                        image_target = preprocess(image_target).unsqueeze(0).to('cuda')

                        k_img_id = img_name.split('scancrop')[0][:-1]
                        k_meta_data = json.loads(Path(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{k_img_id}_meta.json')).read_text())
                        all_meta_datas += [k_meta_data]

                        if k == 4:
                            img_id = img_name.split('scancrop')[0][:-1]
                            meta_data = json.loads(Path(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{img_id}_meta.json')).read_text())
                        images_target += [image_target]
                    images_target = torch.cat(images_target, dim=0)

                    obj_id = meta_data['model']['id_cad']
                    camera_pose = np.array(meta_data['pose'])
                    camera_pose[:3, 3] = 0
                    camera_pose = np.linalg.inv(camera_pose)
                    camera_pose = torch.FloatTensor(camera_pose).cuda()
                    model_to_scan = np.array(meta_data['model_to_scan'])
                    model_to_scan[:3, 3] = 0
                    model_to_scan = torch.FloatTensor(model_to_scan).cuda()
                    transform_matrices = (model_to_scan, camera_pose)
                    camera_delta_angle = 0.0

                    mesh_shapenet = trimesh.load(f'{SHAPENET_DIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                    mesh_shapenet_mean = np.zeros(4)
                    mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                    mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                    mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                    norm_params = (mesh_shapenet_mean, mesh_shapenet_max)
                    t = 1000 * time.time()
                    np.random.seed(int(t) % 2 ** 32)

                    meta_store = {
                        'shape_id': batch['shapenet_id'][0],
                        'sample_name': batch['name'][0],
                        'image_filenames': [str(x) for x in random_img_names],
                        'scene_id': scene_id
                    }

                try:
                    gt_nocs = np.load(os.path.join(NOCS_DIR, f'nocs_gt_{batch_idx}.npy'))
                    gt_nocs = torch.FloatTensor(gt_nocs).cuda()
                except FileNotFoundError:
                    print(f'NOCs image {os.path.join(NOCS_DIR, f'nocs_gt_{batch_idx}.npy')} not found')
                    continue

                try:
                    batch = self.train_set.get_by_shapenet_id(obj_id)
                    batch = collater([batch])
                except (ValueError, KeyError):
                    print(f'The sample with shapenet id {obj_id} was not found!')
                    continue

                batch = to_device(batch, 'cuda')
                self.E = self.E.to('cuda')
                self.D = self.D.to('cuda')
                self.set_shape_codes(batch)
                shape = batch['shape']

                image_target_clip = images_target[4].to('cuda')
                image_target_clip = pre()(image_target_clip)
                image_target_clip = image_target_clip[None, ...]

                image_target_mask = torch.where((image_target_clip[:, 0] > 0.99) & (image_target_clip[:, 1] > 0.99) & (image_target_clip[:, 2] > 0.99), 0, 1)
                image_target_mask = image_target_mask[:, None, ...]
                image_target_mask = image_target_mask.float()
                gt_nocs_mask = torch.cat([gt_nocs, image_target_mask], dim=1)

                # StyleLoss.set_style_image(image_target_clip, num_levels)
                StyleLoss.set_style_image(image_target_clip, num_levels, image_target_mask)

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
                lr_mlp = 1e-2

                optimizer_latent = torch.optim.Adam([
                    {'params': [latent], 'lr': lr_latent, 'betas': (0.9, 0.99), 'eps': 1e-8}
                ])
                optimizer = torch.optim.Adam([
                    {'params': [init_rand_face_feature], 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': list(self.G.synthesis.tuner.parameters()), 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    # {'params': list(self.R.parameters()), 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': init_shape, 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8}
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

                        pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(batch['graph_data'],
                                                                       latent, shape, noise_mode='const')

                        np.random.seed(15)
                        camera_delta_angle_array = np.random.uniform(-np.pi / 2, np.pi / 2, self.config.views_per_sample)
                        camera_delta_angle_array[0] = camera_delta_angle
                        # camera_delta_elevation_array = [np.pi - np.pi / 18] * self.config.views_per_sample
                        # camera_delta_angle_array = None
                        t = 1000 * time.time()
                        np.random.seed(int(t) % 2 ** 32)

                        # render faces (not useful)
                        fake_render_faces = self.render_color(batch,
                                                              face_color=pred_colors,
                                                              use_color_bg=False,
                                                              norm_params=norm_params,
                                                              camera_delta_angle=camera_delta_angle,
                                                              transform_matrices=transform_matrices,
                                                              camera_delta_angle_array=camera_delta_angle_array)
                        fake_render_faces_mask = fake_render_faces[:, 3:4, :, :]
                        fake_render_faces_mask = torch.nn.functional.interpolate(fake_render_faces_mask, (512, 512),
                                                                                 mode='nearest')
                        fake_render_faces = fake_render_faces[:, :3, :, :]
                        fake_render_faces_reshaped = torch.nn.functional.interpolate(fake_render_faces, (512, 512),
                                                                                     mode='bilinear', align_corners=True)
                        fake_render_faces_reshaped = (fake_render_faces_reshaped + 1.0) / 2.0

                        if iter_idx < 200:
                            opt_features = tuned_feature
                            w_global_loss = w_global_base_loss
                        else:
                            opt_features = tuned_feature + init_rand_face_feature
                            w_global_loss = 0.0

                        if iter_idx in [0, 200, 300, 400, num_iterations - 1]:
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
                            norm_params=norm_params,
                            camera_delta_angle=camera_delta_angle,
                            transform_matrices=transform_matrices,
                            camera_delta_angle_array=camera_delta_angle_array,
                            # camera_delta_elevation_array=camera_delta_elevation_array,
                            only_first_view=only_first_view,
                            scannet_mode=scannet_mode,
                            render_nocs=True,
                            render_sil=False
                        )
                        fake_render_mask = fake_render[:, 3:4, :, :]
                        fake_render = fake_render[:, :3, :, :]
                        fake_render_reshaped = torch.nn.functional.interpolate(fake_render, (512, 512),
                                                                               mode='bilinear', align_corners=True)
                        fake_render_reshaped = (fake_render_reshaped + 1.0) / 2.0
                        fake_render_reshaped_save = fake_render_reshaped

                        image_ref = torch.permute(image_ref, (0, 3, 1, 2))
                        image_ref_mask = image_ref[:, 3:4, ...]
                        image_ref = image_ref[:, :3, ...]
                        image_ref = torch.nn.functional.interpolate(image_ref, (512, 512),
                                                                    mode='bilinear', align_corners=True)
                        image_ref_mask = torch.nn.functional.interpolate(image_ref_mask, (512, 512),
                                                                         mode='nearest')
                        pred_nocs = image_ref
                        scale = gt_nocs.max() / pred_nocs.max()
                        pred_nocs = pred_nocs * scale
                        pred_nocs_mask = torch.cat([image_ref, image_ref_mask[0:1]], dim=1)

                        p_fake = self.D(fake_render)
                        gen_loss = torch.nn.functional.softplus(-p_fake).mean()

                        preprocessed = []
                        for k_view in range(len(fake_render_reshaped)):
                            fake_render_reshaped_view = pre()(fake_render_reshaped[k_view])
                            preprocessed += [fake_render_reshaped_view[None, ...]]
                        fake_render_reshaped = torch.cat(preprocessed, dim=0)


                        # Global loss branch
                        global_loss = 0.0
                        for k_view in range(len(fake_render_reshaped)):

                            fake_pyramid = image_pyramid(fake_render_reshaped[k_view][None, ...], np.arange(num_levels), reverse=True)
                            fake_mask_pyramid = image_pyramid(fake_render_mask[k_view][None, ...], np.arange(num_levels), reverse=True, mode='nearest')

                            style_loss, content_loss, pyramid = StyleLoss(fake_pyramid, image_target_clip, fake_mask_pyramid)

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
                        num_patches_per_view = 1 # 1 / 4

                        for k_patch in range(num_patches_per_view):
                            target_render_crop_, target_mask_crop_, target_coords, target_nocs_value, target_texture_side = self.crop_texture_cover(
                                image_target_clip,
                                1,
                                coords=None,
                                nocs_map=gt_nocs_mask,
                                use_nocs=True,
                                patch_size=patch_size)

                            target_mask_crop_ = target_mask_crop_.float()
                            target_render_crop = torch.nn.functional.interpolate(target_render_crop_, (512, 512),
                                                                                 mode='bilinear', align_corners=True)
                            target_mask_crop = torch.nn.functional.interpolate(target_mask_crop_, (512, 512),
                                                                               mode='nearest')
                            target_render_crop_save = torch.nn.functional.interpolate(target_render_crop_, (128, 128),
                                                                                      mode='bilinear', align_corners=True)
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
                                target_texture_side=target_texture_side,
                                patch_size=patch_size)

                            fake_render_mask_crop_ = fake_render_mask_crop_.float()
                            fake_render_crop = torch.nn.functional.interpolate(fake_render_crop_, (512, 512),
                                                                               mode='bilinear', align_corners=True)
                            fake_render_mask_crop = torch.nn.functional.interpolate(fake_render_mask_crop_, (512, 512),
                                                                                    mode='nearest')
                            fake_render_crop_save = torch.nn.functional.interpolate(fake_render_crop_, (128, 128),
                                                                               mode='bilinear', align_corners=True)
                            fake_render_mask_crop_save = torch.nn.functional.interpolate(fake_render_mask_crop_, (128, 128),
                                                                                         mode='nearest')

                            fake_render_crops_save += [fake_render_crop_save]
                            fake_render_crops += [fake_render_crop]
                            fake_render_mask_crops_save += [fake_render_mask_crop_save]
                            fake_render_mask_crops += [fake_render_mask_crop]

                        fake_patch_pyramids = []
                        fake_render_patch_mask_pyramids = []
                        for k in range(len(fake_render_crops)):
                            fake_patch_pyramid = image_pyramid(fake_render_crops[k], np.arange(num_levels), reverse=True)
                            fake_render_patch_mask_pyramid = image_pyramid(fake_render_mask_crops[k], np.arange(num_levels),
                                                                           reverse=True, mode='nearest')
                            fake_patch_pyramids += [fake_patch_pyramid]
                            fake_render_patch_mask_pyramids += [fake_render_patch_mask_pyramid]

                        StylePatchLoss.set_style_image_multiview(torch.cat(gt_render_crops), num_levels,
                                                                 torch.cat(gt_render_mask_crops))

                        style_loss, content_loss = StylePatchLoss.forward_multiview(fake_patch_pyramids, gt_render_crops,
                                                                                    fake_render_patch_mask_pyramids)

                        # style_weight = base_style_weight if k_view != 0 else 0.0
                        style_weight = base_style_weight
                        # content_weight = base_content_weight if k_view == 0 else 0.0
                        content_weight = base_content_weight

                        patch_loss = patch_loss + (style_weight * style_loss + content_weight * content_loss)


                        if iter_idx % 10 == 0:
                            print(f'LOSSES: {global_loss} (global), {gen_loss} (disc.)')

                        loss = (w_global_loss * global_loss + w_patch_loss * patch_loss) / 1.0
                        loss.backward(retain_graph=True)

                        all_content_losses += [content_loss.cpu().detach()]
                        all_style_losses += [style_loss.cpu().detach()]
                        all_latent_norms += [torch.sum(latent ** 2).cpu().detach()]

                        if iter_idx in [0, 200, 300, 400, num_iterations - 1]:

                            with open(str(odir_samples / f"metadata_{batch_idx}.json"), 'w') as fout:
                                json.dump(meta_store, fout)

                            save_image(fake_render_reshaped.cpu().detach(),
                                       odir_samples / f"fake_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)
                            # save_image(pred_nocs.cpu().detach(),
                            #            odir_samples / f"fakenocs_{batch_idx}_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                            # save_image(gt_nocs.cpu().detach(),
                            #            odir_samples / f"gtnocs_{batch_idx}_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

                            if mode == 'shapenet':
                                camera_delta_elevation_array_gt = [np.pi - np.pi / 18] * self.config.views_per_sample
                                camera_delta_angle_array_gt = [0,
                                                               np.pi / 4,
                                                               45 * np.pi / 180,
                                                               70 * np.pi / 180,
                                                               120 * np.pi / 180,
                                                               210 * np.pi / 180]
                                mesh_shapenet_mean = np.zeros(4)
                                mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                                mesh_shapenet_max = 0.32
                                norm_params_gt = (mesh_shapenet_mean, mesh_shapenet_max)
                                fake_render, _, _ = self.render(opt_features, batch,
                                                                face_color=pred_colors,
                                                                use_shape_features=True,
                                                                use_color_bg=False,
                                                                style_latent=z_out,
                                                                norm_params=norm_params_gt,
                                                                camera_delta_angle=camera_delta_angle,
                                                                transform_matrices=transform_matrices,
                                                                camera_delta_angle_array=camera_delta_angle_array_gt,
                                                                camera_delta_elevation_array=camera_delta_elevation_array_gt,
                                                                only_first_view=False,
                                                                scannet_mode=scannet_mode,
                                                                render_nocs=False,
                                                                render_sil=False)
                                fake_render_mask = fake_render[:, 3:4, :, :]
                                fake_render = fake_render[:, :3, :, :]
                                fake_render_reshaped = torch.nn.functional.interpolate(fake_render, (512, 512),
                                                                                       mode='bilinear', align_corners=True)
                                fake_render_reshaped = (fake_render_reshaped + 1.0) / 2.0
                                save_image(fake_render_reshaped.cpu().detach(),
                                           odir_samples / f"fakeasgt_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)

                            elif mode == 'scannet':
                                meta_store['model_to_scan'] = []
                                meta_store['camera_pose'] = []

                                all_scannet_renders = []
                                for meta_data in all_meta_datas:
                                    camera_pose = np.array(meta_data['pose'])
                                    camera_pose = np.linalg.inv(camera_pose)
                                    camera_pose = torch.FloatTensor(camera_pose).cuda()
                                    model_to_scan = np.array(meta_data['model_to_scan'])
                                    model_to_scan = torch.FloatTensor(model_to_scan).cuda()
                                    scannet_transform_matrices = (model_to_scan, camera_pose)

                                    meta_store['model_to_scan'] += [model_to_scan.cpu().numpy().tolist()]
                                    meta_store['camera_pose'] += [camera_pose.cpu().numpy().tolist()]

                                    render, _, _ = self.render(opt_features, batch,
                                                             face_color=pred_colors,
                                                             use_shape_features=True,
                                                             use_color_bg=False,
                                                             style_latent=z_out,
                                                             norm_params=norm_params,
                                                             camera_delta_angle=camera_delta_angle,
                                                             transform_matrices=scannet_transform_matrices,
                                                             camera_delta_angle_array=camera_delta_angle_array,
                                                             # camera_delta_elevation_array=camera_delta_elevation_array,
                                                             only_first_view=True,
                                                             render_scannet=True,
                                                             )
                                    render = render[0:1, :3, :, :]
                                    render = (render + 1.0) / 2.0
                                    render = torch.nn.functional.interpolate(render, (968, 1296),
                                                                                mode='bilinear', align_corners=True)
                                    all_scannet_renders += [render]
                                save_image(torch.cat(all_scannet_renders, dim=0).cpu().detach(),
                                           odir_samples / f"scannetfake_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)

                        if iter_idx % 10 == 0:
                            print("Iter {}:".format(iter_idx))
                            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                                style_loss.item(), content_loss.item()))
                            print('Saving to:', odir_samples)
                            print()
                        if iter_idx % 50 == 0:
                            all_images += [fake_render_reshaped[0:1, ...].cpu().detach()]
                            # all_faces_images += [fake_render_faces_reshaped.cpu().detach()]
                        if iter_idx % 20 == 0:
                            target_crops += [gt_render_crops_save[0].cpu().detach()]
                            fake_crops += [fake_render_crops_save[0].cpu().detach()]
                            target_mask_crops += [gt_render_mask_crops_save[0].cpu().detach()]
                            fake_mask_crops += [fake_render_mask_crops_save[0].cpu().detach()]

                        # Additional outputs in case you need them
                        # if iter_idx == num_iterations - 1:
                        #     save_image(fake_render_reshaped_save.cpu().detach(),
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
                            all_content_losses = []
                            all_style_losses = []
                            all_latent_norms = []
                            target_crops = []
                            target_mask_crops = []
                            fake_crops = []
                            fake_mask_crops = []

                        return 1.0 * (style_loss + content_loss)

                    # optimizer_latent.step(closure)
                    # optimizer_latent.zero_grad()

                    if iter_idx < 200:
                        optimizer_latent.step(closure)
                        optimizer_latent.zero_grad()
                    else:
                        optimizer.step(closure)
                        optimizer.zero_grad()

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

            # compute_multiview_score(str(odir_samples), self.config.views_per_sample)

    def optimize_stylemesh_image_vgg_mlp_unaligned(
            self, 
            ckpt_ema=None, 
            ckpt_ema_r=None, 
            mode='shapenet', 
            transfer=False,
            vgg_path=None,
            scannet_crops_dir=None,
            legal_scannet_ids_path=None,
            nocs_dir=None,
            shapenet_dir=None,
            pose_model_ckpt_path=None,
            shapenet_ids_path=None,
            prefix=None
        ):

        def adjust_learning_rate(
                initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
        ):
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            return lr

        import torchvision.transforms as transforms
        from PIL import Image
        from util.stylemesh import image_pyramid, ContentAndStyleLoss, pre, post
        from copy import deepcopy

        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()])
        
        device = torch.device("cuda:0")

        # loss weights (probably need to adjust) and other hyperparams
        base_style_weight = 1e6 # 1e6 / 1.0 
        base_content_weight = 1.0  # 1.0
        base_view_weight = 1.0
        w_patch_loss = 1.0
        w_global_base_loss = 1.0
        num_levels = 3
        num_iterations = 501

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

        print('Mode', mode)
        if mode not in ['scannet', 'shapenet']:
            raise NotImplementedError(f'Mode {mode} is not implemented')
        
        if prefix is None:
            PREFIX = 'chair_shapenet_500iter_stylemeshvgg_nocspatch50_delta+tuner+shape_layer3_unaligned'
        else:
            PREFIX = prefix
        odir_real, odir_fake, odir_samples, odir_grid, odir_meshes, output_dir_samples_levels = self.create_store_directories(
            prefix=PREFIX
        )

        len_loader = len(self.val_dataloader())
        collater = Collater([], [], True)

        # load target ScanNet crops for optimization
        if scannet_crops_dir is None:
            SCANNET_CROPS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ScanNet_chair_masked'
        else:
            SCANNET_CROPS_DIR = scannet_crops_dir
        sceneid_to_imgnames = defaultdict(list)
        all_scannet_imagenames = defaultdict(list)
        scene_ids = sorted(os.listdir(SCANNET_CROPS_DIR))
        for scene_id in scene_ids:
            if len(os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id))) > 0:
                img_names = [x for x in os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id)) if 'scancrop' in x]
                sceneid_to_imgnames[scene_id] = img_names
                all_scannet_imagenames[scene_id] = defaultdict(list)
                for img_name in img_names:
                    all_scannet_imagenames[scene_id][img_name.split('_')[1]] += [img_name]
        for scene_id in all_scannet_imagenames.copy():
            for img_id in all_scannet_imagenames[scene_id].copy():
                if len(all_scannet_imagenames[scene_id][img_id]) < 6:
                    all_scannet_imagenames[scene_id].pop(img_id)
            if len(all_scannet_imagenames) == 0:
                all_scannet_imagenames.pop(scene_id)
        used_scene_ids = []
        for scene_id in all_scannet_imagenames:
            used_scene_ids += [scene_id]
        used_scene_ids = sorted(used_scene_ids)
        all_scannet_imagenames_tuples = []
        for scene_id in used_scene_ids:
            for img_id in all_scannet_imagenames[scene_id]:
                all_scannet_imagenames_tuples += [(scene_id, all_scannet_imagenames[scene_id][img_id])]
        random.seed(13)
        random.shuffle(all_scannet_imagenames_tuples)


        legal_scannet_ids = None
        if legal_scannet_ids_path is not None:
            # LEGAL_SCANNET_IDS_PATH = '/cluster/valinor/abokhovkin/scannet-texturing/data/scannet_legal_ids.json'
            LEGAL_SCANNET_IDS_PATH = legal_scannet_ids_path
            with open(LEGAL_SCANNET_IDS_PATH, 'r') as fin:
                legal_scannet_ids = json.load(fin)['ids']


        if mode == 'scannet':
            patch_size = 50
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/chair_scannet_aligned_EVAL'
            else:
                NOCS_DIR = nocs_dir
        elif mode == 'shapenet' and not transfer:
            patch_size = 100
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/chair_shapenet_unaligned_EVAL'
            else:
                NOCS_DIR = nocs_dir
        elif mode == 'shapenet' and transfer:
            patch_size = 100
            if nocs_dir is None:
                NOCS_DIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/pred_nocs/chair_shapenet_unalignedtrans_EVAL'
            else:
                NOCS_DIR = nocs_dir
            # path to the list with used ShapeNet ids
            if shapenet_ids_path is None:
                SHAPENET_IDS_PATH = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/used_obj_ids.json'
            else:
                SHAPENET_IDS_PATH = None
            with open(SHAPENET_IDS_PATH, 'r') as fin:
                used_obj_ids = json.load(fin)['obj_ids']

        if shapenet_dir is None:
            SHAPENET_DIR = '/canis/ShapeNet/ShapeNetCore.v2/03001627'
        else:
            SHAPENET_DIR = shapenet_dir

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)

            all_optimized_images = []
            all_optimized_faces_images = []
            for batch_idx, batch in enumerate(self.val_dataloader()):

                print('len loader', len(self.val_dataloader()))

                if legal_scannet_ids is not None:
                    if mode == 'scannet' and batch_idx not in legal_scannet_ids:
                        continue

                # if batch_idx > 512:
                #     continue

                print(f'Image #{batch_idx}')
                set_real_sample_resolution(batch)
                input_obj_id = batch['shapenet_id'][0]

                norm_params = None
                camera_delta_angle = None
                transform_matrices = None

                if mode == 'shapenet':
                    scannet_mode = False
                    np.random.seed(13)
                    camera_delta_angle = float(np.random.uniform(-np.pi / 2, np.pi / 2))

                    if pose_model_ckpt_path is None:
                        POSE_MODEL_CKPT = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/23021410_ImageLatent_chair_scannet_pose_resnet18_lr0.0003/checkpoints/_epoch=9.ckpt'
                    else:
                        POSE_MODEL_CKPT = pose_model_ckpt_path
                    pose_model = self.load_pose_model(POSE_MODEL_CKPT)
                    pose_model = pose_model.cuda()
                    pose_model.eval()

                    np.random.seed(14)
                    if transfer:
                        obj_id = used_obj_ids[(batch_idx + 5) % len(used_obj_ids)]  # style transfer
                    else:
                        obj_id = input_obj_id # not style transfer, different shape
                    camera_delta_angle_array_gt = np.random.uniform(-np.pi / 2, np.pi / 2, size=(6))
                    mesh_shapenet = trimesh.load(
                        f'{SHAPENET_DIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                    if len(mesh_shapenet.vertices) > 25000:
                        continue
                    mesh_shapenet_mean = np.zeros(4)
                    mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                    mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                    mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                    norm_params = (mesh_shapenet_mean, mesh_shapenet_max)
                    images_target = []
                    for view_id in range(len(camera_delta_angle_array_gt)):
                        image_target = self.render_shapenet(obj_id, camera_delta_angle=camera_delta_angle_array_gt[view_id])
                        images_target += [image_target]
                    images_target = torch.cat(images_target, dim=0)
                    t = 1000 * time.time()
                    np.random.seed(int(t) % 2 ** 32)

                    # Photoshape (additional mode if you want to try)
                    # image_target = self.train_set.get_color_bg_real(batch, highres=True)
                    # image_target = torch.nn.functional.interpolate(image_target, (512, 512), mode='bilinear',
                    #                                                align_corners=True)
                    # image_target = image_target[0:1]
                    # image_target = (image_target + 1.0) / 2.0

                elif mode == 'scannet':
                    scannet_mode = False
                    np.random.seed(13)
                    scene_id, instance_img_names = all_scannet_imagenames_tuples[batch_idx]
                    random_img_names = np.random.choice(instance_img_names, 6, replace=False)

                    if pose_model_ckpt_path is None:
                        POSE_MODEL_CKPT = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/23021410_ImageLatent_chair_scannet_pose_resnet18_lr0.0003/checkpoints/_epoch=9.ckpt'
                    else:
                        POSE_MODEL_CKPT = pose_model_ckpt_path
                    pose_model = self.load_pose_model(POSE_MODEL_CKPT)
                    pose_model = pose_model.cuda()
                    pose_model.eval()
    
                    images_target = []
                    all_meta_datas = []
                    for k, img_name in enumerate(random_img_names):
                        image_target = Image.open(os.path.join(SCANNET_CROPS_DIR, scene_id, img_name))
                        image_target = preprocess(image_target).unsqueeze(0).to('cuda')

                        k_img_id = img_name.split('scancrop')[0][:-1]
                        k_meta_data = json.loads(
                            Path(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{k_img_id}_meta.json')).read_text())
                        all_meta_datas += [k_meta_data]

                        if k == 4:
                            img_id = img_name.split('scancrop')[0][:-1]
                            meta_data = json.loads(
                                Path(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{img_id}_meta.json')).read_text())
                        images_target += [image_target]
                    images_target = torch.cat(images_target, dim=0)

                    if not transfer:
                        obj_id = meta_data['model']['id_cad'] # not style transfer
                    else:
                        obj_id = used_obj_ids[batch_idx % len(used_obj_ids)]  # style transfer, different shape
                    camera_pose = np.array(meta_data['pose'])
                    camera_pose[:3, 3] = 0
                    camera_pose = np.linalg.inv(camera_pose)
                    camera_pose = torch.FloatTensor(camera_pose).cuda()
                    model_to_scan = np.array(meta_data['model_to_scan'])
                    model_to_scan[:3, 3] = 0
                    model_to_scan = torch.FloatTensor(model_to_scan).cuda()
                    transform_matrices = None
                    camera_delta_angle = 0.0

                    mesh_shapenet = trimesh.load(
                        f'{SHAPENET_DIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                    mesh_shapenet_mean = np.zeros(4)
                    mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                    mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                    mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                    norm_params = (mesh_shapenet_mean, mesh_shapenet_max)
                    print('max y', mesh_shapenet_max)
                    input_obj_id = obj_id
                    t = 1000 * time.time()
                    np.random.seed(int(t) % 2 ** 32)

                    meta_store = {
                        'shape_id': batch['shapenet_id'][0],
                        'sample_name': batch['name'][0],
                        'image_filenames': [str(x) for x in random_img_names],
                        'scene_id': scene_id
                    }

                    print(meta_store)

                try:
                    gt_nocs = np.load(os.path.join(NOCS_DIR, f'nocs_gt_{batch_idx}.npy'))
                    gt_nocs = torch.FloatTensor(gt_nocs).cuda()
                except FileNotFoundError:
                    print(f'NOCs image {os.path.join(NOCS_DIR, f'nocs_gt_{batch_idx}.npy')} not found')
                    continue

                try:
                    batch = self.train_set.get_by_shapenet_id(input_obj_id)
                    batch = collater([batch])
                except ValueError:
                    print(f'The sample with shapenet id {input_obj_id} was not found!')
                    continue

                batch = to_device(batch, 'cuda')
                self.E = self.E.to('cuda')
                self.D = self.D.to('cuda')
                self.set_shape_codes(batch)
                shape = batch['shape']

                image_target_clip = images_target[4].to('cuda')
                image_target_clip = pre()(image_target_clip)
                image_target_clip = image_target_clip[None, ...]

                # predict pose for target shape
                azimuth_logits, elevation_logits = pose_model(image_target_clip)
                camera_azimuths = [0.0, 1 / 6 * np.pi, 2 / 6 * np.pi, 3 / 6 * np.pi, 4 / 6 * np.pi, 5 / 6 * np.pi,
                                   6 / 6 * np.pi, 7 / 6 * np.pi, 8 / 6 * np.pi, 9 / 6 * np.pi, 10 / 6 * np.pi,
                                   11 / 6 * np.pi]
                camera_elevations = [10 / 12 * np.pi, 11 / 12 * np.pi, np.pi, 13 / 12 * np.pi, 14 / 12 * np.pi]
                init_azimuth = camera_azimuths[int(torch.argmax(azimuth_logits).cpu())]
                init_elevation = camera_elevations[int(torch.argmax(elevation_logits).cpu())]

                image_target_mask = torch.where((image_target_clip[:, 0] > 0.99) & (image_target_clip[:, 1] > 0.99) & (image_target_clip[:, 2] > 0.99), 0, 1)
                image_target_mask = image_target_mask[:, None, ...]
                image_target_mask = image_target_mask.float()
                gt_nocs_mask = torch.cat([gt_nocs, image_target_mask], dim=1)

                # StyleLoss.set_style_image(image_target_clip, num_levels)
                StyleLoss.set_style_image(image_target_clip, num_levels, image_target_mask)

                self.R.create_angles(init_azimuth=None, init_elevation=init_elevation)

                # define optimizable parameters
                init_rand_face_feature = torch.randn((24576, 256)).cuda() / 25.
                init_rand_face_feature.requires_grad = True
                latent = torch.randn((1, 512)).cuda() # random latent
                latent = latent.cuda()
                latent.requires_grad = True
                init_shape = [shape[i].clone().detach()[:].cuda() for i in range(len(shape))]
                for i in range(len(init_shape)):
                    init_shape[i].requires_grad = True

                lr_latent = 1e-2
                lr_mlp = 1e-2
                lr_pose = 0

                optimizer_latent = torch.optim.Adam([
                    {'params': [latent], 'lr': lr_latent, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': [self.R.camera_azimuth], 'lr': lr_pose, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': [self.R.camera_elevation], 'lr': lr_pose, 'betas': (0.9, 0.99), 'eps': 1e-8}
                ])

                torch.manual_seed(14)
                opt_azimuth = torch.randn((1)).cuda()
                opt_azimuth.requires_grad = True
                opt_elevation = torch.randn((1)).cuda()
                opt_elevation.requires_grad = True
                optimizer = torch.optim.Adam([
                    {'params': [init_rand_face_feature], 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': list(self.G.synthesis.tuner.parameters()), 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    # {'params': list(self.R.parameters()), 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                    {'params': init_shape, 'lr': lr_mlp, 'betas': (0.9, 0.99), 'eps': 1e-8},
                ])
                # list(self.G.synthesis.tuner.parameters()) + [init_rand_face_feature]

                decreased_by = 10
                adjust_lr_every = 450

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
                scale = 1.0
                pred_nocs, pred_nocs_mask = None, None
                for iter_idx in range(num_iterations):

                    adjust_learning_rate(lr_mlp, optimizer, iter_idx, decreased_by, adjust_lr_every)
                    # adjust_learning_rate(lr_latent, optimizer, iter_idx, decreased_by, adjust_lr_every)

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
                        nonlocal gt_nocs, pred_nocs, pred_nocs_mask
                        nonlocal scale

                        t = 1000 * time.time()
                        torch.manual_seed(int(t) % 2 ** 32)

                        pred_colors, all_face_colors, feature, tuned_feature, z_out = self.G(batch['graph_data'],
                                                                       latent, shape, noise_mode='const')

                        np.random.seed(batch_idx + 1)
                        camera_delta_angle_array = np.random.uniform(-np.pi / 2, np.pi / 2, self.config.views_per_sample)
                        camera_delta_elevation_array = [np.pi - np.pi / 18] * self.config.views_per_sample
                        # camera_delta_elevation_array = [np.pi] * self.config.views_per_sample
                        # camera_delta_angle_array[0] = camera_delta_angle
                        # camera_delta_angle_array = None
                        t = 1000 * time.time()
                        np.random.seed(int(t) % 2 ** 32)

                        camera_delta_angle_array[0] = self.R.camera_azimuth.cpu().detach().numpy()
                        camera_delta_elevation_array[0] = self.R.camera_elevation.cpu().detach().numpy()

                        # render faces (not useful)
                        fake_render_faces = self.render_color(batch,
                                                              face_color=pred_colors,
                                                              use_color_bg=False,
                                                              norm_params=norm_params,
                                                              camera_delta_angle=camera_delta_angle,
                                                              transform_matrices=transform_matrices,
                                                              camera_delta_angle_array=camera_delta_angle_array)
                        fake_render_faces_mask = fake_render_faces[:, 3:4, :, :]
                        fake_render_faces = fake_render_faces[:, :3, :, :]
                        fake_render_faces_reshaped = torch.nn.functional.interpolate(fake_render_faces, (512, 512),
                                                                                     mode='bilinear', align_corners=True)
                        fake_render_faces_reshaped = (fake_render_faces_reshaped + 1.0) / 2.0

                        if iter_idx < 200:
                            opt_features = tuned_feature
                            w_global_loss = w_global_base_loss
                        else:
                            opt_features = tuned_feature + init_rand_face_feature
                            w_global_loss = 0.0

                        if iter_idx in [0, 200, 300, 400, num_iterations - 1]:
                            only_first_view = False
                        else:
                            only_first_view = True

                        if iter_idx < 100:
                            render_nocs = True
                            render_sil = True
                        else:
                            render_nocs = False
                            render_sil = False

                        fake_render, image_ref, sillhouette = self.render(
                            opt_features, 
                            batch,
                            face_color=pred_colors,
                            use_shape_features=True,
                            use_color_bg=False,
                            style_latent=z_out,
                            norm_params=norm_params,
                            camera_delta_angle=camera_delta_angle,
                            transform_matrices=transform_matrices,
                            camera_delta_angle_array=camera_delta_angle_array,
                            camera_delta_elevation_array=camera_delta_elevation_array,
                            only_first_view=only_first_view,
                            scannet_mode=scannet_mode,
                            unaligned=True,
                            render_nocs=render_nocs,
                            render_sil=render_sil
                        )
                        fake_render_mask = fake_render[:, 3:4, :, :]
                        fake_render = fake_render[:, :3, :, :]
                        p_fake = self.D(fake_render)
                        gen_loss = torch.nn.functional.softplus(-p_fake).mean()
                        fake_render_reshaped = torch.nn.functional.interpolate(fake_render, (512, 512), mode='bilinear', align_corners=True)
                        fake_render_reshaped = (fake_render_reshaped + 1.0) / 2.0
                        fake_render_reshaped_save = fake_render_reshaped

                        if iter_idx < 100:
                            image_ref = torch.permute(image_ref, (0, 3, 1, 2))
                            image_ref_mask = image_ref[:, 3:4, ...]
                            image_ref = image_ref[:, :3, ...]
                            image_ref = torch.nn.functional.interpolate(image_ref, (512, 512),
                                                                        mode='bilinear', align_corners=True)
                            image_ref_mask = torch.nn.functional.interpolate(image_ref_mask, (512, 512),
                                                                             mode='nearest')
                            pred_nocs = image_ref
                            scale = gt_nocs.max() / pred_nocs.max()
                            pred_nocs = pred_nocs * scale
                            pred_nocs_mask = torch.cat([pred_nocs, image_ref_mask[0:1]], dim=1)

                        if iter_idx < 100:
                            sillhouette_mask = sillhouette[..., 3][:, None, ...]
                            sillhouette_mask = torch.nn.functional.interpolate(sillhouette_mask, (512, 512),
                                                                               mode='nearest')
                            loss_pose = torch.sum((sillhouette_mask - image_target_mask) ** 2) / 10000.
                        elif iter_idx < 0:
                            loss_pose = torch.sum((pred_nocs - gt_nocs) ** 2) / 10000.
                        else:
                            loss_pose = 0.0


                        preprocessed = []
                        for k_view in range(len(fake_render_reshaped)):
                            fake_render_reshaped_view = pre()(fake_render_reshaped[k_view])
                            preprocessed += [fake_render_reshaped_view[None, ...]]
                        fake_render_reshaped = torch.cat(preprocessed, dim=0)

                        
                        # Global loss branch
                        global_loss = 0.0
                        for k_view in range(len(fake_render_reshaped)):
                            fake_pyramid = image_pyramid(fake_render_reshaped[k_view][None, ...], np.arange(num_levels),
                                                         reverse=True)
                            fake_mask_pyramid = image_pyramid(fake_render_mask[k_view][None, ...],
                                                              np.arange(num_levels), reverse=True, mode='nearest')

                            style_loss, content_loss, pyramid = StyleLoss(fake_pyramid, image_target_clip,
                                                                          fake_mask_pyramid)

                            style_weight = 1.0
                            view_weight = base_view_weight if k_view == 0 else 0.0
                            content_weight = base_content_weight

                            global_loss = global_loss + (
                                        view_weight * style_weight * style_loss + view_weight * content_weight * content_loss)

                            break  # only first view


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
                        num_patches_per_view = 1  # 1 / 4

                        for k_patch in range(num_patches_per_view):
                            target_render_crop_, target_mask_crop_, target_coords, target_nocs_value, target_texture_side = self.crop_texture_cover(
                                image_target_clip,
                                1,
                                coords=None,
                                nocs_map=gt_nocs_mask,
                                use_nocs=True,
                                patch_size=patch_size)

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
                                target_texture_side=target_texture_side,
                                patch_size=patch_size)

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

                        # print('fake patch loss', fake_render_crop.shape, fake_patch_mask.shape, target_render_crop.shape)
                        style_loss, content_loss = StylePatchLoss.forward_multiview(fake_patch_pyramids,
                                                                                    gt_render_crops,
                                                                                    fake_render_patch_mask_pyramids)

                        # style_weight = base_style_weight if k_view != 0 else 0.0
                        style_weight = base_style_weight
                        # content_weight = base_content_weight if k_view == 0 else 0.0
                        content_weight = base_content_weight

                        patch_loss = patch_loss + (style_weight * style_loss + content_weight * content_loss)


                        if iter_idx % 10 == 0:
                            print(f'LOSSES: {global_loss} (global), {gen_loss} (disc.), {loss_pose} (pose)')

                        loss = (w_global_loss * global_loss + w_patch_loss * patch_loss) / 1.0
                        if iter_idx < 200:
                            loss = loss + loss_pose
                        loss.backward(retain_graph=True)

                        all_content_losses += [content_loss.cpu().detach()]
                        all_style_losses += [style_loss.cpu().detach()]
                        all_latent_norms += [torch.sum(latent ** 2).cpu().detach()]

                        if iter_idx in [0, 200, 300, 400, num_iterations - 1]:
                            save_image(fake_render_reshaped.cpu().detach(),
                                       odir_samples / f"fake_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)
                            # save_image(pred_nocs.cpu().detach(),
                            #            odir_samples / f"fakenocs_{batch_idx}_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)

                            if mode == 'shapenet':
                                camera_delta_elevation_array_gt = [np.pi - np.pi / 18] * self.config.views_per_sample
                                camera_delta_angle_array_gt = [0,
                                                               np.pi / 4,
                                                               45 * np.pi / 180,
                                                               70 * np.pi / 180,
                                                               120 * np.pi / 180,
                                                               210 * np.pi / 180]
                                mesh_shapenet_mean = np.zeros(4)
                                mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                                mesh_shapenet_max = 0.32
                                norm_params_gt = (mesh_shapenet_mean, mesh_shapenet_max)
                                fake_render, _, _ = self.render(opt_features, batch,
                                                                face_color=pred_colors,
                                                                use_shape_features=True,
                                                                use_color_bg=False,
                                                                style_latent=z_out,
                                                                norm_params=norm_params_gt,
                                                                camera_delta_angle=camera_delta_angle,
                                                                transform_matrices=transform_matrices,
                                                                camera_delta_angle_array=camera_delta_angle_array_gt,
                                                                camera_delta_elevation_array=camera_delta_elevation_array_gt,
                                                                only_first_view=False,
                                                                scannet_mode=scannet_mode,
                                                                render_nocs=False,
                                                                render_sil=False)
                                fake_render_mask = fake_render[:, 3:4, :, :]
                                fake_render = fake_render[:, :3, :, :]
                                fake_render_reshaped = torch.nn.functional.interpolate(fake_render, (512, 512),
                                                                                       mode='bilinear', align_corners=True)
                                fake_render_reshaped = (fake_render_reshaped + 1.0) / 2.0
                                save_image(fake_render_reshaped.cpu().detach(),
                                           odir_samples / f"fakeasgt_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1),
                                           normalize=True)
                            
                            elif mode == 'scannet':
                                all_scannet_renders = []
                                for meta_data in all_meta_datas:
                                    camera_pose = np.array(meta_data['pose'])
                                    camera_pose = np.linalg.inv(camera_pose)
                                    camera_pose = torch.FloatTensor(camera_pose).cuda()
                                    model_to_scan = np.array(meta_data['model_to_scan'])
                                    model_to_scan = torch.FloatTensor(model_to_scan).cuda()
                                    scannet_transform_matrices = (model_to_scan, camera_pose)
                            
                                    render, _, _ = self.render(opt_features, batch,
                                                             face_color=pred_colors,
                                                             use_shape_features=True,
                                                             use_color_bg=False,
                                                             style_latent=z_out,
                                                             norm_params=norm_params,
                                                             camera_delta_angle=camera_delta_angle,
                                                             transform_matrices=scannet_transform_matrices,
                                                             camera_delta_angle_array=camera_delta_angle_array,
                                                             camera_delta_elevation_array=camera_delta_elevation_array,
                                                             only_first_view=True,
                                                             render_scannet=True,
                                                             )
                                    render = render[0:1, :3, :, :]
                                    render = (render + 1.0) / 2.0
                                    render = torch.nn.functional.interpolate(render, (968, 1296),
                                                                                mode='bilinear', align_corners=True)
                                    all_scannet_renders += [render]
                                save_image(torch.cat(all_scannet_renders, dim=0).cpu().detach(),
                                           odir_samples / f"scannetfake_{batch_idx}_{iter_idx}.jpg", value_range=(0, 1), normalize=True)

                        if iter_idx % 10 == 0:
                            print("run {}:".format(iter_idx))
                            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                                style_loss.item(), content_loss.item()))
                            print('Saving to:', odir_samples)
                            print()
                        if iter_idx % 50 == 0:
                            all_images += [fake_render_reshaped[0:1, ...].cpu().detach()]
                            # all_faces_images += [fake_render_faces_reshaped.cpu().detach()]
                        if iter_idx % 20 == 0:
                            target_crops += [gt_render_crops_save[0].cpu().detach()]
                            fake_crops += [fake_render_crops_save[0].cpu().detach()]
                            target_mask_crops += [gt_render_mask_crops_save[0].cpu().detach()]
                            fake_mask_crops += [fake_render_mask_crops_save[0].cpu().detach()]
                        
                        # Additional outputs in case you need them
                        # if iter_idx == num_iterations - 1:
                        #     save_image(fake_render_reshaped_save.cpu().detach(),
                        #                odir_samples / f"fake_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     save_image(fake_render_mask.cpu().detach(),
                        #                odir_samples / f"fakemask_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     # save_image(fake_render_faces_reshaped.cpu().detach(),
                        #     #            odir_samples / f"fake_faces_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                        #     save_image(pred_nocs.cpu().detach(),
                        #                odir_samples / f"fakenocs_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        #     save_image(gt_nocs.cpu().detach(),
                        #                odir_samples / f"gtnocs_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        #     all_optimized_images += [fake_render_reshaped.cpu().detach()]
                        #     # all_optimized_faces_images += [fake_render_faces_reshaped.cpu().detach()]
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
                            all_content_losses = []
                            all_style_losses = []
                            all_latent_norms = []
                            target_crops = []
                            target_mask_crops = []
                            fake_crops = []
                            fake_mask_crops = []

                        return 1.0 * (style_loss + content_loss)

                    if iter_idx < 200:
                        optimizer_latent.step(closure)
                        optimizer_latent.zero_grad()
                    else:
                        optimizer.step(closure)
                        optimizer.zero_grad()

                    # optimizer.step(closure)
                    # optimizer.zero_grad()

                del norm_params
                del camera_delta_angle
                del transform_matrices

                # save_image(image_target_clip.cpu().detach(),
                #            odir_samples / f"gttarget_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                # save_image(images_target.cpu().detach(),
                #            odir_samples / f"gt_{batch_idx}.jpg", value_range=(0, 1), normalize=True)
                # save_image(image_target_mask.cpu().detach(),
                #            odir_samples / f"gtmask_{batch_idx}.jpg", value_range=(0, 1), normalize=True)

                self.G.synthesis.tuner.load_state_dict(initial_G_parameters)

            # compute_multiview_score(str(odir_samples), self.config.views_per_sample)

    # store Scannet crops (chair category) for NOCs training
    def store_scannet_fornocs(self, savedir=None, shapenetdir=None, scannetcropsdir=None):

        from model.differentiable_renderer import ChairRenderer
        from pytorch3d.io import load_objs_as_meshes, load_obj
        import torchvision.transforms as transforms
        from PIL import Image
        import json

        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()])
        toTensor = transforms.Compose([
            transforms.ToTensor()
        ])

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/chairs_scannet_fornocs'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)

        if shapenetdir is None:
            SHAPENETDIR = '/canis/ShapeNet/ShapeNetCore.v2/03001627'
        else:
            SHAPENETDIR = shapenetdir

        if scannetcropsdir is None:
            SCANNET_CROPS_DIR = '/cluster/daidalos/abokhovkin/scannet-texturing/data/ScanNet_chair_masked'
        else:
            SCANNET_CROPS_DIR = scannetcropsdir
        sceneid_to_imgnames = defaultdict(list)
        all_scannet_imagenames = defaultdict(list)
        for scene_id in os.listdir(SCANNET_CROPS_DIR):
            # in case you want to specify a particular scene
            # if scene_id != 'scene0470_00':
            #     continue
            if len(os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id))) > 0:
                img_names = [x for x in os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id)) if 'scancrop' in x]
                sceneid_to_imgnames[scene_id] = img_names
                all_scannet_imagenames[scene_id] = defaultdict(list)
                for img_name in img_names:
                    all_scannet_imagenames[scene_id][img_name.split('_')[1]] += [img_name]
        all_scannet_imagenames_tuples = []
        for scene_id in all_scannet_imagenames:
            for img_id in all_scannet_imagenames[scene_id]:
                all_scannet_imagenames_tuples += [(scene_id, all_scannet_imagenames[scene_id][img_id])]

        for scene_imagenames in all_scannet_imagenames_tuples:
            scene_id, instance_img_names = scene_imagenames
            random_img_names = np.random.choice(instance_img_names, min(len(instance_img_names), 20), replace=False)
            for k, img_name in enumerate(random_img_names):
                image_target = Image.open(os.path.join(SCANNET_CROPS_DIR, scene_id, img_name))
                image_target = preprocess(image_target).unsqueeze(0)
                img_id = img_name.split('scancrop')[0][:-1]
                image_scan = Image.open(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{img_id}.jpg'))
                image_scan = toTensor(image_scan).unsqueeze(0)
                meta_data = json.loads(
                    Path(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{img_id}_meta.json')).read_text())

                obj_id = meta_data['model']['id_cad']
                intrinsics = meta_data['intrinsics']
                intrinsics = torch.FloatTensor(np.array(intrinsics))[None, ...]
                camera_pose = np.array(meta_data['pose'])
                camera_pose = np.linalg.inv(camera_pose)
                camera_pose = torch.FloatTensor(camera_pose).cuda()
                model_to_scan = np.array(meta_data['model_to_scan'])
                model_to_scan = torch.FloatTensor(model_to_scan).cuda()
                transform_matrices = (model_to_scan, camera_pose)

                mesh_shapenet = load_objs_as_meshes(
                    [f'{SHAPENETDIR}/{obj_id}/models/model_normalized.obj'],
                    load_textures=True,
                    create_texture_atlas=True,
                    texture_atlas_size=4,
                    device='cuda'
                )
                if len(mesh_shapenet._verts_list[0]) > 20000:
                    continue

                renderer = ChairRenderer(img_size=(968, 1296))
                images = renderer.render(mesh_shapenet, transform_matrices, intrinsics)
                images = images[..., :3]
                images = torch.permute(images, (0, 3, 1, 2))


                save_image(images.cpu().detach(),
                           os.path.join(SAVEDIR, f"render_{scene_id}_{img_id}.jpg"),
                           value_range=(0, 1), normalize=True)
                save_image(image_target.cpu().detach(),
                           os.path.join(SAVEDIR, f"crop_{scene_id}_{img_id}.jpg"),
                           value_range=(0, 1), normalize=True)
                save_image(image_scan.cpu().detach(),
                           os.path.join(SAVEDIR, f"scan_{scene_id}_{img_id}.jpg"),
                           value_range=(0, 1), normalize=True)

    # store ShapeNet renders (chair category) for NOCs training
    def store_shapenet_fornocs(self, savedir=None, shapenetdir=None):

        from model.differentiable_renderer import ChairRenderer
        from pytorch3d.io import load_objs_as_meshes, load_obj

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/chairs_shapenet_fornocs'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)

        if shapenetdir is None:
            SHAPENETDIR = '/canis/ShapeNet/ShapeNetCore.v2/03001627'
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
            if len(mesh_shapenet._verts_list[0]) > 60000:
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
            
    # store ShapeNet renders (chair category)
    def store_shapenet_textures(self, savedir=None, shapenetdir=None, scannetcropsdir=None):

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/chairs'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)

        if shapenetdir is None:
            SHAPENETDIR = '/canis/ShapeNet/ShapeNetCore.v2/03001627'
        else:
            SHAPENETDIR = shapenetdir

        if scannetcropsdir is None:
            SCANNET_CROPS_DIR = '/cluster/daidalos/abokhovkin/scannet-texturing/data/ScanNet_chair_masked'
        else:
            SCANNET_CROPS_DIR = scannetcropsdir
        sceneid_to_imgnames = defaultdict(list)
        for scene_id in os.listdir(SCANNET_CROPS_DIR):
            if len(os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id))) > 0:
                img_names = [x for x in os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id)) if 'scancrop' in x]
                sceneid_to_imgnames[scene_id] = img_names
        all_scannet_imagenames = []
        for scene_id in sceneid_to_imgnames:
            for img_name in sceneid_to_imgnames[scene_id]:
                all_scannet_imagenames += [(scene_id, img_name)]
                break

        camera_azimuths = [0.0, 1/6 * np.pi, 2/6 * np.pi, 3/6 * np.pi, 4/6 * np.pi, 5/6 * np.pi,
                           6/6 * np.pi, 7/6 * np.pi, 8/6 * np.pi, 9/6 * np.pi, 10/6 * np.pi, 11/6 * np.pi]
        camera_elevations = [10/12 * np.pi, 11/12 * np.pi, np.pi, 13/12 * np.pi, 14/12 * np.pi]
        camera_parameters = []
        for x in camera_azimuths:
            for y in camera_elevations:
                camera_parameters += [(x, y)]

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)
            for batch_idx, batch in enumerate(self.train_dataloader()):

                randomly_sampled_indices = np.random.choice(len(camera_parameters), 8, replace=False) # 6
                randomly_sampled_parameters = np.array(camera_parameters)[randomly_sampled_indices]

                if batch_idx > 32767:
                    break

                obj_id = batch['shapenet_id'][0]

                # ShapeNet
                mesh_shapenet = trimesh.load(f'{SHAPENETDIR}/{obj_id}/models/model_normalized.obj', force='mesh')
                if len(mesh_shapenet.vertices) > 20000:
                    continue

                metadata = {'obj_id': obj_id}

                mesh_shapenet_mean = np.zeros(4)
                mesh_shapenet_mean[:3] = (mesh_shapenet.vertices.max(axis=0) + mesh_shapenet.vertices.min(axis=0)) / 2
                mesh_shapenet_mean = torch.FloatTensor(mesh_shapenet_mean).cuda()
                mesh_shapenet_max = float(mesh_shapenet.vertices[:, 1].max())
                norm_params = (mesh_shapenet_mean, mesh_shapenet_max)

                metadata['shapenet_mean'] = list(mesh_shapenet_mean.cpu().numpy().astype('float'))
                metadata['shapenet_max'] = mesh_shapenet_max
                with open(os.path.join(SAVEDIR, f'obj_id_{batch_idx}.json'), 'w') as fout:
                    json.dump(metadata, fout)

                try:
                    images_target = []
                    for k, camera_azimuth_elevation in enumerate(randomly_sampled_parameters):
                        image_target = self.render_shapenet(obj_id,
                                                            camera_delta_angle=camera_azimuth_elevation[0],
                                                            camera_elevation=camera_azimuth_elevation[1])
                        images_target += [image_target]
                    images_target = torch.cat(images_target, dim=0)
                except:
                    continue

                for i in range(len(images_target)):
                    save_image(images_target.cpu().detach()[i][None, ...],
                               os.path.join(SAVEDIR, f"clip_{batch_idx}_{i}.jpg"),
                               value_range=(0, 1), normalize=True)
                    params = {'azimuth': float(randomly_sampled_parameters[i][0]),
                              'elevation': float(randomly_sampled_parameters[i][1]),
                              'camera_id': int(randomly_sampled_indices[i])}
                    with open(os.path.join(SAVEDIR, f"clip_{batch_idx}_{i}.json"), 'w') as fout:
                        json.dump(params, fout)

    # store ScanNet GT images
    def store_scannet_textures(self, savedir=None, scannetcropsdir=None):

        import torchvision.transforms as transforms
        from PIL import Image
        from scipy.spatial.transform import Rotation

        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()])

        self.G = self.G.to('cuda')
        self.R = self.R.to('cuda')

        if savedir is None:
            SAVEDIR = '/cluster/valinor/abokhovkin/scannet-texturing/data/ShapeNetViews/chairs_scannet'
        else:
            SAVEDIR = savedir
        os.makedirs(SAVEDIR, exist_ok=True)
        SAVEDIR = Path(SAVEDIR)

        if scannetcropsdir is None:
            SCANNET_CROPS_DIR = '/cluster/daidalos/abokhovkin/scannet-texturing/data/ScanNet_chair_masked'
        else:
            SCANNET_CROPS_DIR = scannetcropsdir
        sceneid_to_imgnames = defaultdict(list)
        all_scannet_imagenames = defaultdict(list)
        for scene_id in os.listdir(SCANNET_CROPS_DIR):
            if len(os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id))) > 0:
                img_names = [x for x in os.listdir(os.path.join(SCANNET_CROPS_DIR, scene_id)) if 'scancrop' in x]
                sceneid_to_imgnames[scene_id] = img_names
                all_scannet_imagenames[scene_id] = defaultdict(list)
                for img_name in img_names:
                    all_scannet_imagenames[scene_id][img_name.split('_')[1]] += [img_name]
        all_scannet_imagenames_tuples = []
        for scene_id in all_scannet_imagenames:
            for img_id in all_scannet_imagenames[scene_id]:
                all_scannet_imagenames_tuples += [(scene_id, all_scannet_imagenames[scene_id][img_id])]

        camera_azimuths = [0.0, 1 / 6 * np.pi, 2 / 6 * np.pi, 3 / 6 * np.pi, 4 / 6 * np.pi, 5 / 6 * np.pi,
                           6 / 6 * np.pi, 7 / 6 * np.pi, 8 / 6 * np.pi, 9 / 6 * np.pi, 10 / 6 * np.pi, 11 / 6 * np.pi]
        camera_azimuths = np.array(camera_azimuths)
        camera_elevations = [10 / 12 * np.pi, 11 / 12 * np.pi, np.pi, 13 / 12 * np.pi, 14 / 12 * np.pi]
        camera_elevations = np.array(camera_elevations)

        with Timer("optimize latent"):
            print('batch size', self.config.batch_size)
            for batch_idx, batch in enumerate(self.train_dataloader()):
                break

            set_real_sample_resolution(batch)
            obj_id = batch['shapenet_id'][0]

            for i in range(30000):

                scene_id, instance_img_names = all_scannet_imagenames_tuples[i]
                random_img_names = np.random.choice(instance_img_names, min(len(instance_img_names), 20), replace=False)
                for k, img_name in enumerate(random_img_names):
                    image_target = Image.open(os.path.join(SCANNET_CROPS_DIR, scene_id, img_name))
                    image_target = preprocess(image_target).unsqueeze(0)
                    img_id = img_name.split('scancrop')[0][:-1]
                    meta_data = json.loads(Path(os.path.join(SCANNET_CROPS_DIR, scene_id, f'{img_id}_meta.json')).read_text())

                    obj_id = meta_data['model']['id_cad']
                    camera_pose = np.array(meta_data['pose'])
                    camera_pose[:3, 3] = 0
                    camera_pose = np.linalg.inv(camera_pose)
                    camera_pose = torch.FloatTensor(camera_pose).cuda()
                    model_to_scan = np.array(meta_data['model_to_scan'])
                    model_to_scan[:3, 3] = 0
                    model_to_scan = torch.FloatTensor(model_to_scan).cuda()

                    transform_matrix = camera_pose @ model_to_scan
                    transform_matrix = np.array(transform_matrix[:3, :3].cpu())
                    rot = Rotation.from_matrix(transform_matrix)
                    rot_euler = rot.as_euler('xyz', degrees=True)
                    print(rot_euler)
                    if rot_euler[0] >= -90:
                        rot_azimuth = (-rot_euler[1]) % 360
                        rot_azimuth_rad = np.pi * rot_azimuth / 180
                    elif rot_euler[0] < -90:
                        rot_azimuth = (180 + rot_euler[1]) % 360
                        rot_azimuth_rad = np.pi * rot_azimuth / 180
                    rot_elevation_rad = 10. / 12. * np.pi

                    save_image(image_target.cpu(),
                               SAVEDIR / f"clip_{i}_{k}.jpg", value_range=(0, 1), normalize=True)

                    params = {'azimuth': float(rot_azimuth),
                              'azimuth_rad': float(rot_azimuth_rad),
                              'azimuth_id': int(np.argmin((camera_azimuths - rot_azimuth_rad) ** 2)),
                              'elevation': float(rot_elevation_rad),
                              'elevation_id': int(np.argmin((camera_elevations - rot_elevation_rad) ** 2))
                              }
                    with open(os.path.join(SAVEDIR, f"clip_{i}_{k}.json"), 'w') as fout:
                        json.dump(params, fout)


# set GT image resolution
def set_real_sample_resolution(batch):
    batch['real2'] = torch.nn.functional.interpolate(batch['real'], size=(512, 512), mode='bilinear', align_corners=False)
    batch['mask2'] = torch.nn.functional.interpolate(batch['mask'], size=(512, 512), mode='nearest')
    batch['real'] = torch.nn.functional.interpolate(batch['real'], size=(256, 256), mode='bilinear', align_corners=False)
    batch['mask'] = torch.nn.functional.interpolate(batch['mask'], size=(256, 256), mode='nearest')

# wrapper for step to clip gradients
def step(opt, module):
    for param in module.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
    opt.step()


@hydra.main(config_path='../config', config_name='stylegan2')
def main(config):
    trainer = create_trainer("StyleGAN23D", config)
    model = StyleGAN2Trainer(config)

    # config.resume = '...'
    # model = StyleGAN2Trainer(config).load_from_checkpoint(config.resume)
    trainer.fit(model)


if __name__ == '__main__':
    main()
