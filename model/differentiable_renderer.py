import torch
from torch import nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
from torchvision.ops import masks_to_boxes
import numpy as np
from model import FullyConnectedLayer, EqualizedConv2d

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase, HardFlatShader
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.io.utils import _make_tensor
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    SoftGouraudShader
)
from pytorch3d.datasets import ShapeNetCore
from util.embedder import get_embedder_nerf
from scipy.spatial.transform import Rotation


class NOCsShader(ShaderBase):
    def __init__(
            self,
            device="cpu",
            cameras=None,
            lights=None,
            materials=None,
            blend_params=None,
    ) -> None:
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        orig_vertices = kwargs['orig_vertices']

        image_shape = fragments.pix_to_face.shape[:3]
        pixel_vertex_map = torch.zeros((*image_shape, 3)).long().to(fragments.pix_to_face.device)
        feature_vertex_map = torch.zeros((*image_shape, 3, 3)).float().to(fragments.pix_to_face.device)
        meshes = meshes.to(fragments.pix_to_face.device)

        mask_foreground = fragments.pix_to_face[..., 0] > 0 # [N, img_size, img_size]
        barycentric = fragments.bary_coords[..., 0, :] # [N, img_size, img_size, 3]
        pixel_vertex_map[mask_foreground] = meshes._faces_list[0][fragments.pix_to_face[..., 0][mask_foreground]] # [N, img_size, img_size, 3]

        feature_vertex_map[mask_foreground] = orig_vertices[pixel_vertex_map[mask_foreground]]  # [N, img_size, img_size, 3, num_feat=3]

        feature_face_map = (feature_vertex_map * barycentric[..., None]).mean(dim=3) # [N, img_size, img_size, num_feat=3]
        feature_face_map = feature_face_map / barycentric.sum(dim=3)[..., None]

        scale = orig_vertices.max() / feature_face_map.max()
        feature_face_map = feature_face_map * scale

        mask_map = torch.zeros((*image_shape, 1)).float().to(fragments.pix_to_face.device)
        mask_map[mask_foreground] = 1

        feature_face_map = torch.cat([feature_face_map, mask_map], dim=3)

        return feature_face_map


class HardMLPShader(ShaderBase):
    def __init__(
            self,
            device="cpu",
            cameras=None,
            lights=None,
            materials=None,
            blend_params=None,
            mlp=None,
            feature_dim=256,
            residual=False,
            tanh=False,
            use_act=True,
            mode='car'
    ) -> None:
        super().__init__(device, cameras, lights, materials, blend_params)
        self.mlp = mlp
        self.feature_dim = feature_dim
        self.residual = residual
        self.tanh = tanh
        self.use_act = use_act
        self.embedder, self.embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
        self.split_size = 6000
        self.mode = mode

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        feature = kwargs['feature']
        color_bg = kwargs['color_bg']
        shape_features = kwargs['shape_features']
        aux_latent = kwargs['aux_latent']
        print_stats = kwargs['print_stats']
        style_latent = kwargs['style_latent']
        image_shape = fragments.pix_to_face.shape[:3]
        if color_bg is None:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
        else:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
            buf_image[:, :, :, 0] = color_bg[:, 0, 0, 0]
            buf_image[:, :, :, 1] = color_bg[:, 1, 0, 0]
            buf_image[:, :, :, 2] = color_bg[:, 2, 0, 0]
        pixel_vertex_map = torch.zeros((*image_shape, 3)).long().to(fragments.pix_to_face.device)
        feature_vertex_map = torch.zeros((*image_shape, 3, self.feature_dim)).float().to(fragments.pix_to_face.device)
        meshes = meshes.to(fragments.pix_to_face.device)
        feature = feature.to(fragments.pix_to_face.device)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask_foreground = fragments.pix_to_face[..., 0] > 0 # [N, img_size, img_size]
        barycentric = fragments.bary_coords[..., 0, :] # [N, img_size, img_size, 3]
        pixel_vertex_map[mask_foreground] = meshes._faces_list[0][fragments.pix_to_face[..., 0][mask_foreground]] # [N, img_size, img_size, 3]
        feature_vertex_map[mask_foreground] = feature[pixel_vertex_map[mask_foreground]] # [N, img_size, img_size, 3, num_feat=256]
        feature_face_map = (feature_vertex_map * barycentric[..., None]).mean(dim=3) # [N, img_size, img_size, num_feat=256]
        feature_face_map_flattened = feature_face_map[mask_foreground]
        barycentric_flattened = barycentric[mask_foreground]

        if shape_features is not None:
            if self.mode == 'chair':
                target_feat_size = 64
            elif self.mode == 'car':
                target_feat_size = 128

            shape_features_vertex_map = torch.zeros((*image_shape, 3, target_feat_size)).float().to(fragments.pix_to_face.device) # 64 / 128
            shape_features_vertex_map[mask_foreground] = shape_features[pixel_vertex_map[mask_foreground]]
            shape_features_vertex_map = (shape_features_vertex_map * barycentric[..., None]).mean(dim=3)
            shape_features_vertex_map_flattened = shape_features_vertex_map[mask_foreground]
            shape_features_vertex_map_flattened = torch.split(shape_features_vertex_map_flattened, self.split_size)

        mask_map = torch.zeros((*image_shape, 1)).float().to(fragments.pix_to_face.device)
        mask_map[mask_foreground] = 1

        feature_face_map_flattened = torch.split(feature_face_map_flattened, self.split_size)
        barycentric_flattened = torch.split(barycentric_flattened, self.split_size)

        all_colors = []
        for i in range(len(feature_face_map_flattened)):

            aux_latent_repeat = style_latent.repeat(len(feature_face_map_flattened[i]), 1)
            x = self.mlp(feature_face_map_flattened[i],
                         shape_features_vertex_map_flattened[i],
                         self.embedder(barycentric_flattened[i]),
                         aux_latent_repeat)

            if self.use_act:
                if not self.tanh:
                    colors = x - 1
                else:
                    colors = x
            else:
                colors = x
            all_colors += [colors]
        all_colors = torch.cat(all_colors, dim=0)

        buf_image[mask_foreground] = all_colors
        buf_image = torch.cat([buf_image, mask_map], dim=3)

        return buf_image


class HardMLPConvShader(ShaderBase):
    def __init__(
            self,
            device="cpu",
            cameras=None,
            lights=None,
            materials=None,
            blend_params=None,
            mlp_prologue=None,
            conv_module=None,
            mlp_epilogue=None,
            feature_dim=256,
            residual=False,
            tanh=False,
            use_act=True
    ) -> None:
        super().__init__(device, cameras, lights, materials, blend_params)
        self.mlp_prologue = mlp_prologue
        self.conv_module = conv_module
        self.mlp_epilogue = mlp_epilogue
        self.feature_dim = feature_dim
        self.residual = residual
        self.tanh = tanh
        self.use_act = use_act
        self.embedder, self.embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
        self.split_size = 16384

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        feature = kwargs['feature']
        color_bg = kwargs['color_bg']
        shape_features = kwargs['shape_features']
        aux_latent = kwargs['aux_latent']
        print_stats = kwargs['print_stats']
        style_latent = kwargs['style_latent']
        image_shape = fragments.pix_to_face.shape[:3]
        if color_bg is None:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
        else:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
            buf_image[:, :, :, 0] = color_bg[:, 0, 0, 0]
            buf_image[:, :, :, 1] = color_bg[:, 1, 0, 0]
            buf_image[:, :, :, 2] = color_bg[:, 2, 0, 0]
        pixel_vertex_map = torch.zeros((*image_shape, 3)).long().to(fragments.pix_to_face.device)
        feature_vertex_map = torch.zeros((*image_shape, 3, self.feature_dim)).float().to(fragments.pix_to_face.device)
        meshes = meshes.to(fragments.pix_to_face.device)
        feature = feature.to(fragments.pix_to_face.device)

        feat_image = torch.zeros((*image_shape, 256)).float().to(fragments.pix_to_face.device)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask_foreground = fragments.pix_to_face[..., 0] > 0 # [N, img_size, img_size]
        barycentric = fragments.bary_coords[..., 0, :] # [N, img_size, img_size, 3]
        pixel_vertex_map[mask_foreground] = meshes._faces_list[0][fragments.pix_to_face[..., 0][mask_foreground]] # [N, img_size, img_size, 3]
        feature_vertex_map[mask_foreground] = feature[pixel_vertex_map[mask_foreground]] # [N, img_size, img_size, 3, num_feat=256]
        feature_face_map = (feature_vertex_map * barycentric[..., None]).mean(dim=3) # [N, img_size, img_size, num_feat=256]
        feature_face_map_flattened = feature_face_map[mask_foreground]
        barycentric_flattened = barycentric[mask_foreground]

        if shape_features is not None:
            shape_features_vertex_map = torch.zeros((*image_shape, 3, 64)).float().to(fragments.pix_to_face.device) # 64 / 128
            shape_features_vertex_map[mask_foreground] = shape_features[pixel_vertex_map[mask_foreground]]
            shape_features_vertex_map = (shape_features_vertex_map * barycentric[..., None]).mean(dim=3)
            shape_features_vertex_map_flattened = shape_features_vertex_map[mask_foreground]
            shape_features_vertex_map_flattened = torch.split(shape_features_vertex_map_flattened, self.split_size)

        mask_map = torch.zeros((*image_shape, 1)).float().to(fragments.pix_to_face.device)
        mask_map[mask_foreground] = 1

        feature_face_map_flattened = torch.split(feature_face_map_flattened, self.split_size)
        barycentric_flattened = torch.split(barycentric_flattened, self.split_size)

        all_prologue_feat = []
        for i in range(len(feature_face_map_flattened)):

            aux_latent_repeat = style_latent.repeat(len(feature_face_map_flattened[i]), 1)
            x = self.mlp_prologue(feature_face_map_flattened[i],
                                  shape_features_vertex_map_flattened[i],
                                  self.embedder(barycentric_flattened[i]),
                                  aux_latent_repeat)

            all_prologue_feat += [x]
        all_prologue_feat = torch.cat(all_prologue_feat, dim=0)
        feat_image[mask_foreground] = all_prologue_feat
        feat_image = torch.permute(feat_image, (0, 3, 1, 2))
        feat_image = self.conv_module(feat_image)
        feat_image = torch.permute(feat_image, (0, 2, 3, 1))
        feat_image_flattened = feat_image[mask_foreground]
        feat_image_flattened = torch.split(feat_image_flattened, self.split_size)
        all_epilogue_feat = []
        for i in range(len(feat_image_flattened)):
            x = self.mlp_epilogue(feat_image_flattened[i])
            all_epilogue_feat += [x]
        all_epilogue_feat = torch.cat(all_epilogue_feat)
        all_epilogue_feat = torch.clamp(all_epilogue_feat, -1, 1)

        buf_image[mask_foreground] = all_epilogue_feat
        buf_image = torch.cat([buf_image, mask_map], dim=3)

        return buf_image


class HardColorMLPShader(ShaderBase):
    def __init__(
            self,
            device="cpu",
            cameras=None,
            lights=None,
            materials=None,
            blend_params=None,
            mlp=None,
            feature_dim=256,
            tanh=False,
            use_act=True,
            residual=False
    ) -> None:
        super().__init__(device, cameras, lights, materials, blend_params)
        self.mlp = mlp
        self.feature_dim = feature_dim
        self.residual = residual
        self.tanh = tanh
        self.use_act = use_act
        self.embedder, self.embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
        self.split_size = 16384

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        feature = kwargs['feature']
        color = kwargs['color'][:, :3]
        color_bg = kwargs['color_bg']
        shape_features = kwargs['shape_features']
        aux_latent = kwargs['aux_latent']
        print_stats = kwargs['print_stats']
        style_latent = kwargs['style_latent']
        image_shape = fragments.pix_to_face.shape[:3]
        if color_bg is None:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
        else:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
            buf_image[:, :, :, 0] = color_bg[:, 0, 0, 0]
            buf_image[:, :, :, 1] = color_bg[:, 1, 0, 0]
            buf_image[:, :, :, 2] = color_bg[:, 2, 0, 0]
        pixel_vertex_map = torch.zeros((*image_shape, 3)).long().to(fragments.pix_to_face.device)
        feature_vertex_map = torch.zeros((*image_shape, 3, self.feature_dim)).float().to(fragments.pix_to_face.device)
        meshes = meshes.to(fragments.pix_to_face.device)
        feature = feature.to(fragments.pix_to_face.device)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask_foreground = fragments.pix_to_face[..., 0] > 0 # [N, img_size, img_size]
        barycentric = fragments.bary_coords[..., 0, :] # [N, img_size, img_size, 3]
        pixel_vertex_map[mask_foreground] = meshes._faces_list[0][fragments.pix_to_face[..., 0][mask_foreground]] # [N, img_size, img_size, 3]
        feature_vertex_map[mask_foreground] = feature[pixel_vertex_map[mask_foreground]] # [N, img_size, img_size, 3, num_feat=256]
        feature_face_map = (feature_vertex_map * barycentric[..., None]).mean(dim=3) # [N, img_size, img_size, num_feat=256]
        feature_face_map_flattened = feature_face_map[mask_foreground]
        barycentric_flattened = barycentric[mask_foreground]

        if shape_features is not None:
            shape_features_vertex_map = torch.zeros((*image_shape, 3, 64)).float().to(fragments.pix_to_face.device) # 64 / 128
            shape_features_vertex_map[mask_foreground] = shape_features[pixel_vertex_map[mask_foreground]]
            shape_features_vertex_map = (shape_features_vertex_map * barycentric[..., None]).mean(dim=3)
            shape_features_vertex_map_flattened = shape_features_vertex_map[mask_foreground]
            shape_features_vertex_map_flattened = torch.split(shape_features_vertex_map_flattened, self.split_size)

        color_vertex_map = torch.zeros((*image_shape, 3, 3)).float().to(fragments.pix_to_face.device)
        color_vertex_map[mask_foreground] = color[pixel_vertex_map[mask_foreground]]
        color_face_map = (color_vertex_map * barycentric[..., None]).mean(dim=3)
        color_face_map_flattened = color_face_map[mask_foreground]

        mask_map = torch.zeros((*image_shape, 1)).float().to(fragments.pix_to_face.device)
        mask_map[mask_foreground] = 1

        feature_face_map_flattened = torch.split(feature_face_map_flattened, self.split_size)
        barycentric_flattened = torch.split(barycentric_flattened, self.split_size)
        color_face_map_flattened = torch.split(color_face_map_flattened, self.split_size)
        all_colors = []
        for i in range(len(feature_face_map_flattened)):
            aux_latent_repeat = aux_latent.repeat(len(feature_face_map_flattened[i]), 1)
            x = torch.cat([feature_face_map_flattened[i], shape_features_vertex_map_flattened[i], aux_latent_repeat], dim=1)  # shape features implicit + learnable vector

            x = self.mlp(x)

            if self.use_act:
                if not self.tanh:
                    colors = x - 1
                else:
                    colors = x
            else:
                colors = x
            colors = colors + color_face_map_flattened[i]
            all_colors += [colors]
        all_colors = torch.cat(all_colors, dim=0)
        if print_stats:
            try:
                print('Colors stats (1)', all_colors[:, 0].min(), all_colors[:, 0].mean(), all_colors[:, 0].max(), all_colors[:, 0].std())
            except:
                pass
        all_colors = torch.clamp(all_colors, -1, 1)
        if print_stats:
            try:
                print('Colors stats (2)', all_colors[:, 0].min(), all_colors[:, 0].mean(), all_colors[:, 0].max(), all_colors[:, 0].std())
            except:
                pass
        buf_image[mask_foreground] = all_colors
        buf_image = torch.cat([buf_image, mask_map], dim=3)

        return buf_image


class ColorShader(ShaderBase):
    def __init__(
            self,
            device="cpu",
            cameras=None,
            lights=None,
            materials=None,
            blend_params=None
    ) -> None:
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        color = kwargs['color'][:, :3]
        color_bg = kwargs['color_bg']
        image_shape = fragments.pix_to_face.shape[:3]
        if color_bg is None:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
        else:
            buf_image = torch.ones((*image_shape, 3)).float().to(fragments.pix_to_face.device)
            buf_image[:, :, :, 0] = color_bg[:, 0, 0, 0]
            buf_image[:, :, :, 1] = color_bg[:, 1, 0, 0]
            buf_image[:, :, :, 2] = color_bg[:, 2, 0, 0]
        pixel_vertex_map = torch.zeros((*image_shape, 3)).long().to(fragments.pix_to_face.device)
        meshes = meshes.to(fragments.pix_to_face.device)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask_foreground = fragments.pix_to_face[..., 0] > 0 # [N, img_size, img_size]
        barycentric = fragments.bary_coords[..., 0, :] # [N, img_size, img_size, 3]
        pixel_vertex_map[mask_foreground] = meshes._faces_list[0][fragments.pix_to_face[..., 0][mask_foreground]] # [N, img_size, img_size, 3]

        color_vertex_map = torch.zeros((*image_shape, 3, 3)).float().to(fragments.pix_to_face.device)
        color_vertex_map[mask_foreground] = color[pixel_vertex_map[mask_foreground]]
        color_face_map = (color_vertex_map * barycentric[..., None]).mean(dim=3)
        color_face_map_flattened = color_face_map[mask_foreground]

        mask_map = torch.zeros((*image_shape, 1)).float().to(fragments.pix_to_face.device)
        mask_map[mask_foreground] = 1

        all_colors = color_face_map_flattened
        buf_image[mask_foreground] = all_colors
        buf_image = torch.cat([buf_image, mask_map], dim=3)

        return buf_image



def transform_pos(pos, projection_matrix, world_to_cam_matrix):
    # (x,y,z) -> (x,y,z,1)
    t_mtx = torch.matmul(projection_matrix, world_to_cam_matrix)
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    return torch.matmul(posw, t_mtx.t())


# transform vertices to the mvp camera view
def transform_pos_mvp(pos, mvp):
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    return torch.bmm(posw.unsqueeze(0).expand(mvp.shape[0], -1, -1), mvp.permute((0, 2, 1))).reshape((-1, 4))


# reshape vertices without transformation
def transform_pos_mvp_idle(pos, mvp):
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    return posw.unsqueeze(0).expand(mvp.shape[0], -1, -1).reshape((-1, 4))


def render(glctx, pos_clip, pos_idx, vtx_col, col_idx, resolution, ranges, _colorspace, background=None):
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution], ranges=ranges)
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    mask = color[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)

    return color[:, :, :, :-1]


def render_in_bounds(glctx, pos_clip, pos_idx, vtx_col, col_idx, resolution, ranges, color_space, background=None):
    render_resolution = int(resolution * 1.2)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[render_resolution, render_resolution], ranges=ranges)
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    mask = color[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)

    color[..., -1:] = mask.float()
    color_crops = []
    boxes = masks_to_boxes(torch.logical_not(mask.squeeze(-1)))
    for img_idx in range(color.shape[0]):
        x1, y1, x2, y2 = [int(val) for val in boxes[img_idx, :].tolist()]
        color_crop = color[img_idx, y1: y2, x1: x2, :].permute((2, 0, 1))
        pad = [[0, 0], [0, 0]]
        if y2 - y1 > x2 - x1:
            total_pad = (y2 - y1) - (x2 - x1)
            pad[0][0] = total_pad // 2
            pad[0][1] = total_pad - pad[0][0]
            pad[1][0], pad[1][1] = 0, 0
            additional_pad = int((y2 - y1) * 0.1)
        else:
            total_pad = (x2 - x1) - (y2 - y1)
            pad[0][0], pad[0][1] = 0, 0
            pad[1][0] = total_pad // 2
            pad[1][1] = total_pad - pad[1][0]
            additional_pad = int((x2 - x1) * 0.1)
        for i in range(4):
            pad[i // 2][i % 2] += additional_pad

        padded = torch.ones((color_crop.shape[0], color_crop.shape[1] + pad[1][0] + pad[1][1], color_crop.shape[2] + pad[0][0] + pad[0][1]), device=color_crop.device)
        padded[:3, :, :] = padded[:3, :, :] * one_tensor[img_idx, :3, :, :]
        padded[:, pad[1][0]: padded.shape[1] - pad[1][1], pad[0][0]: padded.shape[2] - pad[0][1]] = color_crop
        color_crop = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).permute((0, 2, 3, 1))
        color_crops.append(color_crop)
    return torch.cat(color_crops, dim=0)


def intrinsic_to_projection(intrinsic_matrix):
    near, far = 0.1, 50.
    a, b = -(far + near) / (far - near), -2 * far * near / (far - near)
    projection_matrix = torch.tensor([
        intrinsic_matrix[0][0] / intrinsic_matrix[0][2],    0,                                                0, 0,
        0,                                                  -intrinsic_matrix[1][1] / intrinsic_matrix[1][2], 0, 0,
        0,                                                  0,                                                a, b,
        0,                                                  0,                                               -1, 0
    ]).float().reshape((4, 4))
    return projection_matrix


class DifferentiableRenderer(nn.Module):

    def __init__(self, resolution, mode='standard', color_space='rgb', num_channels=3):
        super().__init__()
        self.glctx = dr.RasterizeGLContext()
        self.resolution = resolution
        self.render_func = render
        self.color_space = color_space
        self.num_channels = num_channels
        if mode == 'bounds':
            self.render_func = render_in_bounds

    def render(self, vertex_positions, triface_indices, vertex_colors, ranges=None, background=None, resolution=None, return_mask=False, levels=False):
        if ranges is None:
            ranges = torch.tensor([[0, triface_indices.shape[0]]]).int()
        if resolution is None:
            resolution = self.resolution
        color = self.render_func(self.glctx, vertex_positions, triface_indices, vertex_colors, triface_indices, resolution, ranges, self.color_space, background)
        num_channels = 4 if return_mask else self.num_channels
        return color[:, :, :, :num_channels]

    def get_visible_triangles(self, vertex_positions, triface_indices, ranges=None, resolution=None):
        if ranges is None:
            ranges = torch.tensor([[0, triface_indices.shape[0]]]).int()
        if resolution is None:
            resolution = self.resolution
        rast, _ = dr.rasterize(self.glctx, vertex_positions, triface_indices, resolution=[resolution, resolution], ranges=ranges)
        return rast[..., 3]


class DifferentiableP3DRenderer(nn.Module):

    def __init__(self, resolution, num_channels=3):
        super().__init__()
        self.resolution = resolution
        self.render_resolution = int(1.1 * self.resolution)
        self.num_channels = num_channels

        R, T = look_at_view_transform(2.4, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 1.55),), degrees=False)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)

        self.shader = ColorShader(cameras=self.cameras)
        raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            blur_radius=1e-5,
            faces_per_pixel=1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=self.shader
        )

    def render(self, vertices, faces, ranges, vertex_colors=None, color_bg=None, resolution=None,
               norm_params=None, camera_delta_angle=None, transform_matrices=None, camera_delta_angle_array=None,
               camera_delta_elevation_array=None, render_scannet=False, only_first_view=False):

        if camera_delta_elevation_array is None:
            camera_delta_elevation_array = [np.pi] * 8

        device = vertices.device
        if device != self.cameras.device:
            print('Incorrect device!')
            R, T = look_at_view_transform(2.4, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 1.55),), degrees=False)
            self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=5.0, fov=50.0)

            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=1e-5,
                faces_per_pixel=1,
            )
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings
                ),
                shader=self.shader
            )

        if resolution is not None:
            if self.resolution != resolution:
                self.resolution = resolution
                self.render_resolution = int(1.1 * self.resolution)
                R, T = look_at_view_transform(2.4, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 1.55),), degrees=False)
                self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=5.0, fov=50.0)

                raster_settings = RasterizationSettings(
                    image_size=self.render_resolution,
                    blur_radius=1e-5,
                    faces_per_pixel=1,
                )
                self.renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=self.cameras,
                        raster_settings=raster_settings
                    ),
                    shader=self.shader
                )

        if norm_params is not None:
            rot_90 = np.eye(4)
            rot_90[:3, :3] = Rotation.from_euler('y', 270, degrees=True).as_matrix()
            rot_90 = torch.FloatTensor(rot_90).to(vertices.device)
            vertices = vertices @ rot_90.T
            vertices[:, 0] = -vertices[:, 0]
            vertices_mean = (vertices.amax(dim=0) + vertices.amin(dim=0)) / 2
            vertices[:, :3] = vertices[:, :3] - vertices_mean[:3]
            vertices = vertices + norm_params[0]
            vertices_max = vertices[:, 1].max()
            vertices = vertices / vertices_max
            vertices = vertices * norm_params[1]

            R, T = look_at_view_transform(1.2,
                                          camera_delta_elevation_array[0],
                                          camera_delta_angle_array[0],
                                          up=((0, 1, 0),),
                                          at=((0.0, 0.0, 0.0),),
                                          degrees=False)  # 1.2
            self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=1e-5,
                faces_per_pixel=1,
            )
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings
                ),
                shader=self.shader
            )

        if transform_matrices is not None:
            vertices_transformed = vertices @ transform_matrices[0].T
            vertices_transformed = vertices_transformed @ transform_matrices[1].T

            if not render_scannet:
                vertices_mean = (vertices_transformed.amax(dim=0) + vertices_transformed.amin(dim=0)) / 2
                vertices_transformed[:, :3] = vertices_transformed[:, :3] - vertices_mean[:3]
                vertices_max = vertices_transformed[:, 1].max()
                vertices_transformed = vertices_transformed / vertices_max
                vertices_transformed = vertices_transformed * 0.4
                vertices_max = vertices_transformed[:, 1].max()

            if not render_scannet:
                R, T = look_at_view_transform(1.2, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
                self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)

                raster_settings = RasterizationSettings(
                    image_size=self.resolution,
                    blur_radius=1e-5,
                    faces_per_pixel=1,
                )
                self.renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=self.cameras,
                        raster_settings=raster_settings
                    ),
                    shader=self.shader
                )
            else:
                raster_settings = RasterizationSettings(
                    image_size=(484, 648),
                    blur_radius=1e-5,
                    faces_per_pixel=1
                )
                R, T = look_at_view_transform(0.001, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
                self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=45.0, znear=0.1)
                self.renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=self.cameras,
                        raster_settings=raster_settings
                    ),
                    shader=self.shader
                )
        else:
            vertices_transformed = vertices

        vertices_render = vertices_transformed

        num_shapes = len(ranges)
        all_images = []
        if camera_delta_elevation_array is None:
            camera_delta_elevation_array = [np.pi] * num_shapes
        for view in range(num_shapes):
            if color_bg is None:
                background_color = None
            else:
                background_color = color_bg[view:view + 1]

            if view != 0 or transform_matrices is None:
                vertices_render = vertices
                if camera_delta_angle_array is not None:
                    R, T = look_at_view_transform(1.2, camera_delta_elevation_array[view],
                                                  camera_delta_angle_array[view], up=((0, 1, 0),),
                                                  at=((0.0, 0.0, 0.0),), degrees=False)  # 1.2
                    self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
                    self.shader = ColorShader(cameras=self.cameras)
                    raster_settings = RasterizationSettings(
                        image_size=self.resolution,
                        blur_radius=1e-5,
                        faces_per_pixel=1,
                    )
                    self.renderer = MeshRenderer(
                        rasterizer=MeshRasterizer(
                            cameras=self.cameras,
                            raster_settings=raster_settings
                        ),
                        shader=self.shader
                    )

            verts = _make_tensor(vertices_render[:, :3], cols=3, dtype=torch.float32, device=device)
            faces_verts_idx = _make_tensor(
                faces[ranges[view, 0]: ranges[view, 0] + ranges[view, 1]], cols=3,
                dtype=torch.int64, device=device)
            p3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces_verts_idx.to(device)])
            image = self.renderer(p3d_mesh, color=vertex_colors, color_bg=background_color)
            all_images += [image[:, :, :, :4]]

            if only_first_view:
                break

        all_images = torch.cat(all_images, dim=0)

        return all_images


class RenderMLPPrologue(nn.Module):
    def __init__(self, feature_size=256, geo_feature_size=256, baryc_size=3, aux_latent_size=256):
        super(RenderMLPPrologue, self).__init__()
        self.lin_feat_1 = nn.Linear(feature_size, 256)
        self.lin_feat_2 = nn.Linear(256, 128, bias=False)

        self.lin_geo_feat_1 = nn.Linear(geo_feature_size, 256)
        self.lin_geo_feat_2 = nn.Linear(256, 128, bias=False)

        self.lin_aux_lat_1 = nn.Linear(aux_latent_size, 256)
        self.lin_aux_lat_2 = nn.Linear(256, 128, bias=False)

        self.lin_1 = nn.Linear(128 + 128 + 128, 256, bias=False)
        self.lin_2 = nn.Linear(256, 256, bias=False)

        self.main_activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, feature=None, geo_feature=None, baryc=None, aux_latent=None):
        feature = self.main_activation(self.lin_feat_1(feature))
        feature = self.main_activation(self.lin_feat_2(feature))

        geo_feature = self.main_activation(self.lin_geo_feat_1(geo_feature))
        geo_feature = self.main_activation(self.lin_geo_feat_2(geo_feature))

        aux_latent = self.main_activation(self.lin_aux_lat_1(aux_latent))
        aux_latent = self.main_activation(self.lin_aux_lat_2(aux_latent))

        concat_feature = torch.hstack([feature, geo_feature, aux_latent])
        concat_feature = self.main_activation(self.lin_1(concat_feature))
        concat_feature = self.main_activation(self.lin_2(concat_feature))

        return concat_feature


class RenderConvModule(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(RenderConvModule, self).__init__()
        self.conv1 = EqualizedConv2d(in_channels, 384, kernel_size=5, stride=2, activation='lrelu')
        self.conv2 = EqualizedConv2d(384, 384, kernel_size=5, stride=1, activation='lrelu')
        self.conv3 = EqualizedConv2d(384, out_channels, kernel_size=5, stride=1, activation='lrelu')

    def forward(self, x):
        input_size = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.upsample(x, size=(input_size[2], input_size[3]), mode='bilinear')
        x = self.conv3(x)
        return x


class RenderMLPEpilogue(nn.Module):
    def __init__(self, feature_size=256):
        super(RenderMLPEpilogue, self).__init__()
        self.lin_3 = nn.Linear(feature_size, 256, bias=False)

        self.lin_4 = nn.Linear(256, 256, bias=False)
        self.lin_5 = nn.Linear(256, 3, bias=False)

        self.main_activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):

        concat_feature = self.main_activation(self.lin_3(x))
        concat_feature = self.main_activation(self.lin_4(x))
        color = self.lin_5(x)

        return color


class RenderMLP(nn.Module):
    def __init__(self, feature_size=256, geo_feature_size=256, baryc_size=3, aux_latent_size=256):
        super(RenderMLP, self).__init__()
        self.lin_feat_1 = nn.Linear(feature_size, 256)
        self.lin_feat_2 = nn.Linear(256, 128, bias=False)

        self.lin_geo_feat_1 = nn.Linear(geo_feature_size, 256)
        self.lin_geo_feat_2 = nn.Linear(256, 128, bias=False)

        self.lin_aux_lat_1 = nn.Linear(aux_latent_size, 256)
        self.lin_aux_lat_2 = nn.Linear(256, 128, bias=False)

        self.lin_1 = nn.Linear(128 + 128 + 128, 256, bias=False)
        self.lin_2 = nn.Linear(256, 256, bias=False)
        self.lin_3 = nn.Linear(256, 256, bias=False)
        self.lin_4 = nn.Linear(256, 256, bias=False)
        self.lin_5 = nn.Linear(256, 3, bias=False)

        self.main_activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, feature=None, geo_feature=None, baryc=None, aux_latent=None):
        feature = self.main_activation(self.lin_feat_1(feature))
        feature = self.main_activation(self.lin_feat_2(feature))

        geo_feature = self.main_activation(self.lin_geo_feat_1(geo_feature))
        geo_feature = self.main_activation(self.lin_geo_feat_2(geo_feature))

        aux_latent = self.main_activation(self.lin_aux_lat_1(aux_latent))
        aux_latent = self.main_activation(self.lin_aux_lat_2(aux_latent))

        concat_feature = torch.hstack([feature, geo_feature, aux_latent])
        concat_feature = self.main_activation(self.lin_1(concat_feature))
        concat_feature = self.main_activation(self.lin_2(concat_feature))

        concat_feature = self.main_activation(self.lin_3(concat_feature))
        concat_feature = self.main_activation(self.lin_4(concat_feature))
        color = self.lin_5(concat_feature)

        return color


class RenderMLP2(nn.Module):
    def __init__(self, feature_size=256, geo_feature_size=256, baryc_size=3, aux_latent_size=256, activation='lrelu', lr_multiplier=1.0):
        super(RenderMLP2, self).__init__()
        self.lin_feat_1 = FullyConnectedLayer(feature_size, 256, activation=activation, lr_multiplier=lr_multiplier)
        self.lin_feat_2 = FullyConnectedLayer(256, 128, bias=False, activation=activation, lr_multiplier=lr_multiplier)

        self.lin_geo_feat_1 = FullyConnectedLayer(geo_feature_size, 256, activation=activation, lr_multiplier=lr_multiplier)
        self.lin_geo_feat_2 = FullyConnectedLayer(256, 128, bias=False, activation=activation, lr_multiplier=lr_multiplier)

        self.lin_aux_lat_1 = FullyConnectedLayer(aux_latent_size, 256, activation=activation, lr_multiplier=lr_multiplier)
        self.lin_aux_lat_2 = FullyConnectedLayer(256, 256, bias=False, activation=activation, lr_multiplier=lr_multiplier)

        self.lin_1 = FullyConnectedLayer(128 + 128 + 256, 256, bias=False, activation=activation, lr_multiplier=lr_multiplier)
        self.lin_2 = FullyConnectedLayer(256, 256, bias=False, activation=activation, lr_multiplier=lr_multiplier)
        self.lin_3 = FullyConnectedLayer(256, 256, bias=False, activation=activation, lr_multiplier=lr_multiplier)
        self.lin_4 = FullyConnectedLayer(256, 3, bias=False, lr_multiplier=lr_multiplier)

    def forward(self, feature=None, geo_feature=None, baryc=None, aux_latent=None):
        feature = self.lin_feat_1(feature)
        feature = self.lin_feat_2(feature)

        geo_feature = self.lin_geo_feat_1(geo_feature)
        geo_feature = self.lin_geo_feat_2(geo_feature)

        aux_latent = self.lin_aux_lat_1(aux_latent)
        aux_latent = self.lin_aux_lat_2(aux_latent)

        concat_feature = torch.hstack([feature, geo_feature, aux_latent])
        concat_feature = self.lin_1(concat_feature)
        concat_feature = self.lin_2(concat_feature)
        concat_feature = self.lin_3(concat_feature)
        color = self.lin_4(concat_feature)

        return color


class ShapeNetRenderer(nn.Module):

    def __init__(self, resolution, num_channels=3):
        super().__init__()
        self.resolution = resolution
        self.render_resolution = int(1.1 * self.resolution)
        self.num_channels = num_channels

        self.shapenet_dataset = ShapeNetCore('/canis/ShapeNet/ShapeNetCore.v2', synsets=['chair'], version=2, texture_resolution=100)

        R, T = look_at_view_transform(1.2, np.pi, 0.0, up=((0, 1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
        self.shader = ColorShader(cameras=self.cameras)
        self.raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            cull_backfaces=True,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device='cuda')[None], device='cuda')


    def render(self, obj_id=None, resolution=None, camera_delta_angle=None, camera_elevation=None):

        print('renderer obj_id', obj_id)

        if camera_elevation is None:
            camera_elevation = np.pi

        if camera_delta_angle is not None:
            R, T = look_at_view_transform(1.2, camera_elevation, 0.0 + camera_delta_angle, up=((0, 1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
            self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
            self.shader = ColorShader(cameras=self.cameras)

        images_by_model_ids = self.shapenet_dataset.render(
            model_ids=[
                obj_id,
            ],
            device='cuda',
            cameras=self.cameras,
            raster_settings=self.raster_settings,
            # lights=self.lights,
            shader_type=HardFlatShader
        )
        print('images', images_by_model_ids.shape)

        return images_by_model_ids


class ShapeNetCarsRenderer(nn.Module):

    def __init__(self, resolution, num_channels=3):
        super().__init__()
        self.resolution = resolution
        self.render_resolution = int(1.1 * self.resolution)
        self.num_channels = num_channels

        self.shapenet_dataset = ShapeNetCore('/canis/ShapeNet/ShapeNetCore.v2', synsets=['car'], version=2, texture_resolution=100)

        R, T = look_at_view_transform(1.2, np.pi, 0.0, up=((0, 1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
        self.shader = ColorShader(cameras=self.cameras)
        self.raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            cull_backfaces=True,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device='cuda')[None], device='cuda')


    def render(self, obj_id=None, resolution=None, camera_delta_angle=None, camera_elevation_angle=None,
               camera_dist=None):

        camera_elevation = np.pi

        if camera_delta_angle is not None:
            R, T = look_at_view_transform(camera_dist, camera_elevation + camera_elevation_angle, 0.0 + camera_delta_angle, up=((0, 1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
            self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=60.0, znear=0.1)
            self.shader = ColorShader(cameras=self.cameras)

        images_by_model_ids = self.shapenet_dataset.render(
            model_ids=[
                obj_id,
            ],
            device='cuda',
            cameras=self.cameras,
            raster_settings=self.raster_settings,
            # lights=self.lights,
            shader_type=HardFlatShader
        )
        print('images', images_by_model_ids.shape)

        return images_by_model_ids


class ChairRenderer(nn.Module):

    def __init__(self, img_size=None):
        super().__init__()

        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            cull_backfaces=True,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device='cuda')[None], device='cuda')


    def render(self, mesh, transform_matrices=None, K=None):

        R, T = look_at_view_transform(0.001, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
        # self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, K=K)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=45.0, znear=0.1)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=HardFlatShader(
                device='cuda',
                cameras=self.cameras,
                lights=self.lights
            )
        )

        vertices = mesh._verts_list[0]
        vertices = torch.hstack([vertices, torch.ones((len(vertices), 1)).to(vertices.device)])
        vertices = vertices @ transform_matrices[0].T
        vertices = vertices @ transform_matrices[1].T
        mesh._verts_list[0] = vertices[:, :3]

        images = renderer(mesh, lights=self.lights)

        return images


class CarsRenderer(nn.Module):

    def __init__(self, img_size=None):
        super().__init__()

        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            cull_backfaces=True,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device='cuda')[None], device='cuda')


    def render(self, mesh, camera_elevation, camera_azimuth):

        R, T = look_at_view_transform(1.2, camera_elevation, camera_azimuth, up=((0, 1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0, znear=0.1)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=HardFlatShader(
                device='cuda',
                cameras=self.cameras,
                lights=self.lights
            )
        )

        images = renderer(mesh, lights=self.lights)

        return images


def rotation_y(angle):

    matrix = torch.FloatTensor([[torch.cos(angle), 0, torch.sin(angle), 0],
                                [0, 1, 0, 0],
                                [-torch.sin(angle), 0, torch.cos(angle), 0],
                                [0, 0, 0, 1]])
    return matrix


def rotation_z(angle):

    matrix = torch.FloatTensor([[torch.cos(angle), -torch.sin(angle), 0, 0],
                                [torch.sin(angle), torch.cos(angle), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    return matrix


class DifferentiableMLPRenderer(nn.Module):

    def __init__(self, resolution, num_channels=3, small_shader=False, residual=False, tanh=False, use_act=True,
                 mode='chair'):
        super().__init__()
        self.resolution = resolution
        self.render_resolution = int(1.1 * self.resolution)
        self.num_channels = num_channels
        self.residual = residual
        self.tanh = tanh
        self.use_act = use_act
        self.mode = mode

        R, T = look_at_view_transform(2.4, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 1.55),), degrees=False)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)

        if self.use_act:
            if self.tanh:
                last_act = nn.Tanh()
            else:
                last_act = nn.ReLU()
        else:
            last_act = nn.Identity()

        main_activation = nn.LeakyReLU(negative_slope=0.1)

        if small_shader:
            self.mlp = nn.Sequential(nn.Linear(384 + 256, 64, bias=False), # 256 + 64 / 384, 131
                                     main_activation,
                                     nn.Linear(64, 3, bias=False),
                                     last_act
                                     )
        else:

            self.mlp = nn.Sequential(nn.Linear(384 + 256, 256, bias=False),  # 256 + 64 + 256 / 384 + 256, 131
                                     main_activation,
                                     nn.Linear(256, 256, bias=False),
                                     main_activation,
                                     nn.Linear(256, 256, bias=False),
                                     main_activation,
                                     nn.Linear(256, 256, bias=False),
                                     main_activation,
                                     nn.Linear(256, 3, bias=False),
                                     last_act
                                     )

        if mode == 'car':
            self.mlp = RenderMLP2(feature_size=256, geo_feature_size=128, baryc_size=63, aux_latent_size=512) # cars
        elif mode == 'chair':
            self.mlp = RenderMLP2(feature_size=256, geo_feature_size=64, baryc_size=63, aux_latent_size=512) # chairs

        self.aux_latent = torch.FloatTensor(torch.rand([1, 256])).cuda()
        self.aux_latent.requires_grad = True

        # self.mlp_prologue = RenderMLPPrologue(feature_size=256, geo_feature_size=128, baryc_size=63, aux_latent_size=512)
        # self.conv_module = RenderConvModule(in_channels=256, out_channels=256)
        # self.mlp_epilogue = RenderMLPEpilogue(feature_size=256)

        # self.mlp.apply(sine_init)
        # self.mlp[0].apply(first_layer_sine_init)

        if not self.residual:
            self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                        feature_dim=256, mode=mode)

            # self.shader = HardMLPConvShader(cameras=self.cameras, mlp_prologue=self.mlp_prologue,
            #                                 conv_module=self.conv_module, mlp_epilogue=self.mlp_epilogue,
            #                                 tanh=self.tanh, use_act=self.use_act,
            #                                 feature_dim=256)
        else:
            self.shader = HardColorMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                             feature_dim=256)

        raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            blur_radius=1e-5,
            faces_per_pixel=1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=self.shader
        )

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )
        cameras = FoVPerspectiveCameras(device='cuda')
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftGouraudShader(device='cuda', cameras=cameras)
        )

        raster_settings = RasterizationSettings(
            image_size=self.render_resolution,
            blur_radius=1e-5,
            faces_per_pixel=1
        )
        lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device='cuda')[None], device='cuda')
        self.flat_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardFlatShader(
                device='cuda',
                cameras=cameras,
                lights=lights
            )
        )

        self.nocs_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=NOCsShader(
                device='cuda',
                cameras=cameras
            )
        )

    def create_angles(self, init_azimuth=None, init_elevation=None):

        if init_azimuth is None:
            init_azimuth = 0.0
        if init_elevation is None:
            init_elevation = np.pi

        self.camera_azimuth = nn.Parameter(
            torch.from_numpy(np.array([init_azimuth], dtype=np.float32)).cuda())
        self.camera_elevation = nn.Parameter(
            torch.from_numpy(np.array([init_elevation], dtype=np.float32)).cuda())
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float32)).cuda())

    def render(self, vertices, faces, vertex_features, ranges, vertex_colors=None, color_bg=None, shape=None, resolution=None,
               style_latent=None, views_per_shape=1, norm_params=None, camera_delta_angle=None, transform_matrices=None,
               camera_delta_angle_array=None, camera_delta_elevation_array=None, only_first_view=False, opt_azimuth=None, opt_elevation=None,
               render_scannet=False, unaligned=False, scannet_mode=False, render_nocs=False, render_sil=False):

        device = vertices.device
        if device != self.cameras.device:
            print('Incorrect device!')
            R, T = look_at_view_transform(2.4, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 1.55),), degrees=False)
            self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
            if not self.residual:
                self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                            feature_dim=256, mode=self.mode)
                # self.shader = HardMLPConvShader(cameras=self.cameras, mlp_prologue=self.mlp_prologue,
                #                                 conv_module=self.conv_module, mlp_epilogue=self.mlp_epilogue,
                #                                 tanh=self.tanh, use_act=self.use_act,
                #                                 feature_dim=256)
            else:
                self.shader = HardColorMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                                 feature_dim=256)
            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=1e-5,
                faces_per_pixel=1,
            )
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings
                ),
                shader=self.shader
            )
            self.aux_latent = self.aux_latent.to(device)

        if resolution is not None:
            if self.resolution != resolution:
                self.resolution = resolution
                self.render_resolution = int(1.1 * self.resolution)
                R, T = look_at_view_transform(2.4, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 1.55),), degrees=False)
                self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
                if not self.residual:
                    self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh,
                                                use_act=self.use_act,
                                                feature_dim=256, mode=self.mode)
                    # self.shader = HardMLPConvShader(cameras=self.cameras, mlp_prologue=self.mlp_prologue,
                    #                                 conv_module=self.conv_module, mlp_epilogue=self.mlp_epilogue,
                    #                                 tanh=self.tanh, use_act=self.use_act,
                    #                                 feature_dim=256)
                else:
                    self.shader = HardColorMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh,
                                                     use_act=self.use_act,
                                                     feature_dim=256)
                raster_settings = RasterizationSettings(
                    image_size=self.render_resolution,
                    blur_radius=1e-5,
                    faces_per_pixel=1,
                )
                self.renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=self.cameras,
                        raster_settings=raster_settings
                    ),
                    shader=self.shader
                )

        if norm_params is not None:
            rot_90 = np.eye(4)
            rot_90[:3, :3] = Rotation.from_euler('y', 270, degrees=True).as_matrix()
            rot_90 = torch.FloatTensor(rot_90).to(vertices.device)
            vertices = vertices @ rot_90.T
            vertices[:, 0] = -vertices[:, 0]
            vertices_mean = (vertices.amax(dim=0) + vertices.amin(dim=0)) / 2
            vertices[:, :3] = vertices[:, :3] - vertices_mean[:3]
            vertices = vertices + norm_params[0]
            vertices_max = vertices[:, 1].max()
            vertices = vertices / vertices_max
            vertices = vertices * norm_params[1]
            vertices_max = vertices[:, 1].max()
            vertices_min = vertices[:, 1].min()

            R, T = look_at_view_transform(1.2,
                                          np.pi,
                                          camera_delta_angle_array[0],
                                          up=((0, 1, 0),),
                                          at=((0.0, 0.0, 0.0),),
                                          degrees=False) # 1.2
            self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
            self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                        feature_dim=256, mode=self.mode)
            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=1e-5,
                faces_per_pixel=1,
            )
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings
                ),
                shader=self.shader
            )

        if transform_matrices is not None:
            vertices_transformed = vertices @ transform_matrices[0].T
            vertices_transformed = vertices_transformed @ transform_matrices[1].T

            if not render_scannet:
                vertices_mean = (vertices_transformed.amax(dim=0) + vertices_transformed.amin(dim=0)) / 2
                vertices_transformed[:, :3] = vertices_transformed[:, :3] - vertices_mean[:3]
                vertices_max = vertices_transformed[:, 1].max()
                vertices_transformed = vertices_transformed / vertices_max
                vertices_transformed = vertices_transformed * 0.4 # 0.4
                vertices_max = vertices_transformed[:, 1].max()

            if not render_scannet:
                R, T = look_at_view_transform(1.2, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
                self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
                self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                            feature_dim=256, mode=self.mode)
                raster_settings = RasterizationSettings(
                    image_size=self.resolution,
                    blur_radius=1e-5,
                    faces_per_pixel=1,
                )
                self.renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=self.cameras,
                        raster_settings=raster_settings
                    ),
                    shader=self.shader
                )
            else:
                raster_settings = RasterizationSettings(
                    image_size=(484, 648),
                    blur_radius=1e-5,
                    faces_per_pixel=1
                )
                R, T = look_at_view_transform(0.001, np.pi, 0.0, up=((0, -1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)
                self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=45.0, znear=0.1)
                self.renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=self.cameras,
                        raster_settings=raster_settings
                    ),
                    shader=self.shader
                )
        else:
            vertices_transformed = vertices

        vertices_render = vertices_transformed

        num_shapes = len(ranges)
        all_images = []
        if camera_delta_elevation_array is None:
            camera_delta_elevation_array = [np.pi] * num_shapes
        for view in range(num_shapes):
            lat_index = int(view / views_per_shape)
            if view >= 0:
                print_stats = True
            else:
                print_stats = False

            if color_bg is None:
                background_color = None
            else:
                background_color = color_bg[view:view + 1]

            if view != 0 or transform_matrices is None:
                vertices_render = vertices
                if camera_delta_angle_array is not None:

                    R, T = look_at_view_transform(1.2, camera_delta_elevation_array[view], camera_delta_angle_array[view], up=((0, 1, 0),), at=((0.0, 0.0, 0.0),), degrees=False)  # 1.2
                    self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0, fov=50.0)
                    self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, tanh=self.tanh, use_act=self.use_act,
                                                feature_dim=256, mode=self.mode)
                    raster_settings = RasterizationSettings(
                        image_size=self.resolution,
                        blur_radius=1e-5,
                        faces_per_pixel=1,
                    )
                    self.renderer = MeshRenderer(
                        rasterizer=MeshRasterizer(
                            cameras=self.cameras,
                            raster_settings=raster_settings
                        ),
                        shader=self.shader
                    )

            verts = _make_tensor(vertices_render[:, :3], cols=3, dtype=torch.float32, device=device)
            faces_verts_idx = _make_tensor(
                faces[ranges[view, 0]: ranges[view, 0] + ranges[view, 1]], cols=3,
                dtype=torch.int64, device=device)
            p3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces_verts_idx.to(device)])
            if not self.residual:
                image = self.renderer(p3d_mesh, feature=vertex_features, color_bg=background_color, shape_features=shape,
                                      aux_latent=self.aux_latent, print_stats=print_stats, style_latent=style_latent[lat_index]
                                      )
            else:
                image = self.renderer(p3d_mesh, feature=vertex_features, color=vertex_colors, color_bg=background_color, shape_features=shape,
                                      aux_latent=self.aux_latent, print_stats=print_stats, style_latent=style_latent
                                      ) # color_bg[view:view + 1]

            all_images += [image[:, :, :, :4]]

            if only_first_view:
                break
        all_images = torch.cat(all_images, dim=0)

        if render_nocs:
            if scannet_mode:
                up_vector = -1
            else:
                up_vector = 1
            if unaligned:
                R, T = look_at_view_transform(1.2,
                                              self.camera_elevation,
                                              self.camera_azimuth,
                                              up=((0, up_vector, 0),),
                                              at=((0.0, 0.0, 0.0),),
                                              degrees=False)
            else:
                R, T = look_at_view_transform(1.2,
                                              camera_delta_elevation_array[0],
                                              camera_delta_angle_array[0],
                                              up=((0, up_vector, 0),),
                                              at=((0.0, 0.0, 0.0),),
                                              degrees=False)
            R = R.to(device)
            T = T.to(device)

            verts = _make_tensor(vertices_transformed[:, :3], cols=3, dtype=torch.float32, device=device)
            faces_verts_idx = _make_tensor(
                faces[ranges[0, 0]: ranges[0, 0] + ranges[0, 1]], cols=3,
                dtype=torch.int64, device=device)
            p3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces_verts_idx.to(device)])
            image_ref = self.nocs_renderer(p3d_mesh, color_bg=background_color, R=R, T=T, fov=50.0, znear=0.01, orig_vertices=vertices[:, :3])
        else:
            image_ref = None

        if render_sil:
            R, T = look_at_view_transform(1.2,
                                          self.camera_elevation,
                                          self.camera_azimuth,
                                          up=((0, 1, 0),),
                                          at=((0.0, 0.0, 0.0),),
                                          degrees=False)  # 1.2
            R = R.to(device)
            T = T.to(device)
            sillhouette = self.silhouette_renderer(p3d_mesh, R=R, T=T, fov=50.0)
        else:
            sillhouette = None

        return all_images, image_ref, sillhouette


class DifferentiableMultiMLPRenderer(nn.Module):

    def __init__(self, resolution, num_channels=3):
        super().__init__()
        self.resolution = resolution
        self.num_channels = num_channels

        R, T = look_at_view_transform(4.5, 0, 0, up=((0, -1, 0),), degrees=True)
        self.cameras_1 = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0)
        self.cameras_2 = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0)

        self.mlp_1 = nn.ModuleList([nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 3),
                                    nn.ReLU()
                                    ])

        self.mlp_2 = nn.ModuleList([nn.Linear(384, 384),
                                    nn.ReLU(),
                                    nn.Linear(384, 384),
                                    nn.ReLU(),
                                    nn.Linear(384, 3),
                                    nn.ReLU()
                                    ])

        self.mlps = [self.mlp_1, self.mlp_2]

        self.shader_1 = HardMLPShader(cameras=self.cameras_1, mlp=self.mlp_1, feature_dim=256)
        self.shader_2 = HardMLPShader(cameras=self.cameras_2, mlp=self.mlp_2, feature_dim=384)

        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0,
            faces_per_pixel=1,
        )
        self.renderer_1 = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras_1,
                raster_settings=raster_settings
            ),
            shader=self.shader_1
        )
        self.renderer_2 = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras_2,
                raster_settings=raster_settings
            ),
            shader=self.shader_2
        )

    def render(self, vertices_maps, faces_maps, vertex_features, ranges_maps, new_device=None):
        device = vertices_maps[0].device
        if new_device is not None:
            device = new_device
        if device != self.cameras_1.device:
            print('Incorrect device! (1)', device, self.cameras_1.device)
            R, T = look_at_view_transform(4.5, 0, 0, up=((0, -1, 0),), degrees=True)
            self.cameras_1 = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=5.0)
            self.shader_1 = HardMLPShader(cameras=self.cameras_1, mlp=self.mlp_1, feature_dim=256)
            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=0,
                faces_per_pixel=1,
            )
            self.renderer_1 = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras_1,
                    raster_settings=raster_settings
                ),
                shader=self.shader_1
            )

        if device != self.cameras_2.device:
            print('Incorrect device! (2)', device, self.cameras_2.device)
            R, T = look_at_view_transform(4.5, 0, 0, up=((0, -1, 0),), degrees=True)
            self.cameras_2 = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=5.0)
            self.shader_2 = HardMLPShader(cameras=self.cameras_2, mlp=self.mlp_2, feature_dim=384)
            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=0,
                faces_per_pixel=1,
            )
            self.renderer_2 = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras_2,
                    raster_settings=raster_settings
                ),
                shader=self.shader_2
            )

        num_shapes = len(ranges_maps[0])
        all_images = []
        for view in range(num_shapes):
            verts = _make_tensor(vertices_maps[0][:, :3], cols=3, dtype=torch.float32, device=device)
            faces_verts_idx = _make_tensor(
                faces_maps[0][ranges_maps[0][view, 0]: ranges_maps[0][view, 0] + ranges_maps[0][view, 1]], cols=3,
                dtype=torch.int64, device=device)
            p3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces_verts_idx.to(device)])

            mesh_sublevels = []
            for i in range(len(self.mlps) - 1):
                verts = _make_tensor(vertices_maps[i + 1][:, :3], cols=3, dtype=torch.float32, device=device)
                faces_verts_idx = _make_tensor(
                    faces_maps[i + 1][ranges_maps[i + 1][view, 0]: ranges_maps[i + 1][view, 0] + ranges_maps[i + 1][view, 1]], cols=3,
                    dtype=torch.int64, device=device)
                p3d_mesh_level = Meshes(verts=[verts.to(device)], faces=[faces_verts_idx.to(device)])
                mesh_sublevels += [p3d_mesh_level]

            image_1 = self.renderer_1(p3d_mesh, feature=vertex_features[0], id='1')
            image_2 = self.renderer_2(mesh_sublevels[0], feature=vertex_features[1], id='2')
            image_1, mask_1 = image_1[..., :3], image_1[..., 3:]
            image_2, mask_2 = image_2[..., :3], image_2[..., 3:]
            mask = (mask_1 * mask_2).bool()
            image_2[(~mask).repeat(1, 1, 1, 3)] = 0
            image = image_1 + image_2
            all_images += [image]
        all_images = torch.cat(all_images, dim=0)

        return all_images


class DifferentiableColorMLPRenderer(nn.Module):

    def __init__(self, resolution, num_channels=3, small_shader=False, residual=False):
        super().__init__()
        self.resolution = resolution
        self.num_channels = num_channels
        self.residual = residual

        R, T = look_at_view_transform(4.5, 0, 0, up=((0, -1, 0),), degrees=True)
        self.cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, zfar=5.0)

        if small_shader:
            self.mlp = nn.ModuleList([nn.Linear(3, 64, bias=True),
                                      nn.ReLU(),
                                      nn.Linear(64, 3, bias=True),
                                      nn.ReLU()
                                      ])
        else:
            self.mlp = nn.ModuleList([nn.Linear(3, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 3),
                                      nn.ReLU()
                                      ])

        self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, feature_dim=3,
                                    residual=self.residual)

        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0,
            faces_per_pixel=1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=self.shader
        )

    def render(self, vertices, faces, vertex_features, ranges):

        device = vertices.device
        if device != self.cameras.device:
            print('Incorrect device!')
            R, T = look_at_view_transform(4.5, 0, 0, up=((0, -1, 0),), degrees=True)
            self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=5.0)
            self.shader = HardMLPShader(cameras=self.cameras, mlp=self.mlp, feature_dim=3, residual=self.residual)
            raster_settings = RasterizationSettings(
                image_size=self.resolution,
                blur_radius=0,
                faces_per_pixel=1,
            )
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings
                ),
                shader=self.shader
            )

        num_shapes = len(ranges)
        all_images = []
        for view in range(num_shapes):
            verts = _make_tensor(vertices[:, :3], cols=3, dtype=torch.float32, device=device)
            faces_verts_idx = _make_tensor(
                faces[ranges[view, 0]: ranges[view, 0] + ranges[view, 1]], cols=3,
                dtype=torch.int64, device=device)
            p3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces_verts_idx.to(device)])
            image = self.renderer(p3d_mesh, feature=vertex_features)
            all_images += [image[:, :, :, :3]]
        all_images = torch.cat(all_images, dim=0)

        return all_images