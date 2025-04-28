import torch
import torch_geometric
import torch_scatter
import math
from torch.nn import init
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch_utils.ops import conv2d_resample


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30.0 * input)


def sine_init(m, scale=30.0):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / scale, np.sqrt(6 / num_input) / scale)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def pool(x, node_count, pool_map, lateral_map, pool_op='max'):
    if pool_op == 'max':
        x_pooled = torch.ones((node_count, x.shape[1]), dtype=x.dtype).to(x.device) * (x.min().detach() - 1e-3)
        torch_scatter.scatter_max(x, pool_map, dim=0, out=x_pooled)
        x_pooled[lateral_map[:, 0], :] = x_pooled[lateral_map[:, 1], :]
    elif pool_op == 'mean':
        x_pooled = torch.zeros((node_count, x.shape[1]), dtype=x.dtype).to(x.device)
        torch_scatter.scatter_mean(x, pool_map, dim=0, out=x_pooled)
        x_pooled[lateral_map[:, 0], :] = x_pooled[lateral_map[:, 1], :]
    else:
        raise NotImplementedError
    return x_pooled


def unpool(x, pool_map):
    x_unpooled = x[pool_map, :]
    return x_unpooled


def create_faceconv_input(x, kernel_size, face_neighborhood, face_is_pad, pad_size):
    padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
    padded_x[torch.logical_not(face_is_pad), :] = x
    f_ = [padded_x[face_neighborhood[:, i]] for i in range(kernel_size)]
    conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3)
    return conv_input


def modulated_face_conv(x, weight, styles, demodulate=True, semantic_maps=None,
                        batch_size=None, num_classes=None, verbose=0):
    batch_size = styles.shape[0]
    num_faces = x.shape[0] // batch_size
    out_channels, in_channels, kh, kw = weight.shape

    w = weight.unsqueeze(0)
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)

    x = x.reshape(batch_size, num_faces, *x.shape[1:]).permute((1, 0, 2, 3, 4)).reshape(num_faces, batch_size * in_channels, 1, weight.shape[-1])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w, groups=batch_size)
    x = x.reshape(num_faces, batch_size * out_channels).reshape(num_faces, batch_size, out_channels).permute((1, 0, 2)).reshape((batch_size * num_faces, out_channels))
    return x


def modulated_face_efficent_conv(x, weight, styles, demodulate=True, semantic_maps=None,
                                 batch_size=None, num_classes=None, verbose=0):
    batch_size = styles.shape[0]
    num_faces = x.shape[0] // batch_size
    out_channels, in_channels, kh, kw = weight.shape

    w = weight.unsqueeze(0)
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)

    x = x.reshape(batch_size, num_faces, *x.shape[1:]).permute((1, 0, 2, 3, 4)).reshape(num_faces, batch_size * in_channels, 1, weight.shape[-1])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w, f=None, up=1, down=1, groups=batch_size,
                                        flip_weight=False)
    x = x.reshape(num_faces, batch_size * out_channels).reshape(num_faces, batch_size, out_channels).permute((1, 0, 2)).reshape((batch_size * num_faces, out_channels))
    return x


def modulated_part_face_conv(x, weight, styles, demodulate=True, semantic_maps=None,
                             batch_size=None, num_classes=None, verbose=0):
    batch_class_size = styles.shape[0]
    styles_feat_size = styles.shape[-1]
    num_faces = x.shape[0] // batch_size
    out_channels, in_channels, kh, kw = weight.shape

    styles = styles.reshape(batch_size, num_classes, styles_feat_size)
    semantic_maps = semantic_maps.permute((1, 0))
    for i in range(num_classes):
        class_styles = styles[:, i, ...]
        w = weight.unsqueeze(0)
        w = w * class_styles.reshape(batch_size, 1, -1, 1, 1)
        if demodulate:
            dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)
        
        x_class = x.reshape(batch_size, num_faces, *x.shape[1:]).permute((1, 0, 2, 3, 4)).reshape(num_faces, batch_size * in_channels, 1, weight.shape[-1])
        w = w.reshape(-1, in_channels, kh, kw)
        x_class = torch.nn.functional.conv2d(x_class, w, groups=batch_size)
        x_class = x_class.reshape(num_faces, batch_size * out_channels).reshape(num_faces, batch_size, out_channels)
        if i == 0:
            x_out = torch.clone(x_class)
            x_out = x_out * 0
        cls_mask = torch.where(semantic_maps.int() == i)
        x_out[cls_mask] = x_class[cls_mask]

    # x = x_out.reshape(num_faces, batch_size * out_channels).reshape(num_faces, batch_size, out_channels).permute((1, 0, 2)).reshape((batch_size * num_faces, out_channels))
    x = x_out.permute((1, 0, 2)).reshape((batch_size * num_faces, out_channels))
    return x


def modulated_texture_conv(x, weight_center, weight_side, weight_corner, styles, aggregate_function, demodulate=True):
    batch_size = styles.shape[0]
    num_faces = x.shape[0] // batch_size
    out_channels, in_channels = weight_center.shape[:2]
    w_center = weight_center.unsqueeze(0)
    w_center = w_center * styles.reshape(batch_size, 1, -1, 1, 1)
    w_side = weight_side.unsqueeze(0)
    w_side = w_side * styles.reshape(batch_size, 1, -1, 1, 1)
    w_corner = weight_corner.unsqueeze(0)
    w_corner = w_corner * styles.reshape(batch_size, 1, -1, 1, 1)

    if demodulate:
        dcoefs = (w_center.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w_center = w_center * dcoefs.reshape(batch_size, -1, 1, 1, 1)
        dcoefs = (w_side.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w_side = w_side * dcoefs.reshape(batch_size, -1, 1, 1, 1)
        dcoefs = (w_corner.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w_corner = w_corner * dcoefs.reshape(batch_size, -1, 1, 1, 1)

    x = x.reshape(batch_size, num_faces, *x.shape[1:]).permute((1, 0, 2, 3, 4)).reshape(num_faces, batch_size * in_channels, 1, -1)
    w_center = w_center.reshape(-1, in_channels, 1, 1)
    w_side = w_side.reshape(-1, in_channels, 1, 1)
    w_corner = w_corner.reshape(-1, in_channels, 1, 1)
    x_center = torch.nn.functional.conv2d(x[:, :, :, 0:1], w_center, groups=batch_size)
    x_side = torch.nn.functional.conv2d(x[:, :, :, [1, 3, 5, 7]], w_side, groups=batch_size)
    x_corner = torch.nn.functional.conv2d(x[:, :, :, [2, 4, 6, 8]], w_corner, groups=batch_size)
    x = aggregate_function(torch.cat([x_center, x_side, x_corner], dim=3))
    x = x.reshape(num_faces, batch_size * out_channels).reshape(num_faces, batch_size, out_channels).permute((1, 0, 2)).reshape((batch_size * num_faces, out_channels))
    return x


class SmoothUpsample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.blur = Blur()

    def forward(self, x, face_neighborhood, face_is_pad, pad_size, pool_map):
        x = unpool(x, pool_map)
        x = self.blur(x, face_neighborhood, face_is_pad, pad_size)
        return x


class Blur(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones((1, 1, 1, 9)).float())
        self.register_buffer("blur_filter", torch.tensor([1/4, 1/8, 1/16, 1/8, 1/16, 1/8, 1/16, 1/8, 1/16]).float())
        self.weight[:, :, 0, :] = self.blur_filter

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded_x[torch.logical_not(face_is_pad), :] = x
        f_ = [padded_x[face_neighborhood[:, i]] for i in range(9)]
        conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3).reshape((-1, 1, 1, 9))
        correction_factor = ((1 - face_is_pad[face_neighborhood].float()) * self.blur_filter.unsqueeze(0).expand(face_neighborhood.shape[0], -1)).sum(-1)
        correction_factor = correction_factor.unsqueeze(1).expand(-1, x.shape[1])
        return torch.nn.functional.conv2d(conv_input, self.weight).squeeze(-1).squeeze(-1).reshape(x.shape[0], x.shape[1]) / correction_factor


class FaceConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.neighborhood_size = 8
        self.conv = torch.nn.Conv2d(in_channels, out_channels, (1, self.neighborhood_size + 1), padding=0)

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        return self.conv(create_faceconv_input(x, self.neighborhood_size + 1, face_neighborhood, face_is_pad, pad_size)).squeeze(-1).squeeze(-1)


class FaceConvDemodulated(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kernel_size = 3
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, self.kernel_size ** 2]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        w = self.weight
        dcoefs = (w.square().sum(dim=[1, 2, 3]) + 1e-8).rsqrt()
        w = w * dcoefs.reshape(-1, 1, 1, 1)
        x = create_faceconv_input(x, self.kernel_size ** 2, face_neighborhood, face_is_pad, pad_size)
        x = torch.nn.functional.conv2d(x, w).squeeze(-1).squeeze(-1) + self.bias
        return x


class TextureConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, aggregation_func="mean"):
        super().__init__()
        self.conv_center = torch.nn.Conv2d(in_channels, out_channels, (1, 1), padding=0)
        self.conv_sides = torch.nn.Conv2d(in_channels, out_channels, (1, 1), padding=0)
        self.conv_corners = torch.nn.Conv2d(in_channels, out_channels, (1, 1), padding=0)
        self.aggregation_function = torch.nn.MaxPool2d(kernel_size=(1, 9)) if aggregation_func == 'max' else torch.nn.AvgPool2d(kernel_size=(1, 9))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        faceconv_input = create_faceconv_input(x, 9, face_neighborhood, face_is_pad, pad_size)
        center = faceconv_input[:, :, :, 0:1]
        sides = faceconv_input[:, :, :, [1, 3, 5, 7]]
        corners = faceconv_input[:, :, :, [2, 4, 6, 8]]
        center = self.conv_center(center)
        sides = self.conv_center(sides)
        corners = self.conv_center(corners)
        out = self.aggregation_function(torch.cat([center, sides, corners], dim=3))
        return out.squeeze(-1).squeeze(-1)


class SymmetricFaceConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_0 = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.weight_1 = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.weight_2 = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.empty((self.out_channels, self.in_channels, 1, 9)))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def create_symmetric_conv_filter(self):
        return torch.cat([self.weight_0, self.weight_1, self.weight_2,
                          self.weight_1, self.weight_2, self.weight_1,
                          self.weight_2, self.weight_1, self.weight_2], dim=-1)

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        conv_input = create_faceconv_input(x, self.neighborhood_size + 1, face_neighborhood, face_is_pad, pad_size)
        return torch.nn.functional.conv2d(conv_input, self.create_symmetric_conv_filter(), self.bias).squeeze(-1).squeeze(-1)


class GraphEncoder(torch.nn.Module):

    def __init__(self, in_channels, layer_dims=(32, 64, 64, 128, 128, 128, 256, 256), conv_layer=FaceConv, norm=torch_geometric.nn.BatchNorm):

        super().__init__()
        self.layer_dims = layer_dims
        self.activation = torch.nn.LeakyReLU()
        self.enc_conv_in = torch.nn.Linear(in_channels, layer_dims[0])
        self.down_0_block_0 = FResNetBlock(layer_dims[0], layer_dims[1], conv_layer, norm, self.activation)
        self.down_0_block_1 = FResNetBlock(layer_dims[1], layer_dims[2], conv_layer, norm, self.activation)  #
        self.down_1_block_0 = FResNetBlock(layer_dims[2], layer_dims[3], conv_layer, norm, self.activation)  #
        self.down_2_block_0 = FResNetBlock(layer_dims[3], layer_dims[4], conv_layer, norm, self.activation)  #
        self.down_3_block_0 = FResNetBlock(layer_dims[4], layer_dims[5], conv_layer, norm, self.activation)  #
        self.down_4_block_0 = FResNetBlock(layer_dims[5], layer_dims[6], conv_layer, norm, self.activation)  #
        self.enc_mid_block_0 = FResNetBlock(layer_dims[6], layer_dims[7], conv_layer, norm, self.activation)
        self.enc_out_conv = conv_layer(layer_dims[7], layer_dims[7])  #
        self.enc_out_norm = norm(layer_dims[7])

    def forward(self, x, graph_data):
        level_feats = []
        pool_ctr = 0
        x = self.enc_conv_in(x)
        x = self.down_0_block_0(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_0_block_1(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_1_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_2_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_3_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_4_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.enc_mid_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.enc_out_norm(x)
        x = self.activation(x)
        x = self.enc_out_conv(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        level_feats.append(x)

        return level_feats


class TwinGraphEncoder(torch.nn.Module):

    def __init__(self, in_channels_0, in_channels_1, layer_dims=(32, 64, 64, 128, 128, 128, 256, 256), conv_layer=FaceConv, norm=torch_geometric.nn.BatchNorm):

        super().__init__()
        self.layer_dims = layer_dims
        self.activation = torch.nn.LeakyReLU()
        self.enc0_conv_in = torch.nn.Linear(in_channels_0, layer_dims[0])
        self.enc1_conv_in = torch.nn.Linear(in_channels_1, layer_dims[0])
        self.down0_0_block_0 = FResNetBlock(layer_dims[0], layer_dims[1], conv_layer, norm, self.activation)
        self.down1_0_block_0 = FResNetBlock(layer_dims[0], layer_dims[1], conv_layer, norm, self.activation)
        self.down_0_block_1 = FResNetBlock(layer_dims[1] * 2, layer_dims[2], conv_layer, norm, self.activation)  #
        self.down_1_block_0 = FResNetBlock(layer_dims[2], layer_dims[3], conv_layer, norm, self.activation)  #
        self.down_2_block_0 = FResNetBlock(layer_dims[3], layer_dims[4], conv_layer, norm, self.activation)  #
        self.down_3_block_0 = FResNetBlock(layer_dims[4], layer_dims[5], conv_layer, norm, self.activation)  #
        self.down_4_block_0 = FResNetBlock(layer_dims[5], layer_dims[6], conv_layer, norm, self.activation)  #
        self.enc_mid_block_0 = FResNetBlock(layer_dims[6], layer_dims[7], conv_layer, norm, self.activation)
        self.enc_out_conv = conv_layer(layer_dims[7], layer_dims[7])  #
        self.enc_out_norm = norm(layer_dims[7])

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.layer_dims = layer_dims

    def forward(self, x_0, x_1, graph_data):
        level_feats = []
        pool_ctr = 0
        # print('x_0', x_0.shape)
        # print('weight', self.in_channels_0, self.layer_dims[0])
        x_0 = self.enc0_conv_in(x_0)
        x_1 = self.enc1_conv_in(x_1)
        x_0 = self.down0_0_block_0(x_0, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x_1 = self.down1_0_block_0(x_1, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_0_block_1(torch.cat([x_0, x_1], dim=1), graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_1_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_2_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_3_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_4_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], graph_data['lateral_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.enc_mid_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.enc_out_norm(x)
        x = self.activation(x)
        x = self.enc_out_conv(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        level_feats.append(x)

        return level_feats


class FResNetBlock(torch.nn.Module):

    def __init__(self, nf_in, nf_out, conv_layer, norm, activation):
        super().__init__()
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.norm_0 = norm(nf_in)
        self.conv_0 = conv_layer(nf_in, nf_out)
        self.norm_1 = norm(nf_out)
        self.conv_1 = conv_layer(nf_out, nf_out)
        self.activation = activation
        if nf_in != nf_out:
            self.nin_shortcut = conv_layer(nf_in, nf_out)

    def forward(self, x, face_neighborhood, face_is_pad, num_pad):
        h = x
        h = self.norm_0(h)
        h = self.activation(h)
        h = self.conv_0(h, face_neighborhood, face_is_pad, num_pad)

        h = self.norm_1(h)
        h = self.activation(h)
        h = self.conv_1(h, face_neighborhood, face_is_pad, num_pad)

        if self.nf_in != self.nf_out:
            x = self.nin_shortcut(x, face_neighborhood, face_is_pad, num_pad)

        return x + h


class GNNChildDecoder(torch.nn.Module):

    def __init__(self, node_feat_size, hidden_size, max_child_num,
                 edge_symmetric_type='max', num_iterations=3, edge_type_num=1):
        super(GNNChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size

        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_edge_latent = nn.Linear(hidden_size * 2, hidden_size)

        self.mlp_edge_exists = nn.ModuleList()
        for i in range(self.edge_type_num):
            self.mlp_edge_exists.append(nn.Linear(hidden_size, 1))

        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size * 3 + self.edge_type_num, hidden_size))

        self.mlp_child = nn.Linear(hidden_size * (self.num_iterations + 1), hidden_size)
        self.mlp_child2 = nn.Linear(hidden_size, node_feat_size)

    def forward(self, child_feats):

        flag_4_dims = False
        if len(child_feats.shape) == 4:
            flag_4_dims = True
            batch_size, num_classes, aux_size, feature_size = child_feats.shape
            child_feats = child_feats.reshape(batch_size, num_classes, aux_size * feature_size)
        else:
            batch_size, num_classes, feature_size = child_feats.shape

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size * self.max_child_num, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # edge features
        edge_latents = torch.cat([
            child_feats.view(batch_size, self.max_child_num, 1, self.hidden_size).expand(-1, -1, self.max_child_num, -1),
            child_feats.view(batch_size, 1, self.max_child_num, self.hidden_size).expand(-1, self.max_child_num, -1, -1)
        ], dim=3)
        edge_latents = torch.relu(self.mlp_edge_latent(edge_latents))

        # edge existence prediction
        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(
                batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)

        """
            decoding stage message passing
            there are several possible versions, this is a simple one:
            use a fixed set of edges, consisting of existing edges connecting existing nodes
            this set of edges does not change during iterations
            iteratively update the child latent features
            then use these child latent features to compute child features and semantics
        """
        # get edges that exist between nodes that exist
        edge_indices = torch.nonzero(edge_exists_logits > 0)
        edge_types = edge_indices[:, 3]
        edge_indices = edge_indices[:, 1:3]
        nodes_exist_mask = (child_exists_logits[0, edge_indices[:, 0], 0] > 0) \
                           & (child_exists_logits[0, edge_indices[:, 1], 0] > 0)
        edge_indices = edge_indices[nodes_exist_mask, :]
        edge_types = edge_types[nodes_exist_mask]

        # get latent features for the edges
        edge_feats_mp = edge_latents[0:1, edge_indices[:, 0], edge_indices[:, 1], :]

        # append edge type to edge features, so the network has information which
        # of the possibly multiple edges between two nodes it is working with
        edge_type_logit = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], :]
        edge_type_logit = edge_feats_mp.new_zeros(edge_feats_mp.shape[:2] + (self.edge_type_num,))
        edge_type_logit[0:1, range(edge_type_logit.shape[1]), edge_types] = \
            edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], edge_types]
        edge_feats_mp = torch.cat([edge_feats_mp, edge_type_logit], dim=2)

        num_edges = edge_indices.shape[0]
        max_childs = child_feats.shape[1]

        iter_child_feats = [child_feats]  # zeroth iteration

        if self.num_iterations > 0 and num_edges > 0:
            edge_indices_from = edge_indices[:, 0].view(-1, 1).expand(-1, self.hidden_size)

        for i in range(self.num_iterations):
            if num_edges > 0:
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[:, 0], :],  # start node features
                    child_feats[0:1, edge_indices[:, 1], :],  # end node features
                    edge_feats_mp], dim=2)  # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, self.hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0,
                                                                   out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0,
                                                                out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0,
                                                                 out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, self.hidden_size)

            # save child features of this iteration
            iter_child_feats.append(child_feats)

        # concatenation of the child features from all iterations (as in GIN, like skip connections)
        child_feats = torch.cat(iter_child_feats, dim=2)

        # transform concatenation back to original feature space size
        child_feats = child_feats.view(-1, self.hidden_size * (self.num_iterations + 1))
        child_feats = torch.relu(self.mlp_child(child_feats))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.hidden_size)

        # node features
        child_feats = self.mlp_child2(child_feats.view(-1, self.hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.hidden_size)
        child_feats = torch.relu(child_feats)

        if flag_4_dims:
            child_feats = child_feats.reshape(batch_size, num_classes, aux_size, feature_size)

        return child_feats


class PointPartClassifierEntropyPointNet(nn.Module):

    def __init__(self, feature_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super(PointPartClassifierEntropyPointNet, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size_1)
        self.mlp2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp3 = nn.Linear(hidden_size_2, hidden_size_3)

    def forward(self, input):
        feat1 = torch.relu(self.mlp1(input))
        feat2 = torch.relu(self.mlp2(feat1))
        feat3 = torch.relu(self.mlp3(feat2))
        global_feat = F.max_pool1d(feat3.T[None, ...], kernel_size=feat3.size()[0])[..., 0]

        return global_feat
