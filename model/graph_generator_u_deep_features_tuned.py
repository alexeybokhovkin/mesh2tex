import torch
import numpy as np
from model import activation_funcs, FullyConnectedLayer, clamp_gain, normalize_2nd_moment
from model.graph import create_faceconv_input, SmoothUpsample, modulated_face_conv


class Generator(torch.nn.Module):

    def __init__(self, z_dim, w_dim, w_num_layers, num_faces, color_channels, e_layer_dims, c_dim=0, channel_base=16384, channel_max=512):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.num_faces = num_faces
        self.color_channels = color_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, num_faces=num_faces, color_channels=color_channels, e_layer_dims=e_layer_dims, channel_base=channel_base, channel_max=channel_max)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, c_dim=c_dim, num_ws=self.num_ws, num_layers=w_num_layers)

    def forward(self, graph_data, z, shape, c=None, truncation_psi=1, truncation_cutoff=None, noise_mode='random'):
        ws, z_out = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img, all_face_colors, feature, tuned_feature = self.synthesis(graph_data, ws, shape, noise_mode)
        return img, all_face_colors, feature, tuned_feature, z_out


class SynthesisNetwork(torch.nn.Module):

    def __init__(self, w_dim, num_faces, color_channels, e_layer_dims, channel_base=16384, channel_max=512):
        super().__init__()
        self.w_dim = w_dim
        self.num_faces = num_faces
        self.color_channels = color_channels
        self.offset = 6 - len(self.num_faces)
        self.block_pow_2 = [2 ** i for i in range(2, len(self.num_faces) + 2)] # 6 / len(self.num_faces)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_pow_2}
        channels_dict_geo = {4: e_layer_dims[-1], 8: e_layer_dims[-2], 16: e_layer_dims[-3], 32: e_layer_dims[-4], 64: e_layer_dims[-5], 128: e_layer_dims[-6]}
        self.blocks = torch.nn.ModuleList()
        block_level = len(self.block_pow_2) - 1
        self.num_ws = 2 * (len(self.block_pow_2) + 1) + 3
        self.first_block = SynthesisPrologue(channels_dict[self.block_pow_2[0]], w_dim=w_dim, geo_channels=channels_dict_geo[4], num_face=num_faces[block_level], color_channels=color_channels)
        for blk_idx, cdict_key in enumerate(self.block_pow_2[1:]):
            block_level -= 1
            in_channels = channels_dict[cdict_key // 2]
            geo_channels = channels_dict_geo[cdict_key]
            out_channels = channels_dict[cdict_key]
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, geo_channels=geo_channels,
                                   num_face=num_faces[block_level],
                                   color_channels=color_channels)
            self.blocks.append(block)
        self.last_block = SynthesisEpilogue(channels_dict[128], channels_dict[128], w_dim=w_dim, num_face=num_faces[0], color_channels=color_channels)

        # as a part of differentiable renderer
        self.tuner = SynthesisTuner(channels_dict[128], channels_dict[128], w_dim=w_dim, num_face=num_faces[0], color_channels=color_channels)

    def forward(self, graph_data, ws, shape, noise_mode='random'):
        split_ws = [ws[:, 0:2, :]] + [ws[:, 2 * n + 1: 2 * n + 4, :] for n in range(len(self.block_pow_2))]
        ws_tune = ws[:, -3:, :]
        neighborhoods = [graph_data['face_neighborhood']] + graph_data['sub_neighborhoods']
        appended_pool_maps = [None] + graph_data['pool_maps']
        block_level = len(self.num_faces) - 1

        all_face_colors = []

        x, face_colors = self.first_block(neighborhoods[block_level],
                                          graph_data['is_pad'][block_level],
                                          graph_data['pads'][block_level],
                                          appended_pool_maps[block_level],
                                          split_ws[0], noise_mode)

        all_face_colors += [face_colors]

        for i in range(len(self.num_faces) - 1):
            block_level -= 1
            sub_neighborhoods, is_pad, pads, pool_maps = [neighborhoods[block_level + 1], neighborhoods[block_level]], \
                                                         [graph_data['is_pad'][block_level + 1], graph_data['is_pad'][block_level]], \
                                                         [graph_data['pads'][block_level + 1], graph_data['pads'][block_level]], \
                                                         [appended_pool_maps[block_level + 1], appended_pool_maps[block_level]]
            x, face_colors = self.blocks[i](sub_neighborhoods, is_pad, pads, pool_maps, x, face_colors, split_ws[i + 1], shape[len(shape) - 1 - i - self.offset], noise_mode)
            all_face_colors += [face_colors]
        all_face_colors = all_face_colors[:-1]

        feature, face_colors = self.last_block(neighborhoods[0], graph_data['is_pad'][0], graph_data['pads'][0], appended_pool_maps[0], x, face_colors, split_ws[-1], shape[0], noise_mode)
        all_face_colors += [face_colors]
        all_face_colors = all_face_colors[::-1]

        tuned_feature = self.tuner(neighborhoods[0], graph_data['is_pad'][0], graph_data['pads'][0], appended_pool_maps[0], x, ws_tune, shape[0], noise_mode)
        return face_colors, all_face_colors, feature, tuned_feature


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, geo_channels, num_face, color_channels):
        super().__init__()
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.const = torch.nn.Parameter(torch.randn([num_face, out_channels]))
        self.conv1 = SynthesisLayer(out_channels, out_channels - geo_channels, w_dim=w_dim, num_face=num_face)
        self.torgb = ToRGBLayer(out_channels - geo_channels, color_channels, w_dim=w_dim, num_face=num_face)

    def forward(self, face_neighborhood, face_is_pad, pad_size, pool_map, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1]).reshape((self.num_face * ws.shape[0], -1))
        x = self.conv1(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, face_neighborhood, face_is_pad, pad_size, next(w_iter))
        return x, img


class SynthesisBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, geo_channels, num_face, color_channels):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, num_face=num_face, resampler=self.resampler)
        self.conv1 = SynthesisLayer(out_channels, out_channels - geo_channels, w_dim=w_dim, num_face=num_face)
        self.torgb = ToRGBLayer(out_channels - geo_channels, color_channels, w_dim=w_dim, num_face=num_face)

    def forward(self, face_neighborhood, face_is_pad, pad_size, pool_map, x, img, ws, shape, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = torch.cat([x, shape], dim=1)
        x = self.conv0(x, face_neighborhood, face_is_pad, pad_size, pool_map[0], next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, face_neighborhood[1:], face_is_pad[1:], pad_size[1:], pool_map[1], next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, face_neighborhood[1], face_is_pad[1], pad_size[1], next(w_iter))
        img = self.resampler(img, face_neighborhood[1], face_is_pad[1], pad_size[1], pool_map[0])
        img = img.add_(y)

        return x, img


class SynthesisEpilogue(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, color_channels):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, num_face=num_face)
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, num_face=num_face)
        self.torgb = ToRGBLayer(out_channels, color_channels, w_dim=w_dim, num_face=num_face)

    def forward(self, face_neighborhood, face_is_pad, pad_size, pool_map, x, img, ws, shape, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = torch.cat([x, shape], dim=1)
        x = self.conv0(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, face_neighborhood, face_is_pad, pad_size, next(w_iter))
        img = img.add_(y)
        return x, img


class SynthesisTuner(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, color_channels):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, num_face=num_face)
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, num_face=num_face)
        # self.conv2 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, num_face=num_face)

    def forward(self, face_neighborhood, face_is_pad, pad_size, pool_map, x, ws, shape, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = torch.cat([x, shape], dim=1)
        x = self.conv0(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)
        # x = self.conv2(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)

        return x


class ToRGBLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_face = num_face
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, kernel_size ** 2]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size, w):
        styles = self.affine(w) * self.weight_gain
        x = create_faceconv_input(x, self.kernel_size ** 2, face_neighborhood, face_is_pad, pad_size)
        x = modulated_face_conv(x=x, weight=self.weight, styles=styles, demodulate=False)
        return torch.clamp(x + self.bias[None, :], -256, 256)


class SynthesisLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, kernel_size=3, resampler=None, activation='lrelu', noise_opt=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_face = num_face
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1,  kernel_size ** 2]))

        self.register_buffer('noise_const', torch.randn([num_face, ]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        if noise_opt:
            self.noise_opt = torch.nn.Parameter(torch.randn([styles.shape[0] * self.num_face, 1]))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size, pool_map, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([styles.shape[0] * self.num_face, 1], device=x.device) * self.noise_strength
        elif noise_mode == 'const':
            noise = self.noise_const.unsqueeze(0).repeat([styles.shape[0], 1]).reshape([styles.shape[0] * self.num_face, 1]) * self.noise_strength

        x = create_faceconv_input(x, self.kernel_size ** 2, face_neighborhood[0], face_is_pad[0], pad_size[0])
        x = modulated_face_conv(x=x, weight=self.weight, styles=styles)
        if self.resampler is not None:
            x = self.resampler(x, face_neighborhood[1], face_is_pad[1], pad_size[1], pool_map)

        x = x + noise

        return clamp_gain(self.activation(x + self.bias[None, :]), self.activation_gain * gain, 256 * gain)


class MappingNetwork(torch.nn.Module):

    def __init__(self, z_dim, w_dim, c_dim, num_ws, num_layers=8, activation='lrelu', lr_multiplier=0.01, w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        features_list = [z_dim] + [w_dim] * num_layers

        if c_dim > 0:
            features_list[0] += w_dim
            self.embed = FullyConnectedLayer(c_dim, w_dim)

        self.layers = torch.nn.ModuleList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            self.layers.append(FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier))

        self.out_layers = torch.nn.ModuleList()
        in_features = w_dim
        out_features = z_dim
        self.out_layers.append(FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier))
        for idx in range(2):
            in_features = z_dim
            out_features = z_dim
            self.out_layers.append(FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier))

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = normalize_2nd_moment(z)

        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c))
            x = torch.cat([x, y], dim=1)

        # Main layers.
        for idx in range(self.num_layers):
            x = self.layers[idx](x)

        # Extra layers.
        x_out = x
        for idx in range(3):
            x_out = self.out_layers[idx](x_out)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x, x_out
