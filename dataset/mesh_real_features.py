import json
import os
import random
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_scatter
import torchvision.transforms as T
import trimesh
from torchvision.io import read_image
from tqdm import tqdm
from skimage import color

from util.camera import spherical_coord_to_cam
from util.misc import EasyDict


class FaceGraphMeshDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None, single_mode=False, dump_data=False):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.image_size = config.image_size
        self.random_views = config.random_views
        self.camera_noise = config.camera_noise
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        # self.real_images_buf = {}
        # for i, key in enumerate(self.real_images):
        #     if i < 2048:
        #         self.real_images_buf[key] = self.real_images[key]
        # self.real_images = self.real_images_buf
        self.masks = {x: Path(config.mask_path) / self.real_images[x].name for x in self.real_images}
        self.erode = config.erode
        if not single_mode:
            self.items = list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size]
        else:
            # real textures
            # config.shape_id = 'shape02880_rank01_pair1'
            # config.shape_id = 'shape06567_rank01_pair7492'
            # config.shape_id = 'shape05021_rank02_pair143369'
            # config.shape_id = 'shape03129_rank02_pair211294'

            # synthetic textures
            # config.shape_id = 'shape02338_rank02_pair162577'
            config.shape_id = 'shape02329_rank02_pair125300'
            # config.shape_id = 'shape02344_rank02_pair183269'
            # config.shape_id = 'shape02325_rank01_pair190635'
            # config.shape_id = 'shape02332_rank00_pair296609'
            self.items = [config.shape_id]
            if limit_dataset_size is None:
                self.items = self.items * config.epoch_steps
            else:
                self.items = self.items * limit_dataset_size

        self.target_name = "model_normalized.obj"
        self.views_per_sample = config.views_per_sample
        if config.random_bg == 'color':
            self.color_generator = random_color
        elif config.random_bg == 'grayscale':
            self.color_generator = random_grayscale
        else:
            self.color_generator = white
        self.random_bg = config.random_bg
        self.predefined_image_views = False

        self.cspace_convert = rgb_to_lab if config.colorspace == 'lab' else (lambda x: x)
        self.cspace_convert_back = lab_to_rgb if config.colorspace == 'lab' else (lambda x: x)
        self.input_feature_extractor, self.num_feats = {
            "normal": (self.input_normal, 3),
            "position": (self.input_position, 3),
            "position+normal": (self.input_position_normal, 6),
            "normal+laplacian": (self.input_normal_laplacian, 6),
            "normal+ff1+ff2": (self.input_normal_ff1_ff2, 15),
            "ff2": (self.input_ff2, 8),
            "curvature": (self.input_curvature, 2),
            "laplacian": (self.input_laplacian, 3),
            "normal+curvature": (self.input_normal_curvature, 5),
            "normal+laplacian+ff1+ff2+curvature": (self.input_normal_laplacian_ff1_ff2_curvature, 20),
            "semantics": (self.input_semantics, 7),
            "normal+semantics": (self.input_normal_semantics, 10),
            "normal+curvature+semantics": (self.input_normal_curvature_semantics, 12),
        }[config.features]
        self.pair_meta, self.all_views, self.all_images = self.load_pair_meta(config.pairmeta_path)
        config.shapes_path = '../metadata/shapes.json'
        self.shapenet_shapes = json.loads(Path(config.shapes_path).read_text())
        self.shapenet_to_shapeid = defaultdict(list)
        with open('../metadata/chair_shapenet_to_shapeid.json', 'r') as fin:
            self.shapenet_to_shapeid = json.load(fin)
        self.shapeids_to_items = defaultdict(list)
        for item in self.items:
            self.shapeids_to_items[str(int(item.split('_')[0][5:]))].append(item)
        self.real_images_preloaded, self.masks_preloaded = {}, {}
        self.stats = torch.load(config.stat_path)
        if config.preload:
            self.preload_real_images()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        pt_arxiv = torch.load(os.path.join(self.dataset_directory, f'{selected_item}.pt'))
        vertices = pt_arxiv['vertices'].float()
        indices = pt_arxiv['indices'].int()
        normals = pt_arxiv['normals'].float()

        # conv_data_key = 'conv_data_semantic'
        conv_data_key = 'conv_data'
        edge_index = pt_arxiv[conv_data_key][0][0].long()
        num_sub_vertices = [pt_arxiv[conv_data_key][i][0].shape[0] for i in range(1, len(pt_arxiv[conv_data_key]))]
        pad_sizes = [pt_arxiv[conv_data_key][i][2].shape[0] for i in range(len(pt_arxiv[conv_data_key]))]
        sub_edges = [pt_arxiv[conv_data_key][i][0].long() for i in range(1, len(pt_arxiv[conv_data_key]))]
        ff2_maps = [x.float() / math.pi for x in pt_arxiv['ff2_data']]
        ff2_maps = [((torch.norm(x, dim=1) / math.sqrt(8)) ** 4 * 2 - 1).unsqueeze(-1) for x in ff2_maps]

        # pool_locations_key = 'pool_locations_semantic_2'
        pool_locations_key = 'pool_locations'
        pool_maps = pt_arxiv[pool_locations_key]
        lateral_maps = pt_arxiv['lateral_maps']
        is_pad = [pt_arxiv[conv_data_key][i][4].bool() for i in range(len(pt_arxiv[conv_data_key]))]
        level_masks = [torch.zeros(pt_arxiv[conv_data_key][i][0].shape[0]).long() for i in range(len(pt_arxiv[conv_data_key]))]

        # noinspection PyTypeChecker
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()

        img_name = random.sample(list(self.real_images.keys()), 1)[0]
        # img_names = [f'{x}-scannet' for x in range(3895, 3914)]
        # img_name = '3912-scannet' # [3895 - 3913]
        # img_names = [f'{x}-scannet' for x in range(6695, 6706)]
        # img_name = '6700-scannet' # [6695 - 6705]
        # img_name = random.sample(img_names, 1)[0]
        real_sample, real_mask, mvp, cam_positions = self.get_image_and_view(selected_item) # selected_item / img_name
        background = self.color_generator(self.views_per_sample)
        background = self.cspace_convert(background)

        shape_id = str(int(selected_item.split('_')[0].split('shape')[1]))
        shapenet_id = self.shapenet_shapes[shape_id]["source_id"]

        vertices_maps = pt_arxiv['vertices_maps']
        indices_maps = pt_arxiv['indices_maps']
        tri_indices_maps = [torch.cat([indices_maps[level][:, [0, 1, 2]], indices_maps[level][:, [0, 2, 3]]], 0) for level in range(len(indices_maps))]
        ranges_maps = [torch.tensor([0, tri_indices_maps[level].shape[0]]).int() for level in range(len(tri_indices_maps))]
        vertex_ctr_maps = [torch.tensor(list(range(vertices_maps[level].shape[0]))).long() for level in range(len(vertices_maps))]
        vertices = vertices - vertices.mean(dim=0)

        return {
            "name": selected_item, # name of file (sample)
            "x": self.input_feature_extractor(pt_arxiv), # additional quadmesh per-face feature
            "y": self.cspace_convert(ff2_maps[0].reshape(-1, 1).repeat(1, 3).float()), # ff2 feature of quadmesh
            "vertex_ctr": vctr, # vertex indices
            "vertices": vertices, # quadmesh vertices
            "normals": normals, # quadmesh face normals
            "indices_quad": indices, # quadmesh face indices for vertices which form a face
            "mvp": mvp, # camera matrices
            "cam_position": cam_positions, # camera positions
            "real": real_sample, # real image of a shape
            "mask": real_mask, # corresponding mask for a shape on an image
            "bg": torch.cat([background, torch.ones([background.shape[0], 1, 1, 1])], dim=1), # background color
            "indices": tri_indices, # quadmesh face indices but every face is split into two triangles
            "ranges": torch.tensor([0, tri_indices.shape[0]]).int(), # range numbers for number of triangles
            "graph_data": self.get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, ff2_maps, lateral_maps, is_pad, level_masks),
            "vertices_maps": vertices_maps,
            "indices_maps": indices_maps,
            "tri_indices_maps": tri_indices_maps,
            "ranges_maps": ranges_maps,
            "vertex_ctr_maps": vertex_ctr_maps,
            "shapenet_id": shapenet_id
        }

    @staticmethod
    def get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, ff2_maps, lateral_maps, is_pad, level_masks):
        return EasyDict({
            'face_neighborhood': edge_index,
            'sub_neighborhoods': sub_edges,
            'pads': pad_sizes,
            'ff2_maps': ff2_maps,
            'node_counts': num_sub_vertices,
            'pool_maps': pool_maps,
            'lateral_maps': lateral_maps,
            'is_pad': is_pad,
            'level_masks': level_masks
        })

    @staticmethod
    def batch_mask(t, graph_data, idx, level=0):
        return t[graph_data['level_masks'][level] == idx]

    def get_image_and_view(self, shape):
        shape_id = int(shape.split('_')[0].split('shape')[1])
        if self.random_views:
            sampled_view = get_random_views(self.views_per_sample)
        else:
            sampled_view = random.sample(self.all_views, self.views_per_sample)

        image_selections = self.get_image_selections(shape_id)

        images, masks, cameras, cam_positions = [], [], [], []
        for c_i, c_v in zip(image_selections, sampled_view):
            images.append(self.get_real_image(self.meta_to_pair(c_i)))
            masks.append(self.get_real_mask(self.meta_to_pair(c_i)))

            if self.predefined_image_views:
                c_v = c_i

            # nvdiffrast rendering
            # azimuth = c_v['azimuth'] + (random.random() - 0.5) * self.camera_noise
            # elevation = c_v['elevation'] + (random.random() - 0.5) * self.camera_noise

            # pytorch3d rendering
            azimuth = c_v['azimuth'] + (random.random() - 0.5) * self.camera_noise
            elevation = c_v['elevation'] + (random.random() - 0.5) * self.camera_noise

            # pytorch3d rendering (OLD)
            # azimuth = c_v['azimuth'] + np.pi + (random.random() - 0.5) * self.camera_noise
            # elevation = c_v['elevation'] + 0.5 + (random.random() - 0.5) * self.camera_noise

            if self.random_views: # MODIFIED TO SUPPORT OUTSOURCE IMAGES
                perspective_cam = spherical_coord_to_cam(50.0, azimuth, elevation, cam_dist=2.5) # cam_dist=2.5 / c_v['distance'] * 0.35
            else:
                perspective_cam = spherical_coord_to_cam(c_i['fov'], azimuth, elevation, cam_dist=2.5)
            # projection_matrix = intrinsic_to_projection(get_default_perspective_cam()).float()
            projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
            # view_matrix = torch.from_numpy(np.linalg.inv(generate_camera(np.zeros(3), c['azimuth'], c['elevation']))).float()
            cam_position = torch.from_numpy(np.linalg.inv(perspective_cam.view_mat())[:3, 3]).float()
            view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
            cameras.append(torch.matmul(projection_matrix, view_matrix))
            cam_positions.append(cam_position)

        image = torch.cat(images, dim=0)
        mask = torch.cat(masks, dim=0)
        mvp = torch.stack(cameras, dim=0)
        cam_positions = torch.stack(cam_positions, dim=0)
        return image, mask, mvp, cam_positions

    def get_by_shapenet_id(self, shapenet_id):
        shapeid_candidates = self.shapenet_to_shapeid[shapenet_id]
        shapeid_candidate = random.sample(shapeid_candidates, 1)[0]
        items_candidates = self.shapeids_to_items[shapeid_candidate]
        selected_item = random.sample(items_candidates, 1)[0]

        pt_arxiv = torch.load(os.path.join(self.dataset_directory, f'{selected_item}.pt'))
        vertices = pt_arxiv['vertices'].float()
        indices = pt_arxiv['indices'].int()
        normals = pt_arxiv['normals'].float()

        conv_data_key = 'conv_data'
        edge_index = pt_arxiv[conv_data_key][0][0].long()
        num_sub_vertices = [pt_arxiv[conv_data_key][i][0].shape[0] for i in range(1, len(pt_arxiv[conv_data_key]))]
        pad_sizes = [pt_arxiv[conv_data_key][i][2].shape[0] for i in range(len(pt_arxiv[conv_data_key]))]
        sub_edges = [pt_arxiv[conv_data_key][i][0].long() for i in range(1, len(pt_arxiv[conv_data_key]))]
        ff2_maps = [x.float() / math.pi for x in pt_arxiv['ff2_data']]
        ff2_maps = [((torch.norm(x, dim=1) / math.sqrt(8)) ** 4 * 2 - 1).unsqueeze(-1) for x in ff2_maps]

        pool_locations_key = 'pool_locations'
        pool_maps = pt_arxiv[pool_locations_key]
        lateral_maps = pt_arxiv['lateral_maps']
        is_pad = [pt_arxiv[conv_data_key][i][4].bool() for i in range(len(pt_arxiv[conv_data_key]))]
        level_masks = [torch.zeros(pt_arxiv[conv_data_key][i][0].shape[0]).long() for i in range(len(pt_arxiv[conv_data_key]))]

        # noinspection PyTypeChecker
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()

        real_sample, real_mask, mvp, cam_positions = self.get_image_and_view(selected_item) # selected_item / img_name
        background = self.color_generator(self.views_per_sample)
        background = self.cspace_convert(background)

        shape_id = str(int(selected_item.split('_')[0].split('shape')[1]))
        shapenet_id = self.shapenet_shapes[shape_id]["source_id"]

        vertices_maps = pt_arxiv['vertices_maps']
        indices_maps = pt_arxiv['indices_maps']
        tri_indices_maps = [torch.cat([indices_maps[level][:, [0, 1, 2]], indices_maps[level][:, [0, 2, 3]]], 0) for level in range(len(indices_maps))]
        ranges_maps = [torch.tensor([0, tri_indices_maps[level].shape[0]]).int() for level in range(len(tri_indices_maps))]
        vertex_ctr_maps = [torch.tensor(list(range(vertices_maps[level].shape[0]))).long() for level in range(len(vertices_maps))]
        vertices = vertices - vertices.mean(dim=0)

        return {
            "name": selected_item, # name of file (sample)
            "x": self.input_feature_extractor(pt_arxiv), # additional quadmesh per-face feature
            "y": self.cspace_convert(ff2_maps[0].reshape(-1, 1).repeat(1, 3).float()), # ff2 feature of quadmesh
            "vertex_ctr": vctr, # vertex indices
            "vertices": vertices, # quadmesh vertices
            "normals": normals, # quadmesh face normals
            "indices_quad": indices, # quadmesh face indices for vertices which form a face
            "mvp": mvp, # camera matrices
            "cam_position": cam_positions, # camera positions
            "real": real_sample, # real image of a shape
            "mask": real_mask, # corresponding mask for a shape on an image
            "bg": torch.cat([background, torch.ones([background.shape[0], 1, 1, 1])], dim=1), # background color
            "indices": tri_indices, # quadmesh face indices but every face is split into two triangles
            "ranges": torch.tensor([0, tri_indices.shape[0]]).int(), # range numbers for number of triangles
            "graph_data": self.get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, ff2_maps, lateral_maps, is_pad, level_masks),
            "vertices_maps": vertices_maps,
            "indices_maps": indices_maps,
            "tri_indices_maps": tri_indices_maps,
            "ranges_maps": ranges_maps,
            "vertex_ctr_maps": vertex_ctr_maps,
            "shapenet_id": shapenet_id
        }

    def get_real_image(self, name):
        if name not in self.real_images_preloaded.keys():
            return self.process_real_image(self.real_images[name])
        else:
            return self.real_images_preloaded[name]

    def get_real_mask(self, name):
        if name not in self.masks_preloaded.keys():
            return self.process_real_mask(self.masks[name])
        else:
            return self.masks_preloaded[name]

    def get_image_selections(self, shape_id):
        candidates = self.pair_meta[shape_id]
        if len(candidates) < self.views_per_sample:
            while len(candidates) < self.views_per_sample:
                meta = self.pair_meta[random.choice(list(self.pair_meta.keys()))]
                candidates.extend(meta[:self.views_per_sample - len(candidates)])
        else:
            candidates = random.sample(candidates, self.views_per_sample)
        return candidates

    def process_real_image(self, path):
        resize = T.Resize(size=(self.image_size, self.image_size))
        pad = T.Pad(padding=(100, 100), fill=1)
        t_image = resize(pad(read_image(str(path)).float() / 127.5 - 1))
        return self.cspace_convert(t_image.unsqueeze(0))

    def process_scannet_image(self, path):
        resize = T.Resize(size=(self.image_size, self.image_size))
        t_image = resize(read_image(str(path)).float() / 127.5 - 1)
        return self.cspace_convert(t_image.unsqueeze(0))

    def export_mesh(self, name, face_colors, output_path):
        try:
            mesh = trimesh.load(self.mesh_directory / name / self.target_name, process=False)
            vertex_colors = torch.zeros(mesh.vertices.shape).to(face_colors.device)
            torch_scatter.scatter_mean(face_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3),
                                       torch.from_numpy(mesh.faces).to(face_colors.device).reshape(-1).long(), dim=0, out=vertex_colors)
            vertex_colors = torch.clamp(vertex_colors, 0, 1)
            out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
            out_mesh.export(output_path)
        except Exception as err:
            print("Failed exporting mesh", err)

    @staticmethod
    def erode_mask(mask):
        import cv2 as cv
        mask = mask.squeeze(0).numpy().astype(np.uint8)
        kernel_size = 3
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    def process_real_mask(self, path):
        resize = T.Resize(size=(self.image_size, self.image_size))
        pad = T.Pad(padding=(100, 100), fill=0)
        if self.erode:
            eroded_mask = self.erode_mask(read_image(str(path)))
        else:
            eroded_mask = read_image(str(path))
        t_mask = resize(pad((eroded_mask > 0).float()))
        return t_mask.unsqueeze(0)

    def load_pair_meta(self, pairmeta_path):
        loaded_json = json.loads(Path(pairmeta_path).read_text())
        ret_dict = defaultdict(list)
        ret_views = []
        ret_images = {}
        for k in loaded_json.keys():
            if self.meta_to_pair(loaded_json[k]) in self.real_images.keys():
                ret_dict[loaded_json[k]['shape_id']].append(loaded_json[k])
                ret_views.append(loaded_json[k])
                ret_images[k] = loaded_json[k]
        return ret_dict, ret_views, ret_images

    def preload_real_images(self):
        for ri in tqdm(self.real_images.keys(), desc='preload'):
            self.real_images_preloaded[ri] = self.process_real_image(self.real_images[ri])
            self.masks_preloaded[ri] = self.process_real_mask(self.masks[ri])

    @staticmethod
    def meta_to_pair(c):
        return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'

    @staticmethod
    def get_color_bg_real(batch, highres=False):
        if not highres:
            real_sample = batch['real'] * batch['mask'].expand(-1, 3, -1, -1) + (1 - batch['mask']).expand(-1, 3, -1, -1) * batch['bg'][:, :3, :, :]
        else:
            real_sample = batch['real2'] * batch['mask2'].expand(-1, 3, -1, -1) + (1 - batch['mask2']).expand(-1, 3, -1, -1) * batch['bg'][:, :3, :, :]
        return real_sample

    @staticmethod
    def input_position(pt_arxiv):
        return pt_arxiv['input_positions']

    @staticmethod
    def input_normal(pt_arxiv):
        return pt_arxiv['input_normals']

    @staticmethod
    def input_position_normal(pt_arxiv):
        return torch.cat([pt_arxiv['input_positions'], pt_arxiv['input_normals']], dim=1)

    @staticmethod
    def input_normal_laplacian(pt_arxiv):
        return torch.cat([pt_arxiv['input_normals'], pt_arxiv['input_laplacian']], dim=1)

    def input_normal_semantics(self, pt_arxiv):
        return torch.cat([pt_arxiv['input_normals'], self.input_semantics(pt_arxiv)], dim=1)

    def input_normal_ff1_ff2(self, pt_arxiv):
        return torch.cat([pt_arxiv['input_normals'], self.normed_feat(pt_arxiv, 'input_ff1'), self.normed_feat(pt_arxiv, 'input_ff2')], dim=1)

    def input_ff2(self, pt_arxiv):
        return self.normed_feat(pt_arxiv, 'input_ff2')

    def input_curvature(self, pt_arxiv):
        return torch.cat([self.normed_feat(pt_arxiv, 'input_gcurv').unsqueeze(-1), self.normed_feat(pt_arxiv, 'input_mcurv').unsqueeze(-1)], dim=1)

    @staticmethod
    def input_laplacian(pt_arxiv):
        return pt_arxiv['input_laplacian']

    def input_normal_curvature(self, pt_arxiv):
        return torch.cat([pt_arxiv['input_normals'], self.normed_feat(pt_arxiv, 'input_gcurv').unsqueeze(-1), self.normed_feat(pt_arxiv, 'input_mcurv').unsqueeze(-1)], dim=1)

    def input_normal_laplacian_ff1_ff2_curvature(self, pt_arxiv):
        return torch.cat([pt_arxiv['input_normals'], pt_arxiv['input_laplacian'], self.normed_feat(pt_arxiv, 'input_ff1'),
                          self.normed_feat(pt_arxiv, 'input_ff2'), self.normed_feat(pt_arxiv, 'input_gcurv').unsqueeze(-1),
                          self.normed_feat(pt_arxiv, 'input_mcurv').unsqueeze(-1)], dim=1)

    def input_semantics(self, pt_arxiv):
        return torch.nn.functional.one_hot(pt_arxiv['semantics'].long(), num_classes=7).float()

    def input_normal_curvature_semantics(self, pt_arxiv):
        return torch.cat([pt_arxiv['input_normals'],
                          self.input_semantics(pt_arxiv),
                          self.normed_feat(pt_arxiv, 'input_gcurv').unsqueeze(-1),
                          self.normed_feat(pt_arxiv, 'input_mcurv').unsqueeze(-1)], dim=1)

    def normed_feat(self, pt_arxiv, feat):
        return (pt_arxiv[feat] - self.stats['mean'][feat]) / (self.stats['std'][feat] + 1e-7)


def random_color(num_views):
    randoms = []
    for i in range(num_views):
        r, g, b = random.randint(0, 255) / 127.5 - 1, random.randint(0, 255) / 127.5 - 1, random.randint(0, 255) / 127.5 - 1
        randoms.append(torch.from_numpy(np.array([r, g, b]).reshape((1, 3, 1, 1))).float())
    return torch.cat(randoms, dim=0)


def random_grayscale(num_views):
    randoms = []
    for i in range(num_views):
        c = random.randint(0, 255) / 127.5 - 1
        randoms.append(torch.from_numpy(np.array([c, c, c]).reshape((1, 3, 1, 1))).float())
    return torch.cat(randoms, dim=0)


def white(num_views):
    return torch.from_numpy(np.array([1, 1, 1]).reshape((1, 3, 1, 1))).expand(num_views, -1, -1, -1).float()


def get_semi_random_views(num_views):
    elevation_params = [1.407, 0.207, 0.785, 1.767]
    azimuth = random.sample(np.arange(0, 2 * math.pi).tolist(), num_views)
    elevation = [min(max(x, elevation_params[2]), elevation_params[3])
                 for x in np.random.normal(loc=elevation_params[0], scale=elevation_params[1], size=num_views).tolist()]
    return [{'azimuth': a, 'elevation': e} for a, e in zip(azimuth, elevation)]


def get_random_views(num_views):
    elevation_params = [1.407, 0.207, 0.785, 1.767]
    azimuth = random.sample(np.arange(0, 2 * math.pi).tolist(), num_views)
    elevation = np.random.uniform(low=elevation_params[2], high=elevation_params[3], size=num_views).tolist()
    return [{'azimuth': a, 'elevation': e} for a, e in zip(azimuth, elevation)]


def rgb_to_lab(rgb_normed):
    permute = (lambda x: x.permute((0, 2, 3, 1))) if len(rgb_normed.shape) == 4 else (lambda x: x)
    permute_back = (lambda x: x.permute((0, 3, 1, 2))) if len(rgb_normed.shape) == 4 else (lambda x: x)
    device = rgb_normed.device
    rgb_normed = permute(rgb_normed).cpu().numpy()
    lab_arr = color.rgb2lab(((rgb_normed * 0.5 + 0.5) * 255).astype(np.uint8))
    lab_arr = torch.from_numpy(lab_arr).float().to(device)
    lab_arr[..., 0] = lab_arr[..., 0] / 50. - 1
    lab_arr[..., 1] = lab_arr[..., 1] / 100
    lab_arr[..., 2] = lab_arr[..., 2] / 100
    lab_arr = permute_back(lab_arr).contiguous()
    return lab_arr


def lab_to_rgb(lab_normed):
    permute = (lambda x: x.permute((0, 2, 3, 1))) if len(lab_normed.shape) == 4 else (lambda x: x)
    permute_back = (lambda x: x.permute((0, 3, 1, 2))) if len(lab_normed.shape) == 4 else (lambda x: x)
    device = lab_normed.device
    lab_normed = permute(lab_normed)
    lab_normed[..., 0] = torch.clamp((lab_normed[..., 0] * 0.5 + 0.5) * 100, 0, 99)
    lab_normed[..., 1] = torch.clamp(lab_normed[..., 1] * 100, -100, 100)
    lab_normed[..., 2] = torch.clamp(lab_normed[..., 2] * 100, -100, 100)
    lab_normed = lab_normed.cpu().numpy()
    rgb_arr = color.lab2rgb(lab_normed)
    rgb_arr = torch.from_numpy(rgb_arr).to(device)
    rgb_arr = permute_back(rgb_arr)
    rgb_normed = rgb_arr * 2 - 1
    return rgb_normed

