import torch
import torch_scatter
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset
from collections.abc import Mapping, Sequence

from model.differentiable_renderer import transform_pos_mvp, transform_pos_mvp_idle
from util.misc import EasyDict


def get_default_perspective_cam():
    camera = np.array([[886.81, 0., 512., 0.],
                       [0., 886.81, 512., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]], dtype=np.float32)
    return camera


def to_vertex_colors_scatter(face_colors, batch):
    assert face_colors.shape[1] == 3
    vertex_colors = torch.zeros((batch["vertices"].shape[0] // batch["mvp"].shape[1], face_colors.shape[1] + 1)).to(face_colors.device) # +1
    torch_scatter.scatter_mean(face_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_colors)
    # torch_scatter.scatter_max(face_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_colors)
    # vertex_colors = torch.hstack([vertex_colors, torch.zeros(len(vertex_colors), 1).to(vertex_colors.device)])
    vertex_colors[:, 3] = 1
    return vertex_colors[batch['vertex_ctr'], :]


def to_vertex_features_scatter(face_features, batch):
    vertex_features = torch.zeros((batch["vertices"].shape[0] // batch["mvp"].shape[1], face_features.shape[1])).to(face_features.device)
    torch_scatter.scatter_mean(face_features.unsqueeze(1).expand(-1, 4, -1).reshape(-1, face_features.shape[1]), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_features)
    # torch_scatter.scatter_max(face_features.unsqueeze(1).expand(-1, 4, -1).reshape(-1, face_features.shape[1]), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_features)
    return vertex_features[batch['vertex_ctr'], :]


def to_vertex_colors_scatter_level(all_face_colors, batch, level, level_specific=None):
    if level_specific is not None:
        lvl = level_specific
    else:
        lvl = level
    assert all_face_colors[level].shape[1] == 3
    vertex_colors = torch.zeros((batch["vertices_maps"][lvl].shape[0] // batch["mvp"].shape[1], all_face_colors[level].shape[1] + 1)).to(all_face_colors[level].device)
    torch_scatter.scatter_mean(all_face_colors[level].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3).to(all_face_colors[level].device),
                               batch["indices_maps"][lvl].reshape(-1).long().to(all_face_colors[level].device), dim=0, out=vertex_colors)
    vertex_colors[:, 3] = 1
    return vertex_colors[batch['vertex_ctr_maps'][lvl], :]


def to_vertex_features_scatter_level(all_face_features, batch, level):
    vertex_features = torch.zeros((batch["vertices_maps"][level].shape[0] // batch["mvp"].shape[1], all_face_features[level].shape[1])).to(all_face_features[level].device)
    torch_scatter.scatter_mean(all_face_features[level].unsqueeze(1).expand(-1, 4, -1).reshape(-1, all_face_features[level].shape[1]).to(all_face_features[level].device),
                               batch["indices_maps"][level].reshape(-1).long().to(all_face_features[level].device), dim=0, out=vertex_features)
    return vertex_features[batch['vertex_ctr_maps'][level], :]


def to_vertex_shininess_scatter(face_shininess, batch):
    assert face_shininess.shape[1] == 1
    vertex_shininess = torch.zeros((batch["vertices"].shape[0] // batch["mvp"].shape[1], face_shininess.shape[1])).to(face_shininess.device)
    torch_scatter.scatter_mean(face_shininess.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 1), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_shininess)
    return vertex_shininess[batch['vertex_ctr'], :]


def to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    if 'graph_data' in batch:
        for k in batch['graph_data'].keys():
            if isinstance(batch['graph_data'][k], torch.Tensor):
                batch['graph_data'][k] = batch['graph_data'][k].to(device)
            elif isinstance(batch['graph_data'][k], list):
                for m in range(len(batch['graph_data'][k])):
                    if isinstance(batch['graph_data'][k][m], torch.Tensor):
                        batch['graph_data'][k][m] = batch['graph_data'][k][m].to(device)
    if 'sparse_data' in batch:
        batch['sparse_data'] = [batch['sparse_data'][0].cuda(), batch['sparse_data'][1].cuda(), batch['sparse_data'][2].cuda()]
    if 'sparse_data_064' in batch:
        batch['sparse_data_064'] = [batch['sparse_data_064'][0].cuda(), batch['sparse_data_064'][1].cuda()]
    return batch


def to_device_graph_data(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], list):
            for m in range(len(batch[k])):
                if isinstance(batch[k][m], torch.Tensor):
                    batch[k][m] = batch[k][m].to(device)
    return batch


class GraphDataLoader(torch.utils.data.DataLoader):

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch=None,
            exclude_keys=None,
            texopt=False,
            **kwargs,
    ):
        if exclude_keys is None:
            exclude_keys = []
        if follow_batch is None:
            follow_batch = []
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.texopt = texopt

        super().__init__(dataset, batch_size, shuffle, collate_fn=Collater(follow_batch, exclude_keys, texopt), **kwargs)


class Collater(object):

    def __init__(self, follow_batch, exclude_keys, texopt):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.texopt = texopt

    @staticmethod
    def cat_collate(batch, dim=0):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                # noinspection PyProtectedMember
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.cat(batch, dim, out=out)
        raise NotImplementedError

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, EasyDict):
            if 'face_neighborhood' in elem:  # face conv data
                face_neighborhood, sub_neighborhoods, pads, node_counts, pool_maps, lateral_maps, ff2_maps, is_pad, level_masks = [], [], [], [], [], [], [], [], []
                positions = []
                pad_sum = 0

                for b_i in range(len(batch)):
                    face_neighborhood.append(batch[b_i].face_neighborhood + pad_sum)
                    pad_sum += batch[b_i].is_pad[0].shape[0]

                for sub_i in range(len(elem.pads)):
                    is_pad_n = []
                    pad_n = 0
                    for b_i in range(len(batch)):
                        is_pad_n.append(batch[b_i].is_pad[sub_i])
                        batch[b_i].level_masks[sub_i][:] = b_i
                        pad_n += batch[b_i].pads[sub_i]
                    is_pad.append(self.cat_collate(is_pad_n))
                    level_masks.append(self.cat_collate([batch[b_i].level_masks[sub_i] for b_i in range(len(batch))]))
                    pads.append(pad_n)

                if 'positions' in elem:
                    for sub_i in range(len(elem.pads)):
                        position_n = []
                        for b_i in range(len(batch)):
                            position_n.append(batch[b_i].positions[sub_i])
                        positions.append(self.cat_collate(position_n))

                    # for sub_i in range(len(elem.pads)):
                    #     positions[sub_i] = positions[sub_i].reshape(len(batch), -1, 3)

                # if 'normal_maps' in elem:
                #     for sub_i in range(len(elem.pads)):
                #         norm = []
                #         for b_i in range(len(batch)):
                #             norm.append(batch[b_i].normal_maps[sub_i])
                #         normal_maps.append(self.cat_collate(norm))

                if 'ff2_maps' in elem:
                    for sub_i in range(len(elem.ff2_maps)):
                        sem = []
                        for b_i in range(len(batch)):
                            sem.append(batch[b_i].ff2_maps[sub_i])
                        ff2_maps.append(self.cat_collate(sem))

                for sub_i in range(len(elem.sub_neighborhoods)):
                    sub_n = []
                    pool_n = []
                    lateral_n = []
                    node_count_n = 0
                    pad_sum = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_neighborhoods[sub_i] + pad_sum)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        lateral_n.append(batch[b_i].lateral_maps[sub_i] + node_count_n)
                        pad_sum += batch[b_i].is_pad[sub_i + 1].shape[0]
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_neighborhoods.append(self.cat_collate(sub_n))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))
                    lateral_maps.append(self.cat_collate(lateral_n))

                batch_dot_dict = {
                    'face_neighborhood': self.cat_collate(face_neighborhood),
                    'sub_neighborhoods': sub_neighborhoods,
                    'pads': pads,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'lateral_maps': lateral_maps,
                    'is_pad': is_pad,
                    'ff2_maps': ff2_maps,
                    'level_masks': level_masks,
                    'positions': positions,
                }

                return batch_dot_dict
            else:  # graph conv data
                edge_index, sub_edges, node_counts, pool_maps, level_masks = [], [], [], [], []
                pad_sum = 0
                for b_i in range(len(batch)):
                    edge_index.append(batch[b_i].edge_index + pad_sum)
                    pad_sum += batch[b_i].pool_maps[0].shape[0]

                for sub_i in range(len(elem.sub_edges) + 1):
                    for b_i in range(len(batch)):
                        batch[b_i].level_masks[sub_i][:] = b_i
                    level_masks.append(self.cat_collate([batch[b_i].level_masks[sub_i] for b_i in range(len(batch))]))

                for sub_i in range(len(elem.sub_edges)):
                    sub_n = []
                    pool_n = []
                    node_count_n = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_edges[sub_i] + node_count_n)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_edges.append(self.cat_collate(sub_n, dim=1))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))

                batch_dot_dict = {
                    'edge_index': self.cat_collate(edge_index, dim=1),
                    'sub_edges': sub_edges,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'level_masks': level_masks
                }

                return batch_dot_dict
        elif isinstance(elem, Mapping):
            retdict = {}
            for key in elem:
                if 'cam_position' in elem:
                    retdict['view_vector'] = self.cat_collate([(d['vertices'].unsqueeze(0).expand(d['cam_position'].shape[0], -1, -1) - d['cam_position'].unsqueeze(1).expand(-1, d['vertices'].shape[0], -1)).reshape(-1, 3) for d in batch])
                if key in ['x', 'y', 'bg', 'x_0', 'x_1']:
                    retdict[key] = self.cat_collate([d[key] for d in batch])
                elif key == 'vertices':
                    if self.texopt:
                        retdict[key] = self.cat_collate([transform_pos_mvp_idle(d['vertices'], d['mvp']) for d in batch])
                    else:
                        retdict[key] = self.cat_collate([transform_pos_mvp(d['vertices'], d['mvp']) for d in batch])
                    # retdict[key] = self.cat_collate([transform_pos_mvp_idle(d['vertices'], d['mvp']) for d in batch])
                elif key == 'vertices_maps':
                    v_maps = []
                    for level in range(len(batch[0]['vertices_maps'])):
                        level_vertices = []
                        for b_i in range(len(batch)):
                            level_vertices += [batch[b_i]['vertices_maps'][level]]
                        if self.texopt:
                            v_maps += [self.cat_collate([transform_pos_mvp_idle(level_vertices[i], d['mvp']) for i, d in enumerate(batch)])]
                        else:
                            v_maps += [self.cat_collate([transform_pos_mvp(level_vertices[i], d['mvp']) for i, d in enumerate(batch)])]
                    retdict[key] = v_maps
                elif key == 'normal_maps':
                    n_maps = []
                    for level in range(len(batch[0]['normal_maps'])):
                        level_normals = []
                        for b_i in range(len(batch)):
                            level_normals += [batch[b_i]['normal_maps'][level]]
                        n_maps += [self.cat_collate([level_normals[i] for i, d in enumerate(batch)])]
                    retdict[key] = n_maps
                elif key == 'normals':
                    retdict[key] = self.cat_collate([d['normals'].unsqueeze(0).expand(d['cam_position'].shape[0], -1, -1).reshape(-1, 3) for d in batch])
                elif key == 'uv':
                    uvs = []
                    for b_i in range(len(batch)):
                        batch[b_i][key][:, 0] += b_i * 6
                        uvs.append(batch[b_i][key].unsqueeze(0).expand(batch[b_i]['cam_position'].shape[0], -1, -1).reshape(-1, 3))
                    retdict[key] = self.cat_collate(uvs)
                elif key == 'indices':
                    num_vertex = 0
                    indices = []
                    for b_i in range(len(batch)):
                        for view_i in range(batch[b_i]['mvp'].shape[0]):
                            indices.append(batch[b_i][key] + num_vertex)
                            num_vertex += batch[b_i]['vertices'].shape[0]
                    retdict[key] = self.cat_collate(indices)
                elif key == 'tri_indices_maps':
                    t_maps = []
                    for level in range(len(batch[0]['tri_indices_maps'])):
                        num_vertex = 0
                        indices = []
                        for b_i in range(len(batch)):
                            for view_i in range(batch[b_i]['mvp'].shape[0]):
                                indices.append(batch[b_i][key][level] + num_vertex)
                                num_vertex += batch[b_i]['vertices_maps'][level].shape[0]
                        t_maps += [self.cat_collate(indices)]
                    retdict[key] = t_maps
                elif key == 'indices_quad':
                    num_vertex = 0
                    indices_quad = []
                    for b_i in range(len(batch)):
                        indices_quad.append(batch[b_i][key] + num_vertex)
                        num_vertex += batch[b_i]['vertices'].shape[0]
                    retdict[key] = self.cat_collate(indices_quad)
                elif key == 'indices_maps':
                    i_maps = []
                    for level in range(len(batch[0]['indices_maps'])):
                        num_vertex = 0
                        indices_quad = []
                        for b_i in range(len(batch)):
                            indices_quad.append(batch[b_i][key][level] + num_vertex)
                            num_vertex += batch[b_i]['vertices_maps'][level].shape[0]
                        i_maps += [self.cat_collate(indices_quad)]
                    retdict[key] = i_maps
                elif key == 'ranges':
                    ranges = []
                    start_index = 0
                    for b_i in range(len(batch)):
                        for view_i in range(batch[b_i]['mvp'].shape[0]):
                            ranges.append(torch.tensor([start_index, batch[b_i]['indices'].shape[0]]).int())
                            start_index += batch[b_i]['indices'].shape[0]
                    retdict[key] = self.collate(ranges)
                elif key == 'ranges_maps':
                    r_maps = []
                    for level in range(len(batch[0]['ranges_maps'])):
                        ranges = []
                        start_index = 0
                        for b_i in range(len(batch)):
                            for view_i in range(batch[b_i]['mvp'].shape[0]):
                                ranges.append(torch.tensor([start_index, batch[b_i]['tri_indices_maps'][level].shape[0]]).int())
                                start_index += batch[b_i]['tri_indices_maps'][level].shape[0]
                        r_maps += [self.collate(ranges)]
                    retdict[key] = r_maps
                elif key == 'vertex_ctr':
                    vertex_counts = []
                    num_vertex = 0
                    for b_i in range(len(batch)):
                        for view_i in range(batch[b_i]['mvp'].shape[0]):
                            vertex_counts.append(batch[b_i]['vertex_ctr'] + num_vertex)
                        num_vertex += batch[b_i]['vertices'].shape[0]
                    retdict[key] = self.cat_collate(vertex_counts)
                elif key == 'vertex_ctr_maps':
                    c_maps = []
                    for level in range(len(batch[0]['vertex_ctr_maps'])):
                        vertex_counts = []
                        num_vertex = 0
                        for b_i in range(len(batch)):
                            for view_i in range(batch[b_i]['mvp'].shape[0]):
                                vertex_counts.append(batch[b_i]['vertex_ctr_maps'][level] + num_vertex)
                            num_vertex += batch[b_i]['vertices_maps'][level].shape[0]
                        c_maps += [self.cat_collate(vertex_counts)]
                    retdict[key] = c_maps
                elif key in ['real', 'mask', 'patch', 'real_hres', 'mask_hres']:
                    retdict[key] = self.cat_collate([d[key] for d in batch])
                elif key == 'uv_vertex_ctr':
                    retdict[key] = [d[key] for d in batch]
                else:
                    retdict[key] = self.collate([d[key] for d in batch])
            return retdict
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            # noinspection PyArgumentList
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]
        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)
