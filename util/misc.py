from collections import OrderedDict

import torch
from ballpark import business
import numpy as np
import sys
sys.path.append('/rhome/abokhovkin/projects/clean-fid')
from cleanfid import fid
from pathlib import Path


def print_model_parameter_count(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {type(model).__name__}: {business(count, precision=3, prefix=True)}")


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen if t.requires_grad]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s)], axis=1)


def get_parameters_from_state_dict(state_dict, filter_key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[k.replace(filter_key + '.', '')] = state_dict[k]
    return new_state_dict


def compute_fid(output_dir_fake, output_dir_real, filepath, device):
    fid_score = fid.compute_fid(str(output_dir_real), str(output_dir_fake), device=device, dataset_res=256, num_workers=0)
    print(f'FID: {fid_score:.3f}')
    kid_score = fid.compute_kid(str(output_dir_real), str(output_dir_fake), device=device, dataset_res=256, num_workers=0)
    print(f'KID: {kid_score:.3f}')
    Path(filepath).write_text(f"fid = {fid_score}\nkid = {kid_score}")
    

from pytorch3d.io.utils import _check_faces_indices, _make_tensor
def _format_faces_indices(faces_indices, max_index: int, device, pad_value=None):
    """
    Format indices and check for invalid values. Indices can refer to
    values in one of the face properties: vertices, textures or normals.
    See comments of the load_obj function for more details.
    Args:
        faces_indices: List of ints of indices.
        max_index: Max index for the face property.
        pad_value: if any of the face_indices are padded, specify
            the value of the padding (e.g. -1). This is only used
            for texture indices indices where there might
            not be texture information for all the faces.
    Returns:
        faces_indices: List of ints of indices.
    Raises:
        ValueError if indices are not in a valid range.
    """
    faces_indices = _make_tensor(
        faces_indices, cols=3, dtype=torch.int64, device=device
    )

    # if pad_value is not None:
    #     mask = faces_indices.eq(pad_value).all(dim=-1)
    #
    # # Change to 0 based indexing.
    # faces_indices[(faces_indices > 0)] -= 1
    #
    # # Negative indexing counts from the end.
    # faces_indices[(faces_indices < 0)] += max_index
    #
    # if pad_value is not None:
    #     # pyre-fixme[61]: `mask` is undefined, or not always defined.
    #     faces_indices[mask] = pad_value

    return _check_faces_indices(faces_indices, max_index, pad_value)

