import shutil
import os
import json
import warnings
warnings.filterwarnings('ignore')

from torch import nn as nn
import numpy as np
import sys
sys.path.append('/rhome/abokhovkin/projects/clean-fid')
from cleanfid import fid


def compute_score(folder=None):
    SOURCE_DIR = folder

    input_images = [x for x in os.listdir(SOURCE_DIR) if x.startswith('clip_')]
    input_image_ids = [x.split('.')[0].split('_')[1] for x in input_images]
    fake_images = [x for x in os.listdir(SOURCE_DIR) if x.startswith('fake_')]
    fake_image_ids = [x.split('.')[0].split('_')[1] for x in fake_images]
    intersection_ids = [x for x in input_image_ids if x in fake_image_ids]

    FAKE_DIR = os.path.join(SOURCE_DIR, 'fake')
    GT_DIR = os.path.join(SOURCE_DIR, 'gt')
    os.makedirs(FAKE_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)

    for filename in os.listdir(FAKE_DIR):
        os.remove(os.path.join(FAKE_DIR, filename))
    for filename in os.listdir(GT_DIR):
        os.remove(os.path.join(GT_DIR, filename))

    for image_id in intersection_ids:
        shutil.copyfile(os.path.join(SOURCE_DIR, f'clip_{image_id}.jpg'),
                        os.path.join(GT_DIR, f'gt_{image_id}.jpg'))
        shutil.copyfile(os.path.join(SOURCE_DIR, f'fake_{image_id}.jpg'),
                        os.path.join(FAKE_DIR, f'fake_{image_id}.jpg'))

    feats1, feats2 = fid.compute_fid(GT_DIR, FAKE_DIR, device='cuda', num_workers=0, only_features=True)

    all_fid_scores = []
    for i in range(len(feats1)):
        mu1 = np.mean(feats1[i:i + 1], axis=0)
        sigma1 = np.cov(feats1[i:i + 1], rowvar=False)
        mu2 = np.mean(feats2[i:i + 1], axis=0)
        sigma2 = np.cov(feats2[i:i + 1], rowvar=False)
        fid_score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
        all_fid_scores += [fid_score]
    fid_score_mean = np.mean(all_fid_scores)
    fid_score_mean = float(fid_score_mean)

    with open(os.path.join(SOURCE_DIR, 'score.json'), 'w') as fout:
        json.dump({'fid': fid_score_mean}, fout)

    for filename in os.listdir(FAKE_DIR):
        os.remove(os.path.join(FAKE_DIR, filename))
    for filename in os.listdir(GT_DIR):
        os.remove(os.path.join(GT_DIR, filename))


if __name__ == '__main__':

    SOURCE_DIR = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/temp_eg3d_chair_shapenet_600iter_vgg_1view_full_patch_lat0-0-1-3_delta/images/000000'

    compute_score(SOURCE_DIR)
