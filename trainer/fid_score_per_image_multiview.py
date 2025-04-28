import os
from skimage import io
import json
import warnings
warnings.filterwarnings('ignore')

from torch import nn as nn
import numpy as np
import sys
sys.path.append('/rhome/abokhovkin/projects/clean-fid')
from cleanfid import fid


def compute_multiview_score(folder=None, num_views=4):
    views_per_sample = num_views
    SOURCE_DIR = folder

    input_images = [x for x in os.listdir(SOURCE_DIR) if x.startswith('gt_')]
    input_image_ids = [x.split('.')[0].split('_')[1] for x in input_images]
    fake_images = [x for x in os.listdir(SOURCE_DIR) if x.startswith('fake_')]
    fake_image_ids = [x.split('.')[0].split('_')[1] for x in fake_images]
    intersection_ids = [x for x in input_image_ids if x in fake_image_ids]

    intersection_ids = [x for x in intersection_ids if int(x) <= 256]

    FAKE_DIR = os.path.join(SOURCE_DIR, 'fake')
    GT_DIR = os.path.join(SOURCE_DIR, 'gt')
    os.makedirs(FAKE_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)

    for filename in os.listdir(FAKE_DIR):
        os.remove(os.path.join(FAKE_DIR, filename))
    for filename in os.listdir(GT_DIR):
        os.remove(os.path.join(GT_DIR, filename))

    for image_id in intersection_ids:

        fake_images = io.imread(os.path.join(SOURCE_DIR, f'fake_{image_id}.jpg'))
        start_x_pixel = 2
        for k in range(views_per_sample):
            one_view = fake_images[2:514, start_x_pixel:start_x_pixel + 512, :]
            start_x_pixel += (512 + 2)
            io.imsave(os.path.join(FAKE_DIR, f'fake_{int(image_id):05d}_{k:03d}.jpg'),
                      one_view, quality=100)

        gt_images = io.imread(os.path.join(SOURCE_DIR, f'gt_{image_id}.jpg'))
        start_x_pixel = 2
        for k in range(views_per_sample):
            one_view = gt_images[2:514, start_x_pixel:start_x_pixel + 512, :]
            start_x_pixel += (512 + 2)
            io.imsave(os.path.join(GT_DIR, f'gt_{int(image_id):05d}_{k:03d}.jpg'),
                      one_view, quality=100)

    feats1, feats2 = fid.compute_fid(GT_DIR, FAKE_DIR, device='cuda', num_workers=0, only_features=True)

    all_fid_scores = []
    for i in range(len(feats1) // views_per_sample):
        mu1 = np.mean(feats1[views_per_sample * i:views_per_sample * (i + 1)], axis=0)
        sigma1 = np.cov(feats1[views_per_sample * i:views_per_sample * (i + 1)], rowvar=False)
        mu2 = np.mean(feats2[views_per_sample * i:views_per_sample * (i + 1)], axis=0)
        sigma2 = np.cov(feats2[views_per_sample * i:views_per_sample * (i + 1)], rowvar=False)
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

    SOURCE_DIR = '/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/temp_chair_scannet_550iter_stylemeshvgg_nocspatch100_delta+tuner+shape_layer3_EVALNEW/images/000000'
    num_views = 4

    compute_multiview_score(SOURCE_DIR, num_views)

