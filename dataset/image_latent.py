import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms


class ImageLatentDataset(torch.utils.data.Dataset):

    def __init__(self, config, file_range=(0, 1024)):
        self.config = config

        self.dataset_directory = Path(config.dataset_path)
        # training
        self.dataset_directory = '/cluster/valinor/abokhovkin/scannet-texturing/data/Projections/22111349_83_cars'
        # self.dataset_directory = '/cluster/valinor/abokhovkin/scannet-texturing/data/Projections/15112113_70_chairs'

        # inference
        # self.dataset_directory = '/cluster/valinor/abokhovkin/scannet-texturing/data/Projections/15112113_70_chairs_proj_stored/shapenet_unaligned' # chairs
        # self.dataset_directory = '/cluster/valinor/abokhovkin/scannet-texturing/data/Projections/22111349_83_cars_proj_stored/shapenet_aligned' # cars

        self.image_size = config.image_size

        # self.items = list(x.stem for x in Path(self.dataset_directory).iterdir() if x.name.endswith('.jpg') and int(x.stem.split('_')[0]) in range(file_range[0], file_range[1]))
        self.items = list(x.stem for x in Path(self.dataset_directory).iterdir() if x.name.endswith('.jpg'))

        self.items = sorted(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocess_store = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])

        selected_item = self.items[idx]
        img_id, view_id, postfix = selected_item.split('_')
        input_image = Image.open(os.path.join(self.dataset_directory, f'{selected_item}.jpg'))
        input_image_train = preprocess(input_image)
        input_image_store = preprocess_store(input_image)

        latent = torch.load(os.path.join(self.dataset_directory, f'globlat_{img_id}_{postfix}.pth'))

        return input_image_train, latent, input_image_store, selected_item