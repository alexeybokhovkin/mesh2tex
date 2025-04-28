import json
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms


class ImagePoseDataset(torch.utils.data.Dataset):

    def __init__(self, config, id_range=(0, 0)):
        self.config = config

        self.dataset_directory = Path(config.dataset_path)
        self.image_size = config.image_size

        self.items = list(x.stem for x in Path(self.dataset_directory).iterdir() if x.name.endswith('.jpg'))
        self.items = sorted(self.items)
        self.items = [x for x in self.items if id_range[0] <= int(x.split('_')[1]) <= id_range[1]]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocess_store = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])

        selected_item = self.items[idx]
        input_image = Image.open(os.path.join(self.dataset_directory, f'{selected_item}.jpg'))
        input_image_train = preprocess(input_image)
        input_image_store = preprocess_store(input_image)

        with open(os.path.join(self.dataset_directory, f'{selected_item}.json'), 'r') as fin:
            pose = json.load(fin)
        azimuth_id = pose['azimuth_id']
        elevation_id = pose['elevation_id']

        return input_image_train, input_image_store, azimuth_id, elevation_id