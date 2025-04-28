import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from skimage.transform import resize


class ImageNocsDataset(torch.utils.data.Dataset):

    def __init__(self, config, id_range=(0, 0)):
        self.config = config

        self.dataset_directory = Path(config.dataset_path)
        self.image_size = config.image_size

        self.items = list(x.stem for x in Path(self.dataset_directory).iterdir() if x.name.endswith('rendercrop.jpg'))
        self.items = sorted(self.items)
        self.items = [x for x in self.items if id_range[0] <= int(x.split('_')[0]) <= id_range[1]]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()
        ])
        preprocess_store = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])

        selected_item = self.items[idx]
        img_id = selected_item.split('_rendercrop')[0]

        input_image = Image.open(os.path.join(self.dataset_directory, f'{selected_item}.jpg'))
        input_image_train = preprocess(input_image)
        input_image_store = preprocess_store(input_image)
        mask = torch.where((input_image_store[0] >= 0.92) & (input_image_store[1] >= 0.92) & (input_image_store[2] >= 0.92), 0, 1)
        input_image_train = torch.cat([input_image_train, mask[None, ...]], dim=0)

        noc = np.load(os.path.join(self.dataset_directory, f'{img_id}_noccrop.npy'))
        noc = resize(noc, (512, 512), anti_aliasing=True)
        noc = torch.permute(torch.FloatTensor(noc), (2, 0, 1))

        return input_image_train, input_image_store, noc