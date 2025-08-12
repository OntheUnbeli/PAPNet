import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class NYUv2(data.Dataset):

    def __init__(self, cfg, mode='train', ):
        assert mode in ['train', 'test']

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        self.mode = mode
        self.class_weight = np.array([4.01302219, 5.17995767, 12.47921102, 13.79726557, 18.47574439, 19.97749822,
                                      21.10995738, 25.86733191, 27.50483598, 27.35425244, 25.12185149, 27.04617447,
                                      30.0332327, 29.30994935, 34.72009825, 33.66136128, 34.28715586, 32.69376342,
                                      33.71574286, 37.0865665, 39.70731054, 38.60681717, 36.37894266, 40.12142316,
                                      39.71753044, 39.27177794, 43.44761984, 42.96761184, 43.98874667, 43.43148409,
                                      43.29897719, 45.88895515, 44.31838311, 44.18898992, 42.93723439, 44.61617778,
                                      47.12778303, 46.21331253, 27.69259756, 25.89111664, 15.65148615, ])

        with open(os.path.join(cfg['root'], f'{mode}.txt'), 'r') as f:
            self.image_depth_labels = f.readlines()

    def __len__(self):
        return len(self.image_depth_labels)

    def __getitem__(self, index):
        image_path, depth_path, label_path = self.image_depth_labels[index].strip().split(',')
        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB')  # 1 channel -> 3
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.mode == 'train':  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0),
                (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128),
                (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
                (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64),
                (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192),
                (0, 128, 192), (128, 128, 192), (64, 0, 64)]  # 41


if __name__ == '__main__':
    import json

    path = '../../configs/nyuv2.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    dataset = NYUv2(cfg, mode='train')
    from toolbox.utils import class_to_RGB
    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        depth = sample['depth']
        label = sample['label']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        depth = depth.numpy()
        depth = depth.transpose((1, 2, 0))
        depth *= np.asarray([0.226, 0.226, 0.226])
        depth += np.asarray([0.449, 0.449, 0.449])

        label = label.numpy()
        label = class_to_RGB(label, N=41, cmap=dataset.cmap)

        plt.subplot('131')
        plt.imshow(image)
        plt.subplot('132')
        plt.imshow(depth)
        plt.subplot('133')
        plt.imshow(label)

        plt.show()
