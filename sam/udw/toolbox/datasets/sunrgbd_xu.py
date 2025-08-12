import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.utils import color_map
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class SUNRGBD(data.Dataset):

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
        # self.dp_to_tensor = transforms.Compose([
        #     transforms.Normalize([19050, 19050, 19050], [9650, 9650, 9650]),
        # ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            Resize(crop_size),
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        self.val_resize = Resize(crop_size)

        self.mode = mode
        # self.class_weight = np.array([3.07860905, 5.86517208, 7.36016973, 25.5572029, 20.28208529, 10.19883565,
        #                               23.16306091, 13.7528649, 27.57017251, 27.4027244, 41.2895782, 41.94953786,
        #                               38.26746684, 44.83723509, 27.28915618, 44.27512151, 38.07656634, 42.32034235,
        #                               39.5349271, 41.88240461, 50.20736118, 45.09341373, 41.1468141, 44.11455285,
        #                               46.24663617, 46.94585461, 45.09312586, 47.1823206, 50.12733943, 41.54890341,
        #                               41.66712935, 48.68193567, 48.46338583, 44.47696597, 42.55602713, 46.22161329,
        #                               46.67457597, 46.3630427, ])

        # self.train_ids, self.test_ids = train_test_split(np.arange(1449), train_s=795, random_state=random_state)
        # with open(os.path.join(cfg['root'], f'{mode}.txt'), 'r') as f:
        #     self.image_depth_labels = f.readlines()
        self.train_ids1 = os.listdir('/home/yangenquan/PycharmProjects/SUN_RGBD/train/image')
        self.test_ids1 = os.listdir('/home/yangenquan/PycharmProjects/SUN_RGBD/test/image')
        self.train_ids2 = os.listdir('/home/yangenquan/PycharmProjects/SUN_RGBD/train/depth')
        self.test_ids2 = os.listdir('/home/yangenquan/PycharmProjects/SUN_RGBD/test/depth')
    def __len__(self):

        if self.mode == 'train':
            return len(self.train_ids1)
        else:
            return len(self.test_ids1)
    # def dp_to_tensor(self, depth):
    #     depth = np.asarray(depth)
    #     depth = torch.from_numpy(depth).float().unsqueeze(0)
    #     depth = transforms.Normalize(mean=[19050],
    #                                  std=[9650])(depth)
    #     depth = torch.cat([depth, depth, depth])
    #     return depth

    def __getitem__(self, index):
        #  image_path, depth_path, label_path = self.image_depth_labels[index].strip().split(',')
        if self.mode == 'train':
            image_index1 = self.train_ids1[index]
            image_index2 = self.train_ids2[index]
            image_path = f'train/image/{image_index1}'
            depth_path = f'train/depth/{image_index2}'
            label_path = f'train/label/{image_index2}'
        else:
            image_index1 = self.test_ids1[index]
            image_index2 = self.test_ids2[index]
            image_path = f'test/image/{image_index1}'
            depth_path = f'test/depth/{image_index2}'
            label_path = f'test/label/{image_index2}'
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
        else:
            sample = self.val_resize(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return color_map(N=self.n_classes)


if __name__ == '__main__':
    import json

    path = '../../configs/sunrgbd.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    # cfg['root'] = '/home/dtrimina/Desktop/lxy/database/SUNRGBD'
    dataset = SUNRGBD(cfg, mode='test')
    from toolbox.utils import class_to_RGB
    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        print(image.shape)
        # print(image.shape, 'iamge')
        depth = sample['depth']
        # print(depth.shape, 'depth')
        label = sample['label']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        depth = depth.numpy()
        depth = depth.transpose((1, 2, 0))
        # depth *= np.asarray([9650, 9650, 9650])
        # depth += np.asarray([19050, 19050, 19050])
        depth *= np.asarray([0.226, 0.226, 0.226])
        depth += np.asarray([0.449, 0.449, 0.449])

        label = label.numpy()
        label = class_to_RGB(label, N=38, cmap=dataset.cmap)
        # print(dataset.cmap)

        plt.subplot('131')
        plt.imshow(image)
        plt.subplot('132')
        plt.imshow(depth)
        plt.subplot('133')
        plt.imshow(label)

        # plt.imshow(depth)

        plt.show()

        break
