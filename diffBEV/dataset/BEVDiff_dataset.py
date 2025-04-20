import torch
import torch.nn.functional as F
import torchvision
# import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pytorch_lightning as pl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from skimage import io
import os

# from src import *

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

import pandas as pd
from torchvision import transforms

class SimpleImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 root_dir,
                 transform=None,):
        self.root_dir = root_dir
        self.transforms = transform

        # data_keys = ['input', 'mask']
        # # set up transforms
        # if self.transforms is not None:
        #     self.input_T = KA.container.AugmentationSequential(
        #         *self.transforms,
        #         data_keys=data_keys,
        #         same_on_batch=False
        #     )
        
        # check files
        # csv_file_path = '/home/gs/workspace/datasets/nuScenes/seq_bev_dataset/csv_files'
        # train_csv = os.path.join(csv_file_path, 'train.csv')
        # val_csv = os.path.join(csv_file_path, 'val.csv')
        # test_csv = os.path.join(csv_file_path, 'test.csv')
        self.examples = pd.read_csv(self.root_dir, header=None)
        self.img = [self.examples.iloc[i, 1] for i in range(len(self.examples))]
        self.bev_label = [self.examples.iloc[i, 3] for i in range(len(self.examples))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img[idx]
        bev_label_path = self.bev_label[idx]

        img_name = img_path.split('/')[-1].split('.')[0]
        scene_name = img_path.split('/')[-3]
        BEV_img_name = img_name + '_BEVmask_c.png'
        BEV_img_path = '/'.join(bev_label_path.split('/')[:-1]) + '/' + BEV_img_name

        image = np.asarray(io.imread(img_path))
        bev_label_orig = np.asarray(io.imread(bev_label_path))
        bev_label = np.where(bev_label_orig==7, 0, bev_label_orig)
        bev_image = np.asarray(io.imread(BEV_img_path))

        sample = {'image': image,
                  'bev_label': bev_label,
                  'bev_image': bev_image,
                  'img_name': img_name,
                  'scene_name': scene_name,
                }
        
        if self.transforms:
            sample = self.transforms(sample)
        
        
        sample = ToTensor()(sample)
        return sample

def collate_fn(data):
    images_batch = list()
    bev_labels_batch = list()
    bev_images_batch = list()
    img_names_batch = list()
    scene_names_batch = list()
    # print('in BEVDiff_dataset, len(data):{}, type(data[0]):{}'.format(len(data), type(data[0])))

    for iter_data in data:
        images = iter_data['image']
        bev_labels = iter_data['bev_label']
        bev_images = iter_data['bev_image']
        img_names = iter_data['img_name']
        scene_names = iter_data['scene_name']
        # print('in BEVDiff_dataset, images.shape: ',images.shape)

        images_batch.append(images)
        bev_labels_batch.append(bev_labels)
        bev_images_batch.append(bev_images)
        img_names_batch.append(img_names)
        scene_names_batch.append(scene_names)

    ret_list = [
        
        torch.stack(bev_labels_batch),
        torch.stack(images_batch),
        torch.stack(bev_images_batch),
        img_names_batch,
        scene_names_batch,
    ]  # images_batch, bev_labels_batch, bev_images_batch, img_names_batch, scene_names_batch
    # print('in BEVDiff_dataset, images.shape in ret_list: ',ret_list[0].shape)
    return ret_list


class ToTensor(object):
    def __call__(self, sample):
        trans = transforms.Compose([transforms.ToTensor()])  # ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。具体可参见如下代码： 
        sample_tensor = {}
        sample_tensor['image'] = trans(sample['image'])
        sample_tensor['bev_image'] = trans(sample['bev_image'])
        sample_tensor['bev_label'] = torch.from_numpy(sample['bev_label'])
        sample_tensor['img_name'] = sample['img_name']
        sample_tensor['scene_name'] = sample['scene_name']
        return sample_tensor

class Img_ColorJitter(object):
    def __init__(self, brightness=0.5, prob=0.9) -> None:
        self.brightness = brightness
        self.prob = prob
    def __call__(self, sample):
        image = sample['image']
        bev_image = sample['bev_image']
        bev_label = sample['bev_label']
        img_name = sample['img_name']
        scene_name = sample['scene_name']

        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.brightness, 0.8+self.brightness)
            image = (image * bright_factor).astype(image.dtype)
        
        sample = {'image': image,
                  'bev_image': bev_image,
                  'bev_label': bev_label,
                  'img_name': img_name,
                  'scene_name': scene_name,
                }
        return sample

if __name__ == '__main__':
    import time
    csv_file = './data/nuScenes/test.csv'
    test_dataset = SimpleImageDataset(csv_file, transform=Img_ColorJitter())
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    print('The length of test_loader is ',len(test_loader))
    start_time = time.time()
    test_data = next(iter(test_loader))
    end_time = time.time()
    duration_1 = end_time - start_time

    start = time.time()
    for i in range(1):
        for batch_idx, input_data in enumerate(test_loader):
            print('this is {} epoch, input_img shape is {}'.format(i, input_data['image'].shape))
            image = input_data['image']
            bev_image = input_data['bev_image']
            bev_label = input_data['bev_label']
            img_name = input_data['img_name']
            scene_name = input_data['scene_name']
            print('the scene name is: ', scene_name)
            print('shape of image:{}, bev_image:{}, bev_label:{}'.format(image.shape, bev_image.shape, bev_label.shape))
            print('the num class of bev_label:{}'.format(np.unique(bev_label)))
            print('the max and min of image: [{}, {}]'.format(image.max(), image.min()) )
            print('the max and min of bev_image: [{}, {}]'.format(bev_image.max(), bev_image.min()) )
    end = time.time()
    duration = end - start
    print("test load duration: ", duration)

    print('type of test_data: ', type(test_data))
    print('test load duration: ', duration_1)
    print('The length of test_dataset is ',len(test_dataset))
    print('The length of test_loader is ',len(test_loader))
