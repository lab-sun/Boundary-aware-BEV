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
from pyquaternion import Quaternion
from PIL import Image
import imageio
from skimage import io
import skimage
import os
import pickle

# from diffBEV.src import *

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

import pandas as pd
from torchvision import transforms

class SimpleImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 is_train,
                 opt,
                 root_dir,
                 transform=None,):
        self.root_dir = root_dir
        self.transforms = transform

        # list_data的顺序为prev, fv_img, next, bev_gt, rl_gt, intrin, rot, tran
        self.examples = []
        with open(self.root_dir,'rb') as f:
            while True:
                try:
                    list_data=pickle.load(f)
                    self.examples.append(list_data)
                except EOFError:
                    break
        #print(len(examples))
        #print(examples[0])
        self.is_train = is_train
        self.opt = opt

        # self.examples = pd.read_csv(self.root_dir, header=None)
        self.img = [self.examples[i][1] for i in range(len(self.examples))]
        self.bev_label = [self.examples[i][3] for i in range(len(self.examples))]
        self.intrin = [self.examples[i][5] for i in range(len(self.examples))]
        self.rot = [self.examples[i][6] for i in range(len(self.examples))]
        self.tran = [self.examples[i][7] for i in range(len(self.examples))]

        self.normalize_img = torchvision.transforms.Compose((
                             torchvision.transforms.ToTensor(),
                             # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ))

    def sample_augmentation(self):
        H, W = self.opt.data_aug_conf['H'], self.opt.data_aug_conf['W']
        fH, fW = self.opt.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.opt.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.opt.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.opt.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.opt.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.opt.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
    
    def img_transform(self, img, post_rot, post_tran,
                    resize, resize_dims, crop,
                    flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate/180*np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    
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

        # image = np.asarray(io.imread(img_path))
        image = Image.open(img_path) # (512,256)
        bev_label_orig = np.asarray(io.imread(bev_label_path))
        bev_label = np.where(bev_label_orig==7, 0, bev_label_orig)
        bev_image = np.asarray(io.imread(BEV_img_path))
        bev_image = skimage.transform.resize(bev_image, (300, 300), anti_aliasing=True)

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        intrin = self.intrin[idx]
        rot = self.rot[idx]
        tran = self.tran[idx]

        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        img, post_rot2, post_tran2 = self.img_transform(image, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        img = self.normalize_img(img)  # img的size在opt里

        sample = {'image': img.unsqueeze(0),
                  'bev_label': bev_label, # bev_label没有增加维度
                  'bev_image': bev_image,
                  'img_name': img_name,
                  'scene_name': scene_name,
                  'rot': rot.unsqueeze(0),
                  'tran': tran.unsqueeze(0),
                  'intrin': intrin.unsqueeze(0),
                  'post_rot': post_rot.unsqueeze(0),
                  'post_tran': post_tran.unsqueeze(0)
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
    rot_batch = list()
    tran_batch = list()
    intrin_batch = list()
    post_rot_batch = list()
    post_tran_batch = list()
    # print('in BEVDiff_dataset, len(data):{}, type(data[0]):{}'.format(len(data), type(data[0])))

    for iter_data in data:
        images = iter_data['image']
        bev_labels = iter_data['bev_label']
        bev_images = iter_data['bev_image']
        img_names = iter_data['img_name']
        scene_names = iter_data['scene_name']
        rots = iter_data['rot']
        trans = iter_data['tran']
        intrins = iter_data['intrin']
        post_rots = iter_data['post_rot']
        post_trans = iter_data['post_tran']
        # print('in BEVDiff_dataset, images.shape: ',images.shape)

        images_batch.append(images)
        bev_labels_batch.append(bev_labels)
        bev_images_batch.append(bev_images)
        img_names_batch.append(img_names)
        scene_names_batch.append(scene_names)
        rot_batch.append(rots)
        tran_batch.append(trans)
        intrin_batch.append(intrins)
        post_rot_batch.append(post_rots)
        post_tran_batch.append(post_trans)

    ret_list = [
        
        torch.stack(bev_labels_batch),
        torch.stack(images_batch),
        torch.stack(bev_images_batch),
        img_names_batch,
        scene_names_batch,
        torch.stack(rot_batch),
        torch.stack(tran_batch),
        torch.stack(intrin_batch),
        torch.stack(post_rot_batch),
        torch.stack(post_tran_batch)
    ]  # bev_labels_batch, images_batch, bev_images_batch, img_names_batch, scene_names_batch, rot_batach, tran_batch, intrin_batch, post_rot_batch, post_tran_batch
    # print('in BEVDiff_dataset, images.shape in ret_list: ',ret_list[0].shape)
    return ret_list


class ToTensor(object):
    def __call__(self, sample):
        trans = transforms.Compose([transforms.ToTensor()])  # ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。具体可参见如下代码： 
        sample_tensor = {}
        sample_tensor['image'] = sample['image'] # 在传入时已经ToTensor了
        sample_tensor['bev_image'] = trans(sample['bev_image'])
        sample_tensor['bev_label'] = torch.from_numpy(sample['bev_label'])
        sample_tensor['img_name'] = sample['img_name']
        sample_tensor['scene_name'] = sample['scene_name']
        sample_tensor['rot'] = sample['rot']
        sample_tensor['tran'] = sample['tran']
        sample_tensor['intrin'] = sample['intrin']
        sample_tensor['post_rot'] = sample['post_rot']
        sample_tensor['post_tran'] = sample['post_tran']

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
        rot = sample['rot']
        tran = sample['tran']
        intrin = sample['intrin']
        post_rot = sample['post_rot']

        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.brightness, 0.8+self.brightness)
            image = (image * bright_factor).astype(image.dtype)
        
        sample = {'image': image,
                  'bev_image': bev_image,
                  'bev_label': bev_label,
                  'img_name': img_name,
                  'scene_name': scene_name,
                  'rot': rot,
                  'tran': tran,
                  'intrin': intrin,
                  'post_rot': post_rot,
                  'post_tran': post_tran
                }
        return sample

if __name__ == '__main__':
    import time
    import sys 
    sys.path.append("/Diffusion-based-Segmentation") 
    sys.path.append("..")
    sys.path.append(".")
    from diffBEV.opt import get_args
    import matplotlib.pyplot as plt
    opt = get_args()
    data_file = './data/nuScenes/test_new.data'
    test_dataset = SimpleImageDataset(is_train=False, opt=opt, root_dir=data_file, transform=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)

    print('The length of test_loader is ',len(test_loader))
    start_time = time.time()
    test_data = next(iter(test_loader))
    end_time = time.time()
    duration_1 = end_time - start_time

    start = time.time()
    for i in range(1):
        for batch_idx, input_data in enumerate(test_loader):
            # print('this is {} epoch, input_img shape is {}'.format(i, input_data['image'].shape))
            # image = input_data['image']
            # bev_image = input_data['bev_image']
            # bev_label = input_data['bev_label']
            # img_name = input_data['img_name']
            # scene_name = input_data['scene_name']
            (bev_label, image, bev_image, img_name, scene_name,\
             rot, tran, intrin, post_rot, post_tran) = input_data
            print('the image name is ', img_name)
            print('this is {} epoch, input_img shape is {}'.format(i, image.shape))
            print('this is {} epoch, bev_label shape is {}'.format(i, bev_label.shape))
            print('the scene name is: ', scene_name)
            print('shape of image:{}, bev_image:{}, bev_label:{}'.format(image.shape, bev_image.shape, bev_label.shape))
            print('the num class of bev_label:{}'.format(np.unique(bev_label)))
            print('the max and min of image: [{}, {}]'.format(image.max(), image.min()) )
            print('the max and min of bev_image: [{}, {}]'.format(bev_image.max(), bev_image.min()) )
            print('the rot is: {} and the type of it is {} the shape of it is {}'.format(rot, type(rot), rot.shape))
            print('the tran is: {} and the type of it is {} the shape of it is {}'.format(tran, type(tran), tran.shape))
            print('the intrin is: {} and the type of it is {} the shape of it is {}'.format(intrin, type(intrin), intrin.shape))
            print('the post_rot is: {} and the type of it is {} the shape of it is {}'.format(post_rot, type(post_rot), post_rot.shape))
            print('the post_tran is: {} and the type of it is {} the shape of it is {}'.format(post_tran, type(post_tran), post_tran.shape))
            plt.figure(figsize=(24, 24))
            plt.imshow(image[0][0].permute(1,2,0))
            plt.show()
            break
    end = time.time()
    duration = end - start
    print("test load duration: ", duration)

    print('type of test_data: ', type(test_data))
    print('test load duration: ', duration_1)
    print('The length of test_dataset is ',len(test_dataset))
    print('The length of test_loader is ',len(test_loader))
