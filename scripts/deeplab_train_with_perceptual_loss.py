import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pytorch_lightning as pl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from skimage import io

import sys
sys.path.append("/Diffusion-based-Segmentation") 
sys.path.append("..")
sys.path.append(".")
# from src import *
from diffBEV.src.EMA import *
#from BEVDiff_LatentDiffusion import LatentDiffusionConditional
from diffBEV.src.autoencoder_lss import AutoencoderKL
# from autoencoder_condition import AutoencoderKL

from diffBEV.dataset.BEVDiff_dataset_new import SimpleImageDataset, Img_ColorJitter, collate_fn

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

#from torchvision.models.segmentation import deeplabv3_resnet50
from diffBEV.nets.deeplabv3_plus_new import DeepLab
from diffBEV.opt import get_args


if __name__ == '__main__':

    # import torch.optim as optim
    # import logging
    import time

    # from utils import compute_losses
    # from scripts.evaluators.metrics_confusion import compute_results
    
    opt = get_args()
    Device = 'cuda'
    Epoch = 20
    lr = 0.005
    batch_size = 32
    log_root = './logs/deeplab_ablation'
    log_frequency = 10
    ckpt_root = './pretrain_weights/deeplab_ablation/'
    save_frequency = 1
    create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_path = os.path.join(log_root, create_time)
    ckpt_path = os.path.join(ckpt_root, create_time)
    label_list = ['background', 'drivable_area', 'ped_crossing', 'walkway', 'movable_object', 'vehicle', 'predestrian']

    best_iou = 0
    best_test_iou = 0

    logger = TensorBoardLogger('logs', name='deeplab_ablation/{}/'.format(create_time))

    train_csv_file = './data/nuScenes/train_new.data'
    # train_csv_file = './data/nuScenes/all_data.data'
    train_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=train_csv_file, transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    val_csv_file = './data/nuScenes/val_new.data'
    val_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=val_csv_file, transform=None)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    test_csv_file = './data/nuScenes/test_new.data'
    test_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=test_csv_file, transform=None)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    # model=DeepLab(num_classes=7, backbone="mobilenet", downsample_factor=16, pretrained=True)
    model = DeepLab(opt, train_ds, val_ds, 
                    lr=1e-4, batch_size=batch_size, 
                    num_classes=7, backbone="mobilenet", 
                    downsample_factor=32, pretrained=True) # ablation study downsample_factor=16


    checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                          mode='max', # 趋势越大越好
                                          dirpath="./logs/deeplab_ablation/{}/".format(create_time), 
                                          monitor="val_iou", 
                                          filename="best-deeplab-epoch{epoch:02d}-val_iou{val_iou:.2f}",
                                          auto_insert_metric_name=False,
                                          save_last=True)

    trainer = pl.Trainer(
        #num_sanity_val_steps=0,
        # limit_train_batches=0.01,
        # limit_val_batches=0.01,
        # max_steps=100,
        max_steps=13310/batch_size*50,
        callbacks=[EMA(0.9999), checkpoint_callback],
        val_check_interval=1.0,
        gpus = [0],
        logger=logger
    )

    import pdb; pdb.set_trace()
    trainer.fit(model)

    # save model
    ckpt_path = './pretrain_weights'
    model_save_path = os.path.join(ckpt_path, 'deeplab.pth')# './pretrain_weights/ae_epoch{}.pth'.format(epoch) #os.path.join('./pretrain_weights/', 'ae_epoch{}.pth'.format(epoch))
    torch.save(model, model_save_path)
    
    # sampling
    # import pdb; pdb.set_trace()
    batch = next(iter(test_loader))
    (bev_labels, images, bev_images, img_names, scene_names) = batch

    # import pdb; pdb.set_trace()
    model.cuda()
    out=model(images.to(Device))
    # 输出也为4张，4张是同一个input的不同采样输出
    print('in train.py, out.shape: ', out.shape)  # out.shape:  torch.Size([4, 7, 150, 150])