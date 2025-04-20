import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn.functional as F
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

# from src import *
import sys 
sys.path.append("/Diffusion-based-Segmentation") 
sys.path.append("..")
sys.path.append(".")
from diffBEV.src.EMA import *
from diffBEV.src.LatentDiffusion_deeplab_GPU3_crossTrans import LatentDiffusionConditional
from diffBEV.src.autoencoder_lss import AutoencoderKL
from diffBEV.nets.deeplabv3_plus import DeepLab

from diffBEV.dataset.BEVDiff_dataset_new import SimpleImageDataset, Img_ColorJitter, collate_fn

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


import torch.optim as optim
import logging
import time

from diffBEV.utils import compute_losses, compute_results
from diffBEV.opt import get_args



if __name__ == '__main__':
    opt = get_args()
    # Device = 'cuda'
    # Epoch = 20
    # lr = 0.005
    # batch_size = 48
    log_root = './logs/DiffBEV_GPU3_crossTrans'
    log_frequency = 10
    ckpt_root = './pretrain_weights/DiffBEV_GPU3_crossTrans/'
    save_frequency = 1
    create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_path = os.path.join(log_root, create_time)
    ckpt_path = os.path.join(ckpt_root, create_time)
    label_list = ['background', 'drivable_area', 'ped_crossing', 'walkway', 'movable_object', 'vehicle', 'predestrian']

    best_iou = 0
    best_test_iou = 0

    logger = TensorBoardLogger('logs', name='DiffBEV_GPU3_crossTrans/{}/'.format(create_time))

    train_csv_file = './data/nuScenes/train_new.data'
    train_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=train_csv_file, transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=opt.batch_size,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    val_csv_file = './data/nuScenes/test_new.data'
    val_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=val_csv_file, transform=None)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=opt.batch_size,   
        shuffle=False,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    test_csv_file = './data/nuScenes/val_new.data'
    test_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=test_csv_file, transform=None)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=opt.batch_size,   
        shuffle=False,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    model=LatentDiffusionConditional(opt, train_ds, val_ds,
                                #  lr=1e-4,
                                #  batch_size=batch_size,
                                 num_timesteps=opt.num_timesteps,
                                 latent_scale_factor=1e-2)

    checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                          mode='max', # 趋势越大越好
                                          dirpath="./logs/DiffBEV_GPU1/{}/checkpoints/".format(create_time), 
                                          monitor="val_iou", 
                                          filename="best-lss-epoch{epoch:02d}-val_iou{val_iou:.4f}",
                                          auto_insert_metric_name=False,
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')


    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        # limit_train_batches=0.02,
        # limit_val_batches=0.02,
        # limit_test_batches=0.05,
        # max_steps=400,
        # max_steps=2e5,
        max_epochs=200, # 200,
        check_val_every_n_epoch=50,
        callbacks=[EMA(0.9999), checkpoint_callback, lr_monitor],  # [EMA(0.9999), checkpoint_callback],
        val_check_interval=1.0,
        gpus = [0],
        logger=logger
    )

    import pdb; pdb.set_trace()
    trainer.fit(model)
    # trainer.fit(model, ckpt_path="./logs/DiffBEV_GPU1/2023-12-02-11-09/checkpoints/best-lss-epoch194-val_iou0.3395.ckpt")

    trainer.test(model, dataloaders=test_loader)

    # save model
    ckpt_path = './pretrain_weights/DiffBEV_GPU1/{}/'.format(create_time)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    model_save_path = os.path.join(ckpt_path, 'DiffBEV_GPU1_last.pth')# './pretrain_weights/ae_epoch{}.pth'.format(epoch) #os.path.join('./pretrain_weights/', 'ae_epoch{}.pth'.format(epoch))
    torch.save(model, model_save_path)
    
    