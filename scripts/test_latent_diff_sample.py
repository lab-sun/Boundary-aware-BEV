"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
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
import skimage
from skimage import io
from sklearn.metrics import confusion_matrix

# from src import *
import sys 
sys.path.append("/Diffusion-based-Segmentation") 
sys.path.append("..")
sys.path.append(".")
from diffBEV.src.EMA import *
from diffBEV.src.LatentDiffusion_deeplab_GPU1 import LatentDiffusionConditional
from diffBEV.src.autoencoder_lss import AutoencoderKL
# from autoencoder_condition import AutoencoderKL
from diffBEV.nets.deeplabv3_plus_new import DeepLab
from diffBEV.nets.attention import SpatialTransformer

#from diffBEV.dataset.BEVDiff_dataset import SimpleImageDataset, Img_ColorJitter
from diffBEV.dataset.BEVDiff_dataset_new import SimpleImageDataset, Img_ColorJitter, collate_fn

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


import torch.optim as optim
import logging
import time

from diffBEV.utils import get_visual_img, compute_losses, compute_results
# from scripts.evaluators.metrics_confusion import compute_results
from diffBEV.opt import get_args

import tqdm

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

import torch.distributed as dist
from guided_diffusion import dist_util, logger
import random

import collections

seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

opt = get_args()

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def calculate_metrics(conf_total, mode='Train'):
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()

        # self.logger.experiment.add_scalar('Metric_{}/average_IoU'.format(mode), average_IoU, self.current_epoch)
        # self.logger.experiment.add_scalar('Metric_{}/average_precision'.format(mode), average_precision, self.current_epoch)
        # 打印结果
        print_output = ("Test:  mIoU: {mIoU:.8f} |  mAP: {mAP:.8f} | mRecall: {mRecall:.8f} |".format(
                        mIoU=average_IoU, mAP=average_precision, mRecall=average_recall))
        print(print_output)
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}    
        for i in range(len(iou_per_class)):  
            precision_record[opt.label_list[i]] = precision_per_class[i]
            recall_record[opt.label_list[i]] = recall_per_class[i]
            iou_record[opt.label_list[i]] = iou_per_class[i]
        metirc_each_class = ("precision for each class: {} | recall for each class: {} | iou for each class: {}\n".format(precision_record, recall_record, iou_record))
        print(metirc_each_class)
        # self.logger.experiment.add_text('{}/loss'.format(mode), print_output, self.current_epoch)  # 结果写入文件
        # self.logger.experiment.add_text('{}/logger'.format(mode), metirc_each_class, self.current_epoch)
        
        return average_precision, average_recall, average_IoU


class AutoEncoder(nn.Module):
    def __init__(self,
                ae_checkpoint,
                cond_checkpoint,
                attn_checkpoint,
                pretrain_weigth = "./pretrain_weights/ae/2023-10-16-10-40/ae_epoch18.pth", 
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        self.model = torch.load(pretrain_weigth)
        # self.model = AutoencoderKL.load_state_dict(ae_checkpoint)
        train_csv_file = './data/nuScenes/train_new.data'
        train_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=train_csv_file, transform=None)
        cond_pretrain_weight = './pretrain_weights/deeplab/2023-10-29-17-04/best-deeplab-epoch48-val_iou0.75.ckpt'  # deeplab model
        self.model_cond = DeepLab(opt, train_ds).to(dist_util.dev())
        self.model_cond.load_state_dict(cond_checkpoint)
        self.corss_attn = SpatialTransformer(in_channels=320, n_heads=8, d_head=40, depth=1, context_dim=320).to(dist_util.dev())
        self.corss_attn.load_state_dict(attn_checkpoint)


        
    def forward(self,input):
        return self.model(input).sample
    
    def encode(self,input,mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    def encode_cond(self,input,mode=False):
        # 自己训练的模型.latent_dist怎么处理?
        dist=self.model_cond.encoder(input)
        return dist
    
    # 加上cross attention
    def attn(self, input, cond):
        x = self.corss_attn(input, cond)
        return x
    
    def decode(self, x, cond=None):
        #return self.model.decode(input, H=150, W=150).sample
        if cond is not None:
            x = self.attn(x, cond)
            # x = x + cond # TODO 先不加试试
            x = self.model_cond.decoder(x)
        else:
            x = self.model_cond.decoder(x)
        return x

def main():
    opt = get_args()
    log_root = './logs/DiffBEV_test'
    create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_path = os.path.join(log_root, create_time)

    logger = TensorBoardLogger('logs', name='DiffBEV_test/{}/'.format(create_time))

    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    


    #model_path = './logs/DiffBEV_GPU1/2023-11-03-05-17/version_0/checkpoints/epoch=135-step=37808.ckpt'
    model_path = './pretrain_weights/DiffBEV_GPU1/2023-11-04-13-19/best-lss-epoch149-val_iou0.34.ckpt'

    train_csv_file = './data/nuScenes/train_new.data'
    train_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=train_csv_file, transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    
    test_csv_file = './data/nuScenes/test_new.data'
    test_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=test_csv_file, transform=None)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        )

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    import pdb; pdb.set_trace()
    ae_model_state_dict = collections.OrderedDict()
    ae_model_cond_state_dict = collections.OrderedDict()
    ae_cross_attn = collections.OrderedDict()
    model_state_dict = collections.OrderedDict()
    for key in checkpoint['state_dict'].keys():
        if 'ae.model.' in key:
            ae_model_state_dict[key] = checkpoint['state_dict'][key]
        elif 'ae.model_cond.' in key:
            new_key = '.'.join(key.split('.')[2:])
            ae_model_cond_state_dict[new_key] = checkpoint['state_dict'][key]
        elif 'ae.corss_attn.' in key:
            # import pdb; pdb.set_trace()
            new_key = '.'.join(key.split('.')[2:])
            ae_cross_attn[new_key] = checkpoint['state_dict'][key]
        else:
            new_key = '.'.join(key.split('.')[1:])
            model_state_dict[new_key] = checkpoint['state_dict'][key]
    import pdb; pdb.set_trace()

    ae=AutoEncoder(ae_model_state_dict, ae_model_cond_state_dict, ae_cross_attn)
    ae.eval()

    model, diffusion = create_model_and_diffusion(
         **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(model_state_dict, strict=False)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    sample_fn = (diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known)

    conf_list = []
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            (bev_labels, images, bev_images, img_names, scene_names,\
                    rots, trans, intrins, post_rots, post_trans) = batch
            FV_img = images.squeeze(dim=1).float().to(dist_util.dev())
            latents_cond = ae.encode_cond(FV_img) # torch.Size([16, 4, 32, 32])
            noise = torch.randn_like(latents_cond[:, -4:, ...])
            sample_input = torch.cat([latents_cond,noise],1).to(dist_util.dev())

            model_kwargs = {}
            sample, x_noisy, org, pred_xstarts = sample_fn(
                    model,
                    (args.batch_size, 4, 32, 32), sample_input,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            # sample = sample + latents_cond
            pred = ae.decode(sample, latents_cond)
            #pred = ae.decode(latents_cond)
            pred_img = pred.clone()
        
            # 计算混淆矩阵
            label = bev_labels.to(torch.int64).cpu().numpy().squeeze().flatten()
            pred = pred.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4,5,6])
            conf_list.append(conf)

            # 保存预测图像
            save_root = './sampling_img/DiffBEV_test'
            pred_color = get_visual_img(imgs = pred_img)
            pred_save_folder = os.path.join(save_root, create_time)
            for i in range(FV_img.shape[0]):
                scene_name = scene_names[i]
                img_name = img_names[i]
                img_path = pred_save_folder + "/" + scene_name + "/"
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                skimage.io.imsave(img_path + img_name + '_nn_pred_c.png', pred_color[i].cpu().detach().numpy().transpose((1, 2, 0)))
            


    conf_total = np.stack([x for x in conf_list]).sum(axis=0)
    average_precision, average_recall, average_IoU = calculate_metrics(conf_total, mode='Val')


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        num_ensemble=1      #number of samples in the ensemble
    )
    import argparse
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
