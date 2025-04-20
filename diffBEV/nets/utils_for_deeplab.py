import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2
from pytorch_toolbelt import losses as L

from torchvision.models import vgg16
from diffBEV.src.autoencoder_lss import AutoencoderKL
# from diffBEV.src.autoencoder_lss import *

def vgg16_loss(feature_module, loss_func, y, y_):
    out = feature_module(y)
    out_ = feature_module(y_)
    loss = loss_func(out, out_)
    return loss

def get_feature_module(layer_index, device=None):
    vgg = vgg16(weights='DEFAULT', progress=True).features # pretrained=True, 
    vgg.eval()
    # pretrain_weigth = "./pretrain_weights/ae/2023-10-16-10-40/ae_epoch18.pth"
    # preceptualLoss_model = torch.load(pretrain_weigth)

     # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False
    # for parm in preceptualLoss_model.parameters():
    #     parm.requires_grad = False

    feature_module = vgg[0:layer_index + 1]
    feature_module.to(device)
    return feature_module

class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, layer_indexs=None, device=None):
        super(PerceptualLoss, self).__init__()
        pretrain_weigth = "./pretrain_weights/ae/2023-10-16-10-40/ae_epoch18.pth"
        self.preceptualLoss_model = torch.load(pretrain_weigth)
        for parm in self.preceptualLoss_model.parameters():
            parm.requires_grad = False
        self.creation = loss_func
        self.layer_indexs = layer_indexs
        self.device = device

    def forward(self, encoded_feat, seg_pred, bev_img):
        loss = 0
        latent = self.preceptualLoss_model.encode(bev_img.float()).latent_dist.mode() # torch.Size([16, 4, 37, 37])
        latent = nn.functional.interpolate(latent, (32,32), mode='bicubic')  # torch.Size([16, 4, 32, 32])
        loss = self.creation(encoded_feat, latent)*1e-4 # 感知损失（与auroencoder）
        seg_pred_color = torch.stack(get_visual_img(seg_pred), dim=0).to(self.device)  # torch.Size([16, 3, 150, 150])
        bev_img_small = nn.functional.interpolate(bev_img, scale_factor=0.5, mode='bicubic')
        for index in self.layer_indexs:
            # import pdb; pdb.set_trace()
            feature_module = get_feature_module(index, self.device)
            loss += vgg16_loss(feature_module, self.creation, seg_pred_color/255.0, bev_img_small.float())
        return loss

NO_LABEL = None  # 背景类也算是语义类

class compute_losses(nn.Module):
    def __init__(self, device='cuda'):
        super(compute_losses, self).__init__()
        self.device = device
        self.seg_criterion_dice = L.DiceLoss(mode='multiclass', ignore_index=NO_LABEL).cuda()   # 分割dice损失
        self.seg_criterion = L.SoftCrossEntropyLoss(reduction='mean', smooth_factor = 0.1, ignore_index=NO_LABEL).cuda()  # 分割CE损失
        self.seg_criterion_focal = L.FocalLoss(reduction="mean", gamma=2).cuda()
        # 感知损失
        layer_indexs = [3, 8, 15, 22]
        loss_func = nn.MSELoss().to(self.device)
        self.seg_criterion_preceptual = PerceptualLoss(loss_func, layer_indexs, self.device)

    def forward(self, opt, outputs, labels, encoded_feat=None, bev_img=None):
        dice_weight = opt.dice_weight
        loss = {}
        if opt.loss_type == 'focal':
            losses = self.seg_criterion_focal(outputs, labels) + dice_weight * self.seg_criterion_dice(outputs, labels)
        else:
            losses = (self.seg_criterion(outputs, labels) + dice_weight * self.seg_criterion_dice(outputs, labels))

        # 感知损失
        if encoded_feat is None:
            preceptual_loss = 0
        else:
            preceptual_loss = self.seg_criterion_preceptual(encoded_feat, outputs, bev_img)
        loss['seg'] = losses
        loss['preceptual'] = preceptual_loss
        #return losses
        return loss

# 单通道结果图 转 彩色图
def create_visual_anno(anno):
    # print("in src/data/utils.py, anno.shape:", anno.shape)
    assert np.max(anno) <= 15, "only 15 classes are supported, add new color in label2color_dict"
    labels = {
        0: "empty_area",
        1: "drivable_area",
        2: "ped_crossing",
        3: "walkway",
        4: "movable_object",
        5: "vehicle",  
        6: "pedestrian",
        7: "mask"
    }

    label2color_dict = {
        # blue style
        0: [0, 0, 0],  #empty area
        1: [0, 49, 118], # driveable_area  blue
        2: [0, 71, 172], # ped_crossing blue
        3: [33, 70, 156], # walkway  blue
        4: [172, 86, 0], # movable_object orange  
        5: [170, 3, 18], # vehicle  brown
        6: [255, 195, 0], # pedestrian yellow
        7: [0, 0, 0] #mask black
    }

    #visual
    # anno.shape: (196,200)
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]
    return visual_anno

# 输出结果转单通道，并调用create_visual_anno转成彩色图
def get_visual_img(imgs):
    bs, c, h, w = imgs.shape
    imgs_to_show = imgs.detach().clone().cpu().data
    imgs_to_show = np.reshape(np.argmax(imgs_to_show.numpy().transpose((0,2,3,1)), axis=3), [bs, h, w]).astype(np.uint8)
    # print('in utils.py, imgs_to_show.shape: ', imgs_to_show.shape)
    color_imgs = []
    for i in range(bs):
        color_img = torch.from_numpy(create_visual_anno(imgs_to_show[i]).transpose((2, 0, 1)))
        color_imgs.append(color_img)
    # TODO if save if show
    return color_imgs

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    # TODO 检查ignore_index=0时，是否将consider_unlabeled设为False
    consider_unlabeled = True  # must consider the unlabeled, please set it to True 
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            #precision_per_class[cid] =  np.nan
            precision_per_class[cid] =  0.0  # 原来是nan，一起算mean时，结果总为nan，GS改为0
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class