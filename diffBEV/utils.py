import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2
from pytorch_toolbelt import losses as L

import torch.optim as optim

NO_LABEL = None  # 背景类也算是语义类

# gt_mask进行onehot操作
def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask==i for i in range(num_classes)]
    onehot = np.stack(_mask, axis=0)
    return onehot

# 提取部分类别边界
def calc_edge(x, mode='canny'):
    x = np.uint8(x)
    edge = cv2.Canny(image=x, threshold1=0, threshold2=0)
    return edge

# 计算到边界距离
def calc_distance_map(x, mode='l2'):
    # Convert the data to grayscale [0,255]
    binary_x = 1 - np.uint8((x-x.min())/(x.max()-x.min()))

    if mode.lower() == 'l1':
        dt_mode = cv2.DIST_L1
    elif mode.lower() == 'l2':
        dt_mode = cv2.DIST_L2
    else:
        raise ValueError("<mode> must be 'l1' or 'l2'.")

    # Calculate the distance transform
    dist_transform= cv2.distanceTransform(binary_x, dt_mode, 0)
    return dist_transform

# 计算边界注意力
def calc_boundary_att(batch_gt, batch_t, T, gamma=1.5, *args, **kwargs):
    """
    Parameters:
        - batch_gt : [tensor] |-> input data matrix
        - batch_t : [tensor] |-> current timestep
        - T       : [int]    |-> maximum timesteps
        - gamma   : [float]  |-> sharpness [default is 1.5]
    Output:
        - boundary_att
    """
    # boundary thickness (max value thickness)
    bt = np.round(batch_gt.shape[-1]*0.01)

    X = batch_gt.detach().cpu().numpy()  # torch.Size([bs, 150, 150])
    # X = (X-X.min())/(X.max()-X.min()) # normalize because X is in range ~ (-1, 1)
    device = batch_gt.get_device()

    atts = []
    # import pdb; pdb.set_trace()
    for x, t in zip(X, batch_t): # x.shape:(150, 150)
        # 提取特定类别
        gt_onehot = mask_to_onehot(x, num_classes=7)  # (7, 150, 150)
        obj_onehot = 1*(gt_onehot[4] + gt_onehot[5] + gt_onehot[6]) # (150, 150)
        if obj_onehot.sum().item() > X.shape[-1]**2/100.: # foreground area is bigger than 1% of the image
            obj_onehot = (obj_onehot-obj_onehot.min())/(obj_onehot.max()-obj_onehot.min())
            edge = calc_edge(obj_onehot)  # (150, 150)
            dist_x = calc_distance_map(edge, mode='l2')
            tmp = X.shape[-1]*1.1415 - dist_x
            normalized_inv_dist_x = (tmp-tmp.min())/ (tmp.max()-tmp.min())

            t_p = ((gamma*(T-t.item()))/T)**gamma
            att = normalized_inv_dist_x**t_p  
        else:
            att = np.ones_like(x)
        atts.append(att)

    # import pdb; pdb.set_trace()
    atts = np.array(atts)
    W = torch.stack([torch.from_numpy(att) for att in atts], dim=0)
    if device != -1:
        W = W.to(device)

    W = torch.unsqueeze(W, dim=1) # torch.Size([8, 1, 150, 150]) # TODO 当t小的时候，边界处的权重大，其他地方权重很小？ 尝试（1+att）* seg_loss
    return W

class compute_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_losses, self).__init__()
        class_weights = torch.Tensor([3.17794119, 2.5617616, 23.64493676,  7.09298426, 45.14709186, 18.77214296, 48.18801878])
        self.device = device
        self.seg_criterion_dice = L.DiceLoss(mode='multiclass', ignore_index=NO_LABEL).cuda()   # 分割dice损失
        # self.seg_criterion = L.SoftCrossEntropyLoss(reduction='mean', smooth_factor = 0.1, ignore_index=NO_LABEL).cuda()  # 分割CE损失
        self.seg_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').cuda()
        self.seg_criterion_focal = L.FocalLoss(reduction="mean", gamma=2).cuda()
        # 边界损失
        self.calc_root_loss = lambda p, t: torch.abs(p-t)**2

    def forward(self, opt, outputs, labels, t=None, T=None):
        """
        outputs: the predict bev maps. torch.Size([bs, 7, 150, 150])
        labels: the gt mask. torch.Size([8, 150, 150])
        """
        dice_weight = opt.dice_weight
        if opt.loss_type == 'focal':
            losses = self.seg_criterion_focal(outputs, labels) + dice_weight * self.seg_criterion_dice(outputs, labels)
        else:
            losses = (self.seg_criterion(outputs, labels) + dice_weight * self.seg_criterion_dice(outputs, labels))
            # losses = self.seg_criterion(outputs, labels)
        
        if opt.if_BoundaryLoss:
            # import pdb; pdb.set_trace()
            boundary_att = calc_boundary_att(labels, t, T=T, gamma=opt.boundaryLoss_gamma)
            pred = outputs.argmax(1)
            root_loss = self.calc_root_loss(pred, labels)
            boundary_loss = (boundary_att * root_loss).double().mean()
            losses = boundary_loss + losses

        return losses

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
    imgs_to_show = imgs.clone().cpu().data
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

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

# 用相机内参计算v-shaped mask
def get_mask(instrinsics_batch, pred, gt, opt):
    masked_gt_list = []
    masked_pred_list = []
    pred = pred.argmax(1).cpu().numpy().squeeze()
    for i in range(instrinsics_batch.shape[0]):
        instrinsics = instrinsics_batch[i].cpu().numpy().squeeze()

        # Get calibration parameters
        fu, cu = instrinsics[0, 0], instrinsics[0, 2]

        # Construct a grid of image coordinates
        x1, z1, x2, z2 = opt.data_aug_conf['map_extents']
        x, z = np.arange(x1, x2, opt.data_aug_conf['map_resolution']), np.arange(z1, z2, opt.data_aug_conf['map_resolution'])
        ucoords = x / z[:, None] * fu + cu
        mask_ = (ucoords >= 0) & (ucoords < opt.data_aug_conf['W'])
        mask = mask_.reshape(mask_.size)
        mask = mask[::-1] * 255
        # mask = mask.reshape(mask_.shape) * 255

        current_gt = gt[i].cpu().numpy().squeeze()
        current_nn = np.reshape(pred[i], [150, 150])
        valid_FOV_index = mask != 0
        valid_index = current_gt.reshape(-1) != 0
        valid_index = valid_index * valid_FOV_index

        current_gt_ = current_gt.copy().reshape(-1)
        current_gt_ = np.where(valid_index, current_gt_, 0)
        # current_gt_ = current_gt_.reshape(current_gt.shape) # (150, 150)

        current_nn_ = current_nn.copy().reshape(-1)
        current_nn_ = np.where(valid_index, current_nn_, 0) # (22500,)
        # current_nn_ = current_nn_.reshape(current_nn.shape)
        
        # current_gt = current_gt.reshape(-1)[valid_index]
        # current_nn = current_nn.reshape(-1)[valid_index]

        current_gt_ = current_gt_.reshape(1, -1)
        current_nn_ = current_nn_.reshape(1, -1)

        masked_gt_list.append(current_gt_)
        masked_pred_list.append(current_nn_)
    
    # import pdb; pdb.set_trace()
    masked_gt = np.stack(masked_gt_list, axis=0).flatten()
    masked_nn = np.stack(masked_pred_list, axis=0).flatten()

    return masked_gt, masked_nn