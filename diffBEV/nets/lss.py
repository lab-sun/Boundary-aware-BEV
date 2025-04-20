"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from sklearn.metrics import confusion_matrix
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum
from opt import get_args
from BEVDiff_dataset_new import SimpleImageDataset, Img_ColorJitter, collate_fn
from utils import get_visual_img, compute_losses, compute_results


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # import pdb; pdb.set_trace()
        # 输入 x1:torch.Size([20, 320, 4, 11]), x2:torch.Size([20, 112, 8, 22])
        x1 = self.up(x1) # torch.Size([20, 320, 8, 22])
        # GS
        if x1.shape[-1] != x2.shape[-1]:
            x1 = F.interpolate(x1, size=(x2.shape[2], x2.shape[3]), mode='nearest')
        x1 = torch.cat([x2, x1], dim=1) # torch.Size([20, 432, 8, 22])
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x) # torch.Size([20, 512, 8, 22])
        # Depth
        x = self.depthnet(x) # torch.Size([20, 105, 8, 22])

        depth = self.get_depth_dist(x[:, :self.D]) # torch.Size([20, 41, 8, 22])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # torch.Size([20, 64, 41, 8, 22])

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem 输入x为torch.Size([20, 3, 128, 352])
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x))) # torch.Size([20, 32, 64, 176])
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2): 
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # x.shape:[torch.Size([20, 16, 64, 176]),torch.Size([20, 24, 32, 88]),torch.Size([20, 24, 32, 88])],torch.Size([20, 40, 16, 44]),torch.Size([20, 40, 16, 44])
        #          torch.Size([20, 80, 8, 22]),torch.Size([20, 80, 8, 22]),torch.Size([20, 80, 8, 22]),torch.Size([20, 112, 8, 22]),torch.Size([20, 112, 8, 22]),torch.Size([20, 112, 8, 22])
        #          torch.Size([20, 192, 4, 11]),torch.Size([20, 192, 4, 11]),torch.Size([20, 192, 4, 11]),torch.Size([20, 192, 4, 11]),torch.Size([20, 320, 4, 11])
        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # torch.Size([20, 512, 8, 22])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)  # depth: torch.Size([20, 41, 8, 22]); x:torch.Size([20, 64, 41, 8, 22])

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )
        self.conv2_down = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv3_down = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv2_up = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv3_up = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)

    def encoder(self, x):
        # 输入x：torch.Size([20, 64, 150, 150])
        x = self.conv1(x) # torch.Size([20, 64, 75, 75])
        x = self.bn1(x)
        x = self.relu(x)
        # added
        x = self.conv2_down(x) # torch.Size([20, 32, 75, 75])
        x = self.conv3_down(x) # torch.Size([20, 32, 38, 38])
        return x

    def decoder(self,x):
        # 输入x:torch.Size([20, 32, 38, 38])
        x = self.conv2_up(x) # torch.Size([20, 64, 38, 38]) 
        x = self.conv3_up(x) # torch.Size([20, 64, 75, 75])
        x1 = self.layer1(x) # torch.Size([20, 64, 75, 75])
        x = self.layer2(x1) # torch.Size([20, 128, 38, 38])
        x = self.layer3(x)  # torch.Size([20, 256, 19, 19])

        x = self.up1(x, x1) # torch.Size([20, 256, 75, 75])
        x = self.up2(x) # torch.Size([20, 7, 150, 150])
        return x
        

    
    # def forward(self, x):
    #     import pdb; pdb.set_trace()
    #     # 输入x：torch.Size([4, 64, 200, 200])
    #     x = self.conv1(x) # torch.Size([4, 64, 100, 100])
    #     x = self.bn1(x)
    #     x = self.relu(x)

    #     x1 = self.layer1(x) # torch.Size([4, 64, 100, 100])
    #     x = self.layer2(x1) # torch.Size([4, 128, 50, 50])
    #     x = self.layer3(x)  # torch.Size([4, 256, 25, 25])

    #     x = self.up1(x, x1) # torch.Size([4, 256, 100, 100])
    #     x = self.up2(x) # torch.Size([4, 1, 200, 200])

    #     return x
    def forward(self,x):
        x_encoded = self.encoder(x)  # torch.Size([20, 32, 75, 75])
        x_pred = self.decoder(x_encoded)
        return x_pred

# log_root = opt.text_logger_path
# if not os.path.exists(log_root):
#     os.makedirs(log_root)

# text_log = open(os.path.join(log_root, 'lss.csv'), 'w')
class LiftSplatShoot(pl.LightningModule):
    def __init__(self, 
                 #grid_conf, 
                 #data_aug_conf, 
                 #outC,
                 opt,
                 train_dataset,
                 valid_dataset=None, 
                 batch_size=1,
                 lr=1e-4,
                 num_classes=7):
        super(LiftSplatShoot, self).__init__()
        self.opt = opt
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size
        self.num_classes = num_classes

        self.grid_conf = opt.grid_conf
        self.data_aug_conf = opt.data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                                )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=self.num_classes)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        self.seg_loss_fn = compute_losses()

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        
        B, N, _ = trans.shape  # torch.Size([bs, n_cams, 3])

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3) # B:batch_size, N:num_cam

        return points # torch.Size([20, 1, 41, 8, 22, 3])

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        # x输入 torch.Size([20, 1, 3, 128, 352])
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW) # torch.Size([20, 3, 128, 352])
        x = self.camencode(x) # torch.Size([20, 64, 41, 8, 22]) 
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  # torch.Size([20, 1, 64, 41, 8, 22])
        x = x.permute(0, 1, 3, 4, 5, 2)  # torch.Size([20, 1, 41, 8, 22, 64])

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  # torch.Size([20, 1, 41, 8, 22, 64])
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)  # torch.Size([144320, 64])

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # torch.Size([20, 1, 41, 8, 22, 3])
        geom_feats = geom_feats.view(Nprime, 3)  # torch.Size([144320, 3])
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # torch.Size([144320, 1])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # torch.Size([144320, 4])

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])  # torch.Size([144320])
        x = x[kept]  # torch.Size([112640, 64])
        geom_feats = geom_feats[kept]  # torch.Size([112640, 4])

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # torch.Size([112640])
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # x：torch.Size([112640, 64]), geom_feats：torch.Size([112640, 4]), ranks：torch.Size([112640])

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) # x:torch.Size([13700, 64])  geom_feats:torch.Size([13700, 4])

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device) # torch.Size([20, 64, 1, 150, 150])
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # torch.Size([4, 64, 200, 200])

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # torch.Size([20, 1, 41, 8, 22, 3])
        x = self.get_cam_feats(x) # torch.Size([20, 1, 41, 8, 22, 64])
        x = self.voxel_pooling(geom, x) # torch.Size([20, 64, 150, 150])

        return x

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)
        else:
            return None
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
    
    def encoder_step(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans) # torch.Size([4, 64, 200, 200])
        # x = self.bevencode(x) # torch.Size([20, 7, 150, 150]) TODO 检查!在BevEncode的forward时输出upsample之前的特征
        x = self.bevencode.encoder(x)
        return x

    def decoder_step(self, x):
        x = self.bevencode.decoder(x)        
        # H, W = 150, 150
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x_encoder = self.encoder_step(x, rots, trans, intrins, post_rots, post_trans)  # torch.Size([20, 32, 75, 75])
        x_pred = self.decoder_step(x_encoder) # torch.Size([20, 7, 150, 150])
        return x_pred

    def training_step(self, batch, batch_idx):
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        FV_image = images
        BEV_labels = bev_labels.to(torch.int64)

        x_encoder = self.encoder_step(FV_image, rots, trans, intrins, post_rots, post_trans)
        seg_pred = self.decoder_step(x_encoder)

        # 计算loss
        seg_loss = self.seg_loss_fn(self.opt, seg_pred, BEV_labels)
        loss = seg_loss
        self.log('Step_loss/train_seg', seg_loss, on_step=True, prog_bar=True, logger=True)
        # 计算混淆矩阵
        label = BEV_labels.cpu().numpy().squeeze().flatten()
        pred = seg_pred.argmax(1).cpu().numpy().squeeze().flatten()
        conf = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4,5,6])

        epoch_dictionary={
            "loss": loss,
            "conf": conf,
        }
        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
        # (bev_labels, images, bev_images, img_names, scene_names, rots, trans, intrins, post_rots, post_trans) = batch
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        FV_image = images
        BEV_labels = bev_labels.to(torch.int64)

        x_encoder = self.encoder_step(FV_image, rots, trans, intrins, post_rots, post_trans)
        seg_pred = self.decoder_step(x_encoder)

        # 计算loss
        seg_loss = self.seg_loss_fn(self.opt, seg_pred, BEV_labels)
        loss = seg_loss
        self.log('Step_loss/val_seg', seg_loss, on_step=True, prog_bar=True, logger=True)
        
        # 计算混淆矩阵
        label = BEV_labels.cpu().numpy().squeeze().flatten()
        pred = seg_pred.argmax(1).cpu().numpy().squeeze().flatten()
        conf = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4,5,6])
        
        epoch_dictionary={
            "loss": loss,
            "conf": conf,
        }
        return epoch_dictionary

    def showPred(self, FV_img, pred_img, BEV_img, mode):
        self.logger.experiment.add_image('{}/FV_image'.format(mode), torch.Tensor.cpu(FV_img[0].squeeze(0)),self.current_epoch,dataformats="CHW")
        
        pred_color = get_visual_img(imgs = pred_img) # pred_img是list，pred_img[0].shape: torch.Size([3, 150, 150])
        self.logger.experiment.add_image('{}/pred_img'.format(mode), torch.Tensor.cpu(pred_color[0]), self.current_epoch, dataformats="CHW")
        
        self.logger.experiment.add_image('{}/BEV_image'.format(mode), torch.Tensor.cpu(BEV_img[0]),self.current_epoch,dataformats="CHW")
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        conf_total = np.stack([x['conf'] for x in outputs]).sum(axis=0)
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()

        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Metric_Train/average_IoU', average_IoU, self.current_epoch)
        self.logger.experiment.add_scalar('Metric_Train/average_precision', average_precision, self.current_epoch)
        
        # 打印结果
        print_output = ("Train Epoch: [{epoch}/{total_epoch}] | loss: {loss:.8f} | mIoU: {mIoU:.8f} |  mAP: {mAP:.8f} | mRecall: {mRecall:.8f} |".format(
                        epoch=self.current_epoch, total_epoch=self.opt.n_epochs_lss, loss=avg_loss, mIoU=average_IoU, mAP=average_precision, mRecall=average_recall))
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}
        for i in range(len(iou_per_class)):  
            precision_record[self.opt.label_list[i]] = precision_per_class[i]
            recall_record[self.opt.label_list[i]] = recall_per_class[i]
            iou_record[self.opt.label_list[i]] = iou_per_class[i]
        metirc_each_class = ("precision for each class: {} | recall for each class: {} | iou for each class: {}\n".format(precision_record, recall_record, iou_record))

        self.logger.experiment.add_text('Train/loss', print_output, self.current_epoch)  # 结果写入文件
        self.logger.experiment.add_text('Train/logger', metirc_each_class, self.current_epoch)

        # 检查images是几张图像？ 是一个batch，送入到showPred后只显示第一张
        train_dataloader = self.train_dataloader()
        batch = next(iter(train_dataloader))
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        with torch.no_grad():
            out = self(images.to(self.device), rots.to(self.device), trans.to(self.device), intrins.to(self.device), post_rots.to(self.device), post_trans.to(self.device))
        self.showPred(images, out, bev_images, mode='Train')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        conf_total = np.stack([x['conf'] for x in outputs]).sum(axis=0)
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()

        self.log("val_iou", average_IoU, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Metric_Val/average_IoU', average_IoU, self.current_epoch)
        self.logger.experiment.add_scalar('Metric_Val/average_precision', average_precision, self.current_epoch)
        
        # 打印结果
        print_output = ("Val Epoch: [{epoch}/{total_epoch}] | loss: {loss:.8f} | mIoU: {mIoU:.8f} |  mAP: {mAP:.8f} | mRecall: {mRecall:.8f} |".format(
                        epoch=self.current_epoch, total_epoch=self.opt.n_epochs_lss, loss=avg_loss, mIoU=average_IoU, mAP=average_precision, mRecall=average_recall))
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}
        for i in range(len(iou_per_class)):  
            precision_record[self.opt.label_list[i]] = precision_per_class[i]
            recall_record[self.opt.label_list[i]] = recall_per_class[i]
            iou_record[self.opt.label_list[i]] = iou_per_class[i]
        metirc_each_class = ("precision for each class: {} | recall for each class: {} | iou for each class: {}".format(precision_record, recall_record, iou_record))

        self.logger.experiment.add_text('Val/loss', print_output, self.current_epoch)  # 结果写入文件
        self.logger.experiment.add_text('Val/logger', metirc_each_class , self.current_epoch)

        # 检查images是几张图像？ 是一个batch，送入到showPred后只显示第一张
        val_dataloader = self.val_dataloader()
        batch = next(iter(val_dataloader))
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        with torch.no_grad():
            out = self(images.to(self.device), rots.to(self.device), trans.to(self.device), intrins.to(self.device), post_rots.to(self.device), post_trans.to(self.device)) # x, rots, trans, intrins, post_rots, post_trans
        self.showPred(images, out, bev_images, mode='Val')




def compile_model(opt, grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
