import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.utils
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import sys
# sys.path.append('./')
#from nets.xception import xception
# from mobilenetv2 import mobilenetv2
from diffBEV.nets.mobilenetv2 import mobilenetv2
# sys.path.append('../../')
from diffBEV.opt import get_args
from diffBEV.dataset.BEVDiff_dataset import SimpleImageDataset, Img_ColorJitter, collate_fn
from diffBEV.utils import get_visual_img, compute_losses, compute_results


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 

#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result


class DeepLab(pl.LightningModule):
    def __init__(self, 
                 opt,
                 train_dataset,
                 valid_dataset=None, 
                 batch_size=1,
                 lr=1e-4,
                 num_classes=7, 
                 backbone="mobilenet", 
                 pretrained=True, 
                 downsample_factor=16):
        super(DeepLab, self).__init__()
        self.opt = opt
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size

        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            # self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            # in_channels = 2048
            # low_level_channels = 256
            pass
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        # added
        self.encoder_conv = nn.Conv2d(256, 32, 1, stride=1)
        self.decoeder_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(32, 16, kernel_size=1, padding=0),
        )

        self.cls_conv = nn.Conv2d(16, num_classes, 1, stride=1)
        #self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        self.seg_loss_fn = compute_losses()

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
                              shuffle=False,
                              num_workers=4,
                              collate_fn=collate_fn)
        else:
            return None
    
    def configure_optimizers(self):
        return  torch.optim.AdamW(self.parameters(), lr=self.lr)

    def encoder(self, x): #将forward拆成encoder和decoder
        # H, W = x.size(2), x.size(3)
        H, W = 150, 150
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)  # low_level_features.shape:torch.Size([4, 24, 64, 128]); x.shape:torch.Size([4, 320, 16, 32])
        x = self.aspp(x) # torch.Size([4, 256, 16, 32])
        low_level_features = self.shortcut_conv(low_level_features)  # torch.Size([4, 48, 64, 128])
        # TODO 方案一：输出不同层级的特征  方案二：low_level_features先降维成x的大小，再送入decoder
        # 方案二：
        low_level_features_resized = F.interpolate(low_level_features, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True) # torch.Size([2, 48, 16, 32])
        x = self.cat_conv(torch.cat((x, low_level_features_resized), dim=1)) # torch.Size([2, 256, 16, 32])
        x = F.interpolate(x, size=(x.size(3), x.size(3)), mode='bilinear', align_corners=True)  # torch.Size([2, 256, 32, 32])
        x = self.encoder_conv(x)  # torch.Size([2, 32, 32, 32])
        return x 

    def decoder(self, x):
        H, W = 150, 150
        x = self.decoeder_up(x) # torch.Size([2, 16, 128, 128])
        x = self.cls_conv(x) # torch.Size([2, 7, 128, 128])
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)  # torch.Size([2, 7, 150, 150])
        return x


    def forward_step(self, x):  # 原来的forward()
        # H, W = x.size(2), x.size(3)
        H, W = 150, 150
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)  # low_level_features.shape:torch.Size([4, 24, 64, 128]); x.shape:torch.Size([4, 320, 16, 32])
        x = self.aspp(x) # torch.Size([4, 256, 16, 32])
        low_level_features = self.shortcut_conv(low_level_features)  # torch.Size([4, 48, 64, 128])
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)  # torch.Size([4, 256, 64, 128])
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))  # torch.Size([4, 256, 64, 128])
        x = self.cls_conv(x)  # torch.Size([4, 7, 64, 128])
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)  # torch.Size([4, 7, 150, 150])
        return x

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = self.forward_step(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        (bev_labels, images, bev_images, img_names, scene_names) = batch
        FV_image = images
        BEV_labels = bev_labels.to(torch.int64)

        #seg_pred = self.forward_step(FV_image)
        x = self.encoder(FV_image)
        seg_pred = self.decoder(x)
        
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
        (bev_labels, images, bev_images, img_names, scene_names) = batch
        FV_image = images
        BEV_labels = bev_labels.to(torch.int64)

        #seg_pred = self.forward_step(FV_image)
        x = self.encoder(FV_image)
        seg_pred = self.decoder(x)
        
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

        n_show = 20
        num = min(n_show, FV_img.shape[0])

        grid_image = torchvision.utils.make_grid(FV_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/FV_image'.format(mode), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")
        
        pred_color = get_visual_img(imgs = pred_img) # pred_img是list，pred_img[0].shape: torch.Size([3, 150, 150])
        grid_image = torchvision.utils.make_grid(pred_color[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/pred_img'.format(mode), torch.Tensor.cpu(grid_image), self.current_epoch, dataformats="CHW")
        
        grid_image = torchvision.utils.make_grid(BEV_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/BEV_image'.format(mode), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")

    
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

        
        # 显示图像
        train_dataloader = self.train_dataloader()
        batch = next(iter(train_dataloader))
        (bev_labels, images, bev_images, img_names, scene_names) = batch
        with torch.no_grad():
            out = self(images.to(self.device))
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

        
        # 显示图像
        val_dataloader = self.val_dataloader()
        batch = next(iter(val_dataloader))
        (bev_labels, images, bev_images, img_names, scene_names) = batch
        with torch.no_grad():
            out = self(images.to(self.device))
        self.showPred(images, out, bev_images, mode='Val')

if __name__ == '__main__':
    opt = get_args()
    train_csv_file = '../../data/nuScenes/train.csv'
    train_ds = SimpleImageDataset(train_csv_file, transform=Img_ColorJitter())
    
    model = DeepLab(opt, train_ds, lr=1e-4, batch_size=4, num_classes=7, backbone="mobilenet", downsample_factor=16, pretrained=True)
    print(model)
    x_input = torch.rand((4,3,256,512))
    output = model(x_input)
    print('output.shape: ', output.shape)