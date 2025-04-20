import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from diffBEV.src.autoencoder_lss import AutoencoderKL
from diffBEV.nets.deeplabv3_plus_new import DeepLab
from diffBEV.nets.attention_GPU3 import SpatialTransformer, CrossTransformer  # GPU3
from diffBEV.src.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder, Encoder_condition

from diffBEV.utils import get_visual_img, compute_losses, compute_results, CosineWarmupScheduler, get_mask

from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.utils
from diffBEV.dataset.BEVDiff_dataset_new import SimpleImageDataset, collate_fn

import tqdm
import argparse
import skimage
#from guided_diffusion import dist_util_GPU1 as dist_util # # GPU1
from guided_diffusion import dist_util # GPU3
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

import time as TIME
import os

create_time = TIME.strftime("%Y-%m-%d-%H-%M", TIME.localtime())
save_root = './sampling_img/DiffBEV_GPU3_crossTrans'
text_save_root = os.path.join('./logs/DiffBEV_GPU3_crossTrans', create_time)
if not os.path.exists(text_save_root):
    os.makedirs(text_save_root)
text_logger = open(os.path.join(text_save_root, 'diffBEV.csv'), 'w')


class AutoEncoder(nn.Module):
    def __init__(self,
                opt,
                #model_type= "stabilityai/sd-vae-ft-ema"#@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                pretrain_weigth = "./pretrain_weights/ae/2023-10-16-10-40/ae_epoch18.pth", 
                # TODO ablation study
                # pretrain_weigth = "./pretrain_weights/ae_ablation/2023-12-20-05-38(64*64*4-4)/ae_epoch4.pth",
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        #self.model=AutoencoderKL.from_pretrained(model_type).to('cuda')
        self.model = torch.load(pretrain_weigth)
        train_csv_file = './data/nuScenes/train_new.data'
        train_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=train_csv_file, transform=None)
        cond_pretrain_weight = './pretrain_weights/deeplab/2023-10-29-17-04/best-deeplab-epoch48-val_iou0.75.ckpt'  # lss model
        # # TODO ablation study
        # cond_pretrain_weight = './pretrain_weights/deeplab_ablation/2023-12-21-03-07(64*64*4)/best-deeplab-epoch20-val_iou0.35.ckpt'  # lss model
        self.model_cond = DeepLab.load_from_checkpoint(cond_pretrain_weight, opt=opt, train_dataset=train_ds, downsample_factor=16).to('cuda') # TODO ablation study downsample_factor=16
        # 原来的atten
        self.corss_attn = SpatialTransformer(in_channels=320, n_heads=8, d_head=40, depth=1, context_dim=320).to('cuda')
        # 新的atten
        # self.corss_attn = CrossTransformer(in_dim=64)

        
    def forward(self,input):
        return self.model(input).sample
    
    def encode(self,input,mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    def encode_cond(self,input,mode=False):
        dist=self.model_cond.encoder(input)  # dist.max()的数量级为0-10
        return dist
    
    # 加上cross attention
    def attn(self, input, cond):
        x = self.corss_attn(input, cond)
        return x
    
    def decode(self, x, cond=None):
        #return self.model.decode(input, H=150, W=150).sample
        #import pdb; pdb.set_trace() 
        # 输入x.shape:torch.Size([6, 4, 32, 32]), cond.shape:torch.Size([6, 4, 32, 32])
        if cond is not None:
            # 原来的SpatialTransformer
            x = self.attn(x, cond)
            # 修改后的CrossTransformer
            # x = self.attn(cond, x)
            
            # x = x + cond # TODO 先不加试试
            x = self.model_cond.decoder(x)
        else:
            x = self.model_cond.decoder(x)
        return x
        

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        """
            This is a simplified version of Latent Diffusion        
        """        
        
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        
        self.ae=AutoEncoder()
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,256,256)).shape[1]
        self.args = self.create_argparser().parse_args()
        dist_util.setup_dist()
        # self.model=DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
        #                                                 condition_channels=self.latent_dim,
        #                                                 num_timesteps=num_timesteps)
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        self.model.to(dist_util.dev())
        self.schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, self.diffusion,  maxt=1000)

    @torch.no_grad()
    def forward(self,*args,**kwargs):  # 返回decode以后的分割pred
    #def forward(self, x, target, *args,**kwargs):  # def forward(self,condition,*args,**kwargs):
        # 问题： *args是什么？传入model的input是什么？dataloader返回的是一个字典  //是在test时调用
        #return self.output_T(self.model(*args,**kwargs))
        # import pdb; pdb.set_trace()
        result = self.output_T(self.ae.decode(self.model(*args,**kwargs)/self.latent_scale_factor))
        # return self.output_T(self.ae.decode(self.model(*args,**kwargs)/self.latent_scale_factor))
        import pdb; pdb.set_trace()
        return result
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):   
        
        # import pdb; pdb.set_trace()
        # output = batch['bev_image']
        # (images, bev_labels, bev_images, img_names, scene_names) = batch
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        output = bev_images

        latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
        # latents=self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('train_loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        
        # condition = batch['image']
        # output = batch['bev_image']
        # (images, bev_labels, bev_images, img_names, scene_names) = batch
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        output = bev_images
        latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
        # latents=self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('val_loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True,
                          collate_fn=collate_fn)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4,
                              drop_last=True,
                              collate_fn=collate_fn)
        else:
            return None

    
    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)  # 只对diffusion部分参数求梯度
        # return  torch.optim.AdamW(self.parameters(), lr=self.lr) # 对模型的所有参数求梯度

    # GS 
    def training_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        # print('in BEVDiff_LatentDiffusion.py, in training_epoch_end, type of outputs:{}, len: {}'.format(type(outputs), len(outputs)))
    
class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 opt,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 ):
        pl.LightningModule.__init__(self)
        self.opt = opt
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = self.opt.lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=self.opt.batch_size
        self.sample_interval = 1
        self.num_timesteps = num_timesteps
        
        self.ae=AutoEncoder(opt=self.opt)
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,256,256).to('cuda')).shape[1]  # 4
        
        dist_util.setup_dist()
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.opt.defaults, model_and_diffusion_defaults().keys())
        )
        self.model.to(dist_util.dev())
        self.schedule_sampler = create_named_schedule_sampler(self.opt.defaults['schedule_sampler'], self.diffusion,  maxt=self.num_timesteps)
        
        # diffusion trainloop
        self.trainloop = TrainLoop(
                model=self.model,
                diffusion=self.diffusion,
                classifier=None,
                data=None, # 原版没有用到
                dataloader=None,  # 原版只有run_loop函数用到
                batch_size=self.opt.batch_size,
                microbatch=self.opt.defaults['microbatch'],
                lr=self.opt.defaults['lr'],
                ema_rate=self.opt.defaults['ema_rate'],
                log_interval=self.opt.defaults['log_interval'],
                save_interval=self.opt.defaults['save_interval'],
                resume_checkpoint=self.opt.defaults['resume_checkpoint'],
                use_fp16=self.opt.defaults['use_fp16'],
                fp16_scale_growth=self.opt.defaults['fp16_scale_growth'],
                schedule_sampler=self.schedule_sampler,
                weight_decay=self.opt.defaults['weight_decay'],
                lr_anneal_steps=self.opt.defaults['lr_anneal_steps'],
            )
        # sampler
        self.sample_fn = (
                self.diffusion.p_sample_loop_known if not self.opt.defaults['use_ddim'] else self.diffusion.ddim_sample_loop_known
            )
        
        # # TODO 添加对FV的encoder, self.ae.model_cond是提取FV的网络
        self.seg_loss_fn = compute_losses()

        self.tensorboard_flag = True

        
    def configure_optimizers(self):        
        # param_list = list(self.model.parameters())+list(self.ae.model_cond.decoder_in.parameters())+\
        #              list(self.ae.model_cond.decoeder_up.parameters())+list(self.ae.model_cond.cls_conv.parameters()) + list(self.ae.corss_attn.parameters())
        # return torch.optim.AdamW(param_list, lr=self.args.lr, weight_decay=self.args.weight_decay)

        
        decoder_param = list(self.ae.model_cond.decoder_in.parameters())+list(self.ae.model_cond.decoeder_up.parameters())+list(self.ae.model_cond.cls_conv.parameters())
        param_group = [
            {'params': list(self.model.parameters()), 'weight_decay': 0.0, 'lr': self.opt.defaults['lr']}, # 1e-4
            {'params': decoder_param, 'weight_decay': self.opt.defaults['weight_decay'], 'lr': 0.001},
            {'params': list(self.ae.corss_attn.parameters()), 'weight_decay': self.opt.defaults['weight_decay'], 'lr': 0.005},
        ]
        optimizer = torch.optim.AdamW(param_group, lr=self.opt.defaults['lr'], weight_decay=self.opt.defaults['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=50, max_iters=300)
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def forward(self,condition, mode='train', *args,**kwargs):  # 在训练完设定的step步后，采样时调用
        self.tensorboard_flag = True
        condition = condition.squeeze(dim=1).float()
        condition_latent=self.ae.encode_cond(condition.to(self.device)).detach() # condition_latent:torch.Size([4, 4, 32, 32])
        noise = torch.randn_like(condition_latent[:, -4:, ...])  # noise latent 是 （bs, 4, 32, 32）
        # cond在前，noise在后 
        sample_input = torch.cat([condition_latent,noise],1).to(self.device) #torch.Size([bs, 8, 32, 32])

        # sampling   
        sample, x_noisy, org, pred_xstarts = self.sample_fn(
                self.model,
                # TODO ablation study
                (self.opt.batch_size, 4, 32, 32), 
                # (self.opt.batch_size, 4, 64, 64), 
                sample_input,
                num_timesteps=self.num_timesteps,
                clip_denoised=self.opt.defaults['clip_denoised'],
                model_kwargs={},
            ) # sample: torch.Size([bs, 4, 32, 32]); 
        self.showIntermediate(sample, 'sample', mode='{}_sampling'.format(mode))
        for i in range(len(pred_xstarts)):
            self.showIntermediate(pred_xstarts[i], 'pred_xstart_step{}'.format(i*100), mode='{}_sampling'.format(mode))
        
        # 保存Tensor
        # tensor_save_path = os.path.join(save_root, create_time) + '/sample_{}.pth'.format(self.current_epoch)
        # pred_xstarts = torch.stack(pred_xstarts, dim=0)  # 将每100步预测的pred_xstart拼成一个tensor，顺序：t=900,800,700...最后是x_0
        # torch.save(pred_xstarts, tensor_save_path)

        # 验证1
        # sample = sample+condition_latent
        # return self.ae.decode(sample)
        # 验证2 TODO 或者sample = sampel + condition_latent后在做attn
        return self.ae.decode(sample, condition_latent) # sample和condition_latent先做attn，在decode
    
    def training_step(self, batch, batch_idx):  # 先调用
        # import pdb; pdb.set_trace()
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        bev_labels = bev_labels.to(torch.int64)
        condition = images.squeeze(dim=1).float()  # torch.Size([1, 3, 256, 512])
        output = bev_images.float()  # torch.Size([1, 3, 150, 150])
                
        with torch.no_grad(): # 不求导？是用已经训练好的模型做autoencoder？再在forward中对diffusion模型进行参数训练？
            latents=self.ae.encode(output, mode=True).detach()  # torch.Size([48, 4, 37, 37])
            latent_scale_factor = latents.std()
            latents = latents / latent_scale_factor # *self.latent_scale_factor

            
            # 保存tensor
            # if self.tensorboard_flag==True:
            #     tensor_save_path = os.path.join(save_root, create_time) + '/train_latent_{}.pth'.format(self.current_epoch)
            #     torch.save(latents, tensor_save_path)
            
            # import pdb; pdb.set_trace()
            latents_condition=self.ae.encode_cond(condition).detach() # torch.Size([48, 4, 32, 32])
            latents = torch.nn.functional.interpolate(latents, latents_condition.shape[-2:], mode='bicubic')

        
        input_for_diff = (latents, latents_condition)
        noise_loss, pred_noise, x_0_pred, t = self.trainloop.run_loop(input_for_diff) # x_0_pred.shape:torch.Size([bs, 4, 18, 18])
        
        # 验证1
        # sample = x_0_pred + latents_condition
        # seg_pred = self.ae.decode(sample)
        # 验证2
        seg_pred = self.ae.decode(x_0_pred, latents_condition)
        # 分割损失 # TODO 与GPU3不同，GPU3计算的是用时间t作为权重计算seg_loss
        if self.opt.if_BoundaryLoss:
            # 加上了boundary_loss, 在运行程序时要加上 --if_BoundaryLoss
            seg_loss = self.seg_loss_fn(self.opt, seg_pred, bev_labels, t, self.num_timesteps)
        else:
            # 没有boundary_loss
            seg_loss = self.seg_loss_fn(self.opt, seg_pred, bev_labels)

        # 查看decoder的梯度  # 只有decoder有梯度
        # import pdb; pdb.set_trace()
        # for name, parms in self.ae.named_parameters():
        #     if  'model_cond' in name:  # 'model_cond.decoder_in'
        #         import pdb; pdb.set_trace()
        #         print('-->name:', name)
        #         # print('-->para:', parms)
        #         print('-->grad_requires: ', parms.requires_grad)
        #         print('-->grad_value:', parms.grad)
        #         print('===')
        #         break
        
        if self.current_epoch % 10 == 0 and self.tensorboard_flag==True:
            self.showIntermediate(latents*latent_scale_factor, 'latents', mode='train')
            self.showIntermediate(latents_condition, 'latents_cond', mode='train')
            self.showIntermediate(pred_noise, 'pred_noise', t=t, mode='train')
            self.showIntermediate(x_0_pred, 'x_0_pred', t=t, mode='train')
            self.tensorboard_flag=False
        loss = noise_loss+seg_loss
                
        self.log('Step_loss/train',loss, on_step=True, prog_bar=True, logger=True)
        self.log('Step_loss/train_noise', noise_loss, on_step=True, prog_bar=True, logger=True)
        self.log('Step_loss/train_seg', seg_loss, on_step=True, prog_bar=True, logger=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):   
        # import pdb; pdb.set_trace()  
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        bev_labels = bev_labels.to(torch.int64)  # torch.Size([bs, 150, 150])
        condition = images.squeeze(dim=1).float()  # torch.Size([bs, 3, 256, 512])
        output = bev_images.float()  # torch.Size([bs, 3, 300, 300]) 用大尺寸的label作为输入
        # bev_labels = bev_labels.to(torch.int64)
                
        with torch.no_grad(): # 不求导？是用已经训练好的模型做autoencoder？再在forward中对diffusion模型进行参数训练？
            latents=self.ae.encode(output, mode=True).detach()
            latent_scale_factor = latents.std()
            latents = latents / latent_scale_factor
            latents_condition=self.ae.encode_cond(condition).detach()
            latents = torch.nn.functional.interpolate(latents, latents_condition.shape[-2:], mode='bicubic')

        input_for_diff = (latents, latents_condition)
        noise_loss, pred_noise, x_0_pred, t  = self.trainloop.run_loop(input_for_diff) # x_0_pred.shape:torch.Size([bs, 4, 18, 18])
 
        # 验证1
        # sample = x_0_pred + latents_condition
        # seg_pred = self.ae.decode(sample)
        # 验证2
        seg_pred = self.ae.decode(x_0_pred, latents_condition)
        # 分割损失
        if self.opt.if_BoundaryLoss:
            # 加上了boundary_loss, 在运行程序时要加上 --if_BoundaryLoss
            seg_loss = self.seg_loss_fn(self.opt, seg_pred, bev_labels, t, self.num_timesteps)
        else:
            # 没有boundary_loss
            seg_loss = self.seg_loss_fn(self.opt, seg_pred, bev_labels)


        if self.current_epoch % 10 == 0 and self.tensorboard_flag==True:
            self.showIntermediate(latents, 'latents', mode='val')
            self.showIntermediate(latents_condition, 'latents_cond', mode='val')
            self.showIntermediate(pred_noise, 'pred_noise', t=t, mode='val')
            self.showIntermediate(x_0_pred, 'x_0_pred', t=t, mode='val')
            self.tensorboard_flag=False
        
        loss = noise_loss+seg_loss

        self.log('Step_loss/val',loss, on_step=True, prog_bar=True, logger=True)
        self.log('Step_loss/val_noise', noise_loss, on_step=True, prog_bar=True, logger=True)
        self.log('Step_loss/val_seg', seg_loss, on_step=True, prog_bar=True, logger=True)
        print('!!!finish the val step!!!')

        return loss

    def test_step(self, batch, batch_idx):
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        
        with torch.no_grad():
            FV_img = images.squeeze(dim=1)
            BEV_img = bev_images
            out=self(FV_img, mode='test', verbose=True) # torch.Size([4, 7, 150, 150])

            # 显示
            if batch_idx == 0:
                self.showPred(FV_img, out, BEV_img, mode='Test')

            # 保存图像
            pred_img = out.clone()
            pred_save_folder = os.path.join(save_root, create_time)
            pred_color = get_visual_img(imgs = pred_img)
            for i in range(FV_img.shape[0]):
                scene_name = scene_names[i]
                img_name = img_names[i]
                img_path = pred_save_folder + "/" + scene_name + "/"
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                skimage.io.imsave(img_path + img_name + '_nn_pred_c.png', pred_color[i].cpu().detach().numpy().transpose((1, 2, 0)))
            
            # 计算混淆矩阵
            if self.opt.if_Vshape: # 加mask
                label, pred = get_mask(intrins, out, bev_labels, self.opt)
            else: # 不加mask
                label = bev_labels.to(torch.int64).cpu().numpy().squeeze().flatten()
                pred = out.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4,5,6])

        return conf

    
    def showIntermediate(self, feature_maps, title, t=None, mode='train'):
        feature_maps = feature_maps.detach().cpu()
        n_show = 4
        num = min(n_show, feature_maps.shape[0])
        
        if title == 'latents' or title == 'x_0_pred' or 'pred_xstart_step' in title:
            # import pdb; pdb.set_trace()
            x_0_upsampling = torch.nn.functional.interpolate(feature_maps, (37, 37), mode='bicubic').to(self.device)
            x_0_upsampling = self.ae.decode(x_0_upsampling)
            pred_color = get_visual_img(imgs = x_0_upsampling) # pred_img是list，pred_img[0].shape: torch.Size([3, 150, 150])
            grid_image = torchvision.utils.make_grid(pred_color[:num], 4, normalize=False)
            if t == None:
                self.logger.experiment.add_image('{}_{}/pred_img_{}_epoch:{}'.format(mode, 'showIntermediate', title, self.current_epoch), grid_image, self.current_epoch, dataformats="CHW")
            else:
                self.logger.experiment.add_image('{}_{}/pred_img_{}_epoch:{}(t:{})'.format(mode, 'showIntermediate', title, self.current_epoch, t[:4]), grid_image, self.current_epoch, dataformats="CHW")

        for n in range(num):
            feat_lists = [feat_map.unsqueeze(dim=0) for feat_map in feature_maps[n][:]]
            grid_image = torchvision.utils.make_grid(feat_lists, 4, padding=20, normalize=True, scale_each=True, pad_value=1)
            if t == None:
                self.logger.experiment.add_image('{}/{}_Fea_map_{}'.format(mode, title, n), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")
            else:
                self.logger.experiment.add_image('{}/{}_Fea_map_{}(t={})'.format(mode, title, n, t[n]), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")


    
    # GS 
    def showPred(self, FV_img, pred_img, BEV_img, mode):
        # import pdb; pdb.set_trace()
        n_show = 20
        num = min(n_show, FV_img.shape[0])

        grid_image = torchvision.utils.make_grid(FV_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/FV_image'.format(mode), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")
        
        pred_color = get_visual_img(imgs = pred_img) # pred_img是list，pred_img[0].shape: torch.Size([3, 150, 150])
        grid_image = torchvision.utils.make_grid(pred_color[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/pred_img'.format(mode), grid_image, self.current_epoch, dataformats="CHW")
        
        grid_image = torchvision.utils.make_grid(BEV_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/BEV_image'.format(mode), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")

    def calculate_metrics(self, conf_total, mode='Train'):
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()

        self.logger.experiment.add_scalar('Metric_{}/average_IoU'.format(mode), average_IoU, self.current_epoch)
        self.logger.experiment.add_scalar('Metric_{}/average_precision'.format(mode), average_precision, self.current_epoch)

        # 打印结果
        print_output = ("Epoch: [{epoch}] | mIoU: {mIoU:.8f} |  mAP: {mAP:.8f} | mRecall: {mRecall:.8f} |".format(
                        epoch=self.current_epoch, mIoU=average_IoU, mAP=average_precision, mRecall=average_recall))
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}    
        for i in range(len(iou_per_class)):  
            precision_record[self.opt.label_list[i]] = precision_per_class[i]
            recall_record[self.opt.label_list[i]] = recall_per_class[i]
            iou_record[self.opt.label_list[i]] = iou_per_class[i]
        metirc_each_class = ("precision for each class: {} | recall for each class: {} | iou for each class: {}\n".format(precision_record, recall_record, iou_record))
        
        self.logger.experiment.add_text('{}/loss'.format(mode), print_output, self.current_epoch)  # 结果写入文件
        self.logger.experiment.add_text('{}/logger'.format(mode), metirc_each_class, self.current_epoch)
        
        # text_log
        text_logger.write('**************{}**************\n'.format(mode))
        text_logger.write(print_output + '\n')
        text_logger.write(metirc_each_class + '\n')
        text_logger.flush()
        return average_IoU, average_precision, average_recall
    
    def training_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)

        conf_list = []
        if self.current_epoch % 50 == 0:
            train_dataloader = self.train_dataloader()
            
            batch = next(iter(train_dataloader))
            (bev_labels, images, bev_images, img_names, scene_names,\
                rots, trans, intrins, post_rots, post_trans) = batch
            
            FV_img = images.squeeze(dim=1)  # torch.Size([3, 256, 512])  # FV image
            BEV_img = bev_images  # torch.Size([3, 150, 150])  # BEV image
            

            # import pdb; pdb.set_trace()
            out=self(FV_img, mode='train', verbose=True)  # torch.Size([4, 7, 150, 150])

            self.showPred(FV_img, out, BEV_img, mode='Train')

            # 计算混淆矩阵
            label = bev_labels.to(torch.int64).cpu().numpy().squeeze().flatten()
            pred = out.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4,5,6])
            conf_list.append(conf)

            conf_total = np.stack([x for x in conf_list]).sum(axis=0)
            average_IoU, average_precision, average_recall = self.calculate_metrics(conf_total, mode='Train')

    # def validation_epoch_end(self, outputs):
    #     # import pdb; pdb.set_trace()
    #     avg_loss = torch.stack([x for x in outputs]).mean()
    #     self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)

    #     val_dataloader = self.val_dataloader()

        
    #     batch = next(iter(val_dataloader))
    #     (bev_labels, images, bev_images, img_names, scene_names,\
    #         rots, trans, intrins, post_rots, post_trans) = batch
        
    #     FV_img = images.squeeze(dim=1)  # FV image
    #     BEV_img = bev_images  # torch.Size([3, 150, 150])  # BEV image
        
    #     # batch_input=torch.stack(4*[FV_img],0)  # torch.Size([4, 3, 256, 512])

    #     # import pdb; pdb.set_trace()
    #     out=self(FV_img, mode='val', verbose=True)  # torch.Size([bs, 7, 150, 150])

    #     self.showPred(FV_img, out, BEV_img, mode='Val')

    def validation_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        print('1 in validation epoch end...')
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)

        val_dataloader = self.val_dataloader()

        # if self.current_epoch % self.sample_interval == 0:
        if self.current_epoch > 138:
            print('2 in if, in validation epoch end...')
            conf_list = []
            for batch_idx, batch in tqdm.tqdm(enumerate(val_dataloader),total =len(val_dataloader)):
                (bev_labels, images, bev_images, img_names, scene_names,\
                    rots, trans, intrins, post_rots, post_trans) = batch
                if images.shape[0] != self.batch_size:
                    break
                
                FV_img = images.squeeze(dim=1)
                BEV_img = bev_images
                out=self(FV_img, mode='val', verbose=True) # torch.Size([4, 7, 150, 150])

                # 显示
                if batch_idx == 0:
                    self.showPred(FV_img, out, BEV_img, mode='Val')
                # 计算混淆矩阵
                if self.opt.if_Vshape: # 加mask
                    label, pred = get_mask(intrins, out, bev_labels, self.opt)
                else: # 不加mask
                    label = bev_labels.to(torch.int64).cpu().numpy().squeeze().flatten()
                    pred = out.argmax(1).cpu().numpy().squeeze().flatten()
                conf = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4,5,6])
                conf_list.append(conf)
            conf_total = np.stack([x for x in conf_list]).sum(axis=0)
            average_IoU, average_precision, average_recall = self.calculate_metrics(conf_total, mode='Val')
            self.log("val_iou", average_IoU, on_epoch=True, prog_bar=True, logger=True)
        else:
            print('3 in else, in validation epoch end...')
            batch = next(iter(val_dataloader))
            (bev_labels, images, bev_images, img_names, scene_names,\
                rots, trans, intrins, post_rots, post_trans) = batch
            
            FV_img = images.squeeze(dim=1)  # FV image
            BEV_img = bev_images  # torch.Size([3, 150, 150])  # BEV image

            # import pdb; pdb.set_trace()
            out=self(FV_img, mode='val', verbose=True)  # torch.Size([bs, 7, 150, 150])

            self.showPred(FV_img, out, BEV_img, mode='Val')
            average_IoU = 0.0
            self.log("val_iou", average_IoU, on_epoch=True, prog_bar=True, logger=True)
    
    def test_epoch_end(self, outputs):
        conf_total = np.stack([x for x in outputs]).sum(axis=0)
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()

        self.log("test_iou", average_IoU, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('Metric_Test/average_IoU', average_IoU, self.current_epoch)
        self.logger.experiment.add_scalar('Metric_Test/average_precision', average_precision, self.current_epoch)

        # 打印结果
        print_output = ("Test Epoch: [{epoch}] | mIoU: {mIoU:.8f} |  mAP: {mAP:.8f} | mRecall: {mRecall:.8f} |".format(
                        epoch=self.current_epoch, mIoU=average_IoU, mAP=average_precision, mRecall=average_recall))
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}
        for i in range(len(iou_per_class)):  
            precision_record[self.opt.label_list[i]] = precision_per_class[i]
            recall_record[self.opt.label_list[i]] = recall_per_class[i]
            iou_record[self.opt.label_list[i]] = iou_per_class[i]
        metirc_each_class = ("precision for each class: {} | recall for each class: {} | iou for each class: {}".format(precision_record, recall_record, iou_record))

        self.logger.experiment.add_text('Test/loss', print_output, self.current_epoch)  # 结果写入文件
        self.logger.experiment.add_text('Test/logger', metirc_each_class , self.current_epoch)

        # text_log
        text_logger.write('**************Test**************\n')
        text_logger.write(print_output + '\n')
        text_logger.write(metirc_each_class + '\n')
        text_logger.flush()
        