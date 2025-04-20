import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from autoencoder_lss import AutoencoderKL
# from autoencoder_condition import AutoencoderKL
from scripts.diffBEV.lss import LiftSplatShoot

from src.DenoisingDiffusionProcess import *
from utils import get_visual_img, compute_losses

from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.utils
from opt import get_args
from BEVDiff_dataset_new import SimpleImageDataset, collate_fn

opt = get_args()

class AutoEncoder(nn.Module):
    def __init__(self,
                # model_type= "stabilityai/sd-vae-ft-ema"#@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                pretrain_weigth = "./pretrain_weights/ae/2023-10-10-10-45/ae_epoch19.pth", 
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        #self.model=AutoencoderKL.from_pretrained(model_type)
        self.model = torch.load(pretrain_weigth).to('cpu')  # autoencoder model
        #self.model = torch.load(pretrain_weigth)
        train_csv_file = './data/nuScenes/train_new.data'
        train_ds = SimpleImageDataset(is_train=False, opt=opt, root_dir=train_csv_file, transform=None)
        cond_pretrain_weight = './pretrain_weights/lss/2023-10-10-22-22/best-lss-epoch40-val_iou0.30.ckpt'  # lss model
        self.model_cond = LiftSplatShoot.load_from_checkpoint(cond_pretrain_weight, opt=opt, train_dataset=train_ds)

        
    def forward(self,input):
        return self.model(input).sample
    
    def encode(self,input,mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    def encode_cond(self,input, rots, trans, intrins, post_rots, post_trans,mode=False):
        # 自己训练的模型.latent_dist怎么处理?
        cond_encoded=self.model_cond.encoder_step(input, rots, trans, intrins, post_rots, post_trans)
        return cond_encoded  # torch.Size([2, 32, 75, 75])
    
    def decode(self,input):
        return self.model_cond.decoder_step(input) # lss的decoder TODO lss decoder的in_channel是32，现在latent+cond_latent的通道为36
        # return self.model.decode(input, H=150, W=150).sample  # autoencoder的decoder
        

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
        self.model=DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                             num_timesteps=num_timesteps)

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
        (bev_labels, images, bev_images, img_names, scene_names) = batch
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
        (bev_labels, images, bev_images, img_names, scene_names) = batch
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
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)

    # GS 
    def training_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        # print('in BEVDiff_LatentDiffusion.py, in training_epoch_end, type of outputs:{}, len: {}'.format(type(outputs), len(outputs)))
    
class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        pl.LightningModule.__init__(self)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        
        self.ae=AutoEncoder()
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,300,300)).shape[1]  # 4
            self.cond_latent_dim = 32 # 32
        
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                        condition_channels=self.cond_latent_dim,
                                                        num_timesteps=num_timesteps)
        # TODO 添加对FV的encoder, self.ae.model_cond是提取FV的网络
        self.seg_loss_fn = compute_losses()
        self.val_iou = 0

        self.out = nn.Conv2d(36, 32, kernel_size=1, stride=1, padding=0, bias=False)
        
            
    @torch.no_grad()
    def forward(self,batch,*args,**kwargs):  # 在训练完设定的step步后，采样时调用
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        condition_latent=self.ae.encode_cond(self.input_T(images.to(self.device)), rots, trans, intrins, post_rots, post_trans).detach()*self.latent_scale_factor # condition_latent:torch.Size([4, 4, 32, 32])
        latents_condition = torch.nn.functional.interpolate(condition_latent, (32, 32))
        # sampling
        # import pdb; pdb.set_trace()
        output_code=self.model(latents_condition,*args,**kwargs)/self.latent_scale_factor  # output_code: torch.Size([8, 4, 32, 32])
        # 将返回结果与cond_latent相加
        output_result = torch.cat([output_code,latents_condition],1).to(self.device) # torch.Size([8, 36, 32, 32])
        output_result = torch.nn.functional.interpolate(output_result, (38, 38))  # 用lss的decode需要(bs, 32, 38, 38)大小的输入
        output_result = self.out(output_result)  # torch.Size([8, 32, 38, 38])

        

        # return self.output_T(self.ae.decode(output_code))
        return self.output_T(self.ae.decode(output_result))
        # return self.output_T(self.ae.decode(condition_latent))
    
    def training_step(self, batch, batch_idx):  # 先调用
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        bev_labels = bev_labels.to(torch.int64)
        condition = images.float()  # torch.Size([1, 3, 256, 512])
        output = bev_images.float()  # torch.Size([1, 3, 150, 150])
                
        # import pdb; pdb.set_trace()
        with torch.no_grad(): # 不求导？是用已经训练好的模型做autoencoder？再在forward中对diffusion模型进行参数训练？
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor  # latent.shape:torch.Size([1, 4, 32, 32])
            # TODO 检查是否需要detach ae的encoder
            latents_condition=self.ae.encode_cond(self.input_T(condition), rots, trans, intrins, post_rots, post_trans).detach()*self.latent_scale_factor  # latents_condition.shape:torch.Size([1, 4, 32, 32])
        latents = torch.nn.functional.interpolate(latents, (32, 32))
        latents_condition = torch.nn.functional.interpolate(latents_condition, (32, 32))
        
        noise_loss, x_0_pred = self.model.p_loss(latents, latents_condition) # x_0_pred.shape:torch.Size([bs, 4, 18, 18])
        
        # import pdb; pdb.set_trace()
        output_result = torch.cat([x_0_pred,latents_condition],1).to(self.device) # torch.Size([8, 36, 32, 32])
        output_result = torch.nn.functional.interpolate(output_result, (38, 38))  # 用lss的decode需要(bs, 32, 38, 38)大小的输入
        output_result = self.out(output_result)  # torch.Size([8, 32, 38, 38])

        # seg_pred = self.output_T(self.ae.decode(x_0_pred/self.latent_scale_factor))  # ?需要除self.latent_scale_factor嘛？
        seg_pred = self.output_T(self.ae.decode(output_result/self.latent_scale_factor))
        seg_loss = self.seg_loss_fn(opt, seg_pred, bev_labels)  # seg_pred.shape:torch.Size([8, 7, 150, 150]); bev_labels.shape:torch.Size([8, 150, 150])
        loss = noise_loss+seg_loss
        #loss = noise_loss
        
        # self.log('train_loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Step_loss/train',loss, on_step=True, prog_bar=True, logger=True)
        self.log('Step_loss/train_noise', noise_loss, on_step=True, prog_bar=True, logger=True)
        self.log('Step_loss/train_seg', seg_loss, on_step=True, prog_bar=True, logger=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx): 
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        condition = images.float()  # torch.Size([2, 1, 3, 128, 256])  # TODO 需不需要squeeze？
        output = bev_images.float()  # torch.Size([2, 3, 300, 300])
        
        with torch.no_grad():
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor  # torch.Size([2, 4, 37, 37])
            latents_condition=self.ae.encode_cond(self.input_T(condition), rots, trans, intrins, post_rots, post_trans).detach()*self.latent_scale_factor  # torch.Size([2, 32, 38, 38])
        # 统一latents和latents_condition的大小
        latents = torch.nn.functional.interpolate(latents, (32, 32))
        latents_condition = torch.nn.functional.interpolate(latents_condition, (32, 32))
        loss, x_0_pred = self.model.p_loss(latents, latents_condition)  # x_0_pred:torch.Size([2, 4, 32, 32]) TODO 检查输出的通道数是否影响结果
        
        self.log('val_loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_iou", self.val_iou, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # GS 
    def showPred(self, FV_img, pred_img, BEV_img, mode):
        # import pdb; pdb.set_trace()
        # FV_img
        n_show = 20
        num = min(n_show, FV_img.shape[0])

        grid_image = torchvision.utils.make_grid(FV_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/FV_image'.format(mode), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")
        
        # pred_img
        pred_img = get_visual_img(imgs = pred_img) # pred_img是list，pred_img[0].shape: torch.Size([3, 150, 150])
        grid_image = torchvision.utils.make_grid(pred_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/pred_img'.format(mode), grid_image, self.current_epoch, dataformats="CHW")
        
        grid_image = torchvision.utils.make_grid(BEV_img[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/BEV_image'.format(mode), torch.Tensor.cpu(grid_image),self.current_epoch,dataformats="CHW")

    def training_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)

        train_dataloader = self.train_dataloader()
        batch = next(iter(train_dataloader))

        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        bev_labels = bev_labels.to(self.device)
        images = images.to(self.device)
        bev_images = bev_images.to(self.device)
        rots = rots.to(self.device)
        trans = trans.to(self.device)
        intrins = intrins.to(self.device)
        post_rots = post_rots.to(self.device)
        post_trans = post_trans.to(self.device)
        batch_input = []
        batch_input = [bev_labels, images, bev_images, img_names, scene_names, rots, trans, intrins, post_rots, post_trans]

        FV_img = images.squeeze(1)  # torch.Size([3, 256, 512])  # FV image
        BEV_img = bev_images  # torch.Size([3, 150, 150])  # BEV image
        
        # import pdb; pdb.set_trace()
        # TODO forward需要rot等参数，self的输入应改成batch，与此同时保持对同一个输入有4个预测输出
        out=self(batch_input, verbose=True)  # torch.Size([4, 7, 150, 150])

        self.showPred(FV_img, out, BEV_img, mode='Train')

    def validation_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)

        val_dataloader = self.val_dataloader()
        batch = next(iter(val_dataloader))
        
        (bev_labels, images, bev_images, img_names, scene_names,\
             rots, trans, intrins, post_rots, post_trans) = batch
        bev_labels = bev_labels.to(self.device)
        images = images.to(self.device)
        bev_images = bev_images.to(self.device)
        rots = rots.to(self.device)
        trans = trans.to(self.device)
        intrins = intrins.to(self.device)
        post_rots = post_rots.to(self.device)
        post_trans = post_trans.to(self.device)
        batch_input = []
        batch_input = [bev_labels, images, bev_images, img_names, scene_names, rots, trans, intrins, post_rots, post_trans]
        
        # FV_img = images[0]  # torch.Size([3, 256, 512])  # FV image
        # BEV_img = bev_images[0]  # torch.Size([3, 150, 150])  # BEV image
        
        # batch_input=torch.stack(4*[FV_img],0)  # torch.Size([4, 3, 256, 512])

        out=self(batch_input, verbose=True)  # torch.Size([4, 7, 150, 150])

        FV_img = images.squeeze(1)
        BEV_img = bev_images
        self.showPred(FV_img, out, BEV_img, mode='Val')

