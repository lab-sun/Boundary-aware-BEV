import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.models import AutoencoderKL
from pytorch_lightning.loggers import TensorBoardLogger
from BEVDiff_dataset import SimpleImageDataset, collate_fn
import torchvision.utils
from utils import get_visual_img, compute_losses

from .DenoisingDiffusionProcess import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type= "stabilityai/sd-vae-ft-ema"#@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        # self.model=AutoencoderKL.from_pretrained(model_type)
        pretrain_weigth = "./pretrain_weights/ae/2023-10-10-10-45/ae_epoch19.pth"
        self.model = torch.load(pretrain_weigth).to('cpu')
        
    def forward(self,input):
        return self.model(input).sample
    
    def encode(self,input,mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample

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
        
        self.ae=AutoEncoder().to('cpu')
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,256,256)).shape[1]
        self.model=DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                             num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        #return self.output_T(self.model(*args,**kwargs))
        pred = self.model(*args,**kwargs)
        # self.ae.to('cpu')
        # pred = pred.to('cpu')
        # latent_scale_factor = self.latent_scale_factor.to('cpu')
        output = self.ae.decode(pred/self.latent_scale_factor)
        result = self.output_T(output)
        #return self.output_T(self.ae.decode(self.model(*args,**kwargs)/self.latent_scale_factor))
        return result
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):   
        (bev_labels, images, bev_images, img_names, scene_names) = batch
        latents=self.ae.encode(self.input_T(bev_images)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        # import pdb; pdb.set_trace()
        (bev_labels, images, bev_images, img_names, scene_names) = batch
        latents=self.ae.encode(self.input_T(bev_images)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('val_loss',loss)
        
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
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)
        else:
            return None
    
    # GS 
    def showPred(self, pred_img, mode):
        # import pdb; pdb.set_trace()
        n_show = 20
        num = min(n_show, pred_img.shape[0])
        
        pred_color = get_visual_img(imgs = pred_img) # pred_img是list，pred_img[0].shape: torch.Size([3, 150, 150])
        grid_image = torchvision.utils.make_grid(pred_color[:num], 4, normalize=False)
        self.logger.experiment.add_image('{}/pred_img'.format(mode), grid_image, self.current_epoch, dataformats="CHW")
        
    def training_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        
        # import pdb; pdb.set_trace()
        out=self(batch_size=1,shape=(150,150),verbose=True)

        # TODO ?检查FV_img是几张图像？
        self.showPred(out, mode='Train')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        
        # import pdb; pdb.set_trace()
        out=self(batch_size=1,shape=(150,150),verbose=True)

        # TODO ?检查FV_img是几张图像？
        self.showPred(out, mode='Val')

    
    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
    
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
            self.latent_dim=self.ae.encode(torch.ones(1,3,256,256)).shape[1]
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                        condition_channels=self.latent_dim,
                                                        num_timesteps=num_timesteps)
        
            
    @torch.no_grad()
    def forward(self,condition,*args,**kwargs):  # 在训练完设定的step步后，采样时调用
        # import pdb; pdb.set_trace()
        condition_latent=self.ae.encode(self.input_T(condition.to(self.device))).detach()*self.latent_scale_factor # condition_latent:torch.Size([4, 4, 32, 32])
        
        # sampling
        output_code=self.model(condition_latent,*args,**kwargs)/self.latent_scale_factor  # output_code: torch.Size([4, 4, 32, 32])

        return self.output_T(self.ae.decode(output_code))
    
    def training_step(self, batch, batch_idx):  # 先调用
        # import pdb; pdb.set_trace()  
        print('in train')
        condition,output=batch  # condition: torch.Size([1, 3, 256, 256]); output:torch.Size([1, 3, 256, 256])
                
        with torch.no_grad(): # 不求导？是用已经训练好的模型做autoencoder？再在forward中对diffusion模型进行参数训练？
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor  # latent.shape:torch.Size([1, 4, 32, 32])
            latents_condition=self.ae.encode(self.input_T(condition)).detach()*self.latent_scale_factor  # latents_condition.shape:torch.Size([1, 4, 32, 32])
        loss = self.model.p_loss(latents, latents_condition)
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        print('in val')
        # import pdb; pdb.set_trace()
        condition,output=batch
        
        with torch.no_grad():
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor  # torch.Size([1, 4, 32, 32])
            latents_condition=self.ae.encode(self.input_T(condition)).detach()*self.latent_scale_factor  # torch.Size([1, 4, 32, 32])
        loss = self.model.p_loss(latents, latents_condition)
        
        self.log('val_loss',loss)
        
        return loss