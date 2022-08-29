"""
Transgrow GAN classic optimization of KL divergence
Whole sequence goes into discriminator
"""

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import timm
import random
import copy

from models.transgrow_generator import TransGrowGenerator
from models.discriminator_networks import DCGANDiscriminator, NLayerDiscriminator
from utils.ms_ssim import MSSSIM
from utils.vgg_perceptual_loss import VGGPerceptualLoss


class TransGrowGANModel(pl.LightningModule):
    def __init__(self, data_shape, g_enc_net, g_enc_net_pretrained, g_dec_net, g_dec_net_pretrained, pe_max_len, pe_fusion_type, dim_img, dim_pe, dim_z, dim, depth, heads, dropout_trf, dropout_emb, prd_token_type, d_net, d_net_pretrained, lr, target_frame, losses_w, final_actvn):
        '''
        Parameters
        ----------
        data_shape : size object of torch module.
            dimension: [S,C,W,H]. Note: S is overrall sequence length (in+out)
        g_enc_net : str
            Name of geneartor encoder network
        g_enc_net_pretrained : bool
            Use pretrained geneartor encoder network?
        g_dec_net : str
            Name of geneartor decoder network
        g_dec_net_pretrained : bool
            Use pretrained geneartor decoder network?
        pe_max_len : int
            maximal length of positional encoding (i.e. max time value of data set)
        pe_fusion_type : str
            How to fuse positional encoding to img embedding: 'add' = addition (DEFAULT), 'cat' = concatenation
        dim_img : int
            dimension of img embedding
        dim_pe : int
            dimension positional encoding
        dim_z : int
            dimension stochasticity
        dim : int
            dimension of transformer
        depth : int
            depth of transformer
        heads : int
            number of attention heads of transformer
        dropout_trf : float
                in range [0 1] to control dropout in transformer encoder layer
        dropout_emb : float
            in range [0 1] to control dropout after encoding.
        prd_token_type : str
            if "mean": average over transformer output, else: only use pred_token
        d_net : str
            Name of discriminator network
        d_net_pretrained : bool
            Use pretrained discriminator?
        lr : float
            learning rate.
        target_frame : int
            specify target frame (usual: last). If None: random during training
        losses_w : dict
            contains weights of losses
        final_actvn : str
            indicates the final image activation


        Returns
        -------
        None.

        '''
        super().__init__()
        self.save_hyperparameters()
        self.data_shape = data_shape
        self.s = data_shape[0] 
        self.s_in = self.s-1
        self.c = data_shape[1]
        self.w = data_shape[2]
        self.h = data_shape[3]
        self.pe_max_len = pe_max_len
        self.pe_fusion_type = pe_fusion_type
        self.dim_img = dim_img
        self.dim_pe = dim_pe
        self.dim_z = dim_z
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout_trf = dropout_trf
        self.dropout_emb = dropout_emb
        self.prd_token_type = prd_token_type
        self.lr = lr
        self.target_frame = target_frame
        self.losses_w = losses_w
        self.final_actvn = final_actvn
        
        # # Generator
        self.generator = TransGrowGenerator(data_shape, g_enc_net, g_enc_net_pretrained, g_dec_net, g_dec_net_pretrained, pe_max_len, pe_fusion_type, dim_img, dim_pe, dim_z, dim, depth, heads, dropout_trf, dropout_emb, prd_token_type)
        print('Generator: Total params:', sum(p.numel() for p in self.generator.parameters()))
        
        # # Diskriminator
        if d_net == 'res18':
            self.discriminator = timm.create_model('resnet18', num_classes=1, pretrained=d_net_pretrained)
            self.discriminator.conv1 = torch.nn.Conv2d(self.c*self.s, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        elif d_net == 'res50':
            self.discriminator = timm.create_model('resnet50', num_classes=1, pretrained=d_net_pretrained)
            self.discriminator.conv1 = torch.nn.Conv2d(self.c*self.s, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        elif d_net == 'patchGAN':
            self.discriminator = NLayerDiscriminator(self.c*self.s)
        elif d_net == 'dcgan':
            self.discriminator = DCGANDiscriminator(self.c*self.s, 32, self.w)
        print('Discriminator: Total params:', sum(p.numel() for p in self.discriminator.parameters()))
        
        # # activations
        if self.final_actvn == 'relu':
            self.actvn = torch.nn.ReLU()
        elif self.final_actvn == 'sigmoid':
            self.actvn = torch.nn.Sigmoid()
        elif self.final_actvn == 'tanh':
            self.actvn=torch.nn.Tanh()
        else:
            print('Wrong final_actvn.')
        
        # # losses
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_mse = torch.nn.MSELoss()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_msssim = MSSSIM(self.c, window_size=11, size_average=True)
        self.loss_percep = VGGPerceptualLoss()
    
        # # example input: this allows the trainer to show input and output sizes in the report (12 is just a sample batch size)
        self.example_input_array = {'x': 
                                    {'img_in': torch.zeros(12, data_shape[0]-1, data_shape[1], data_shape[2], data_shape[3]), 
                                     'timedelta_in': torch.zeros(12, data_shape[0]-1).long(),
                                     'timedelta_target': torch.zeros(12).long()}}
        
    
    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [optimizer_g, optimizer_d]
    
    
    def gan_criterion(self, img_pred, img_target):
        loss_bce = self.loss_bce(img_pred, img_target)
        return loss_bce
    
    
    def l1_criterion(self, img_pred, img_target):
        loss_l1 = self.loss_l1(img_pred, img_target)
        return loss_l1
    
    
    def _forward(self, img_in, timedelta_in, timedelta_target, mask=None, z=None):
        x  = self.generator(img_in, timedelta_in, timedelta_target, mask=None, z=z)
        return x
    
    
    def training_step(self, batch, batch_nb, optimizer_idx):
        # create idx tuple of "in" and "target" frames
        if self.target_frame > self.s or self.target_frame <=0 or self.target_frame is None:
             idx_target = random.randint(0,self.s_in)
        else:
            idx_target = self.target_frame-1
        idx_in = tuple(list(range(self.s))[:idx_target]+list(range(self.s))[idx_target+1:])
        
        # # original img_seq
        img_seq = batch['seq_img']
        # # img_in & timedelta_in                
        img_in = batch['seq_img'][:,idx_in,:]
        timedelta_in = batch['seq_timedelta'][:,idx_in]
        # # img_target % timedelta_target
        img_target = batch['seq_img'][:,idx_target,:]
        timedelta_target = batch['seq_timedelta'][:,idx_target]
        # # run forward pass with img_in, timedelta_in, timedelta_target 
        logits_pred = self._forward(img_in, timedelta_in, timedelta_target)
        # # activate output 
        img_pred = self.actvn(logits_pred)
        # # get img_seq_pred were the target position is replaced with fake img
        img_seq_pred = copy.deepcopy(img_seq)
        img_seq_pred[:,idx_target,:] = img_pred
        # # change view of img_seq and img_seq_pred (concatenate imgs of sequence along the channel dimension)
        img_seq = img_seq.view(img_seq.shape[0],-1, img_seq.shape[3], img_seq.shape[4])
        img_seq_pred = img_seq_pred.view(img_seq_pred.shape[0],-1, img_seq_pred.shape[3], img_seq_pred.shape[4])
        
        # # train generator
        if optimizer_idx == 0:
            # # run discriminator with fake images
            fake_pred = self.discriminator(img_seq_pred)
            # # compute gan loss
            valid = torch.ones_like(fake_pred)
            g_loss = self.gan_criterion(self.actvn(fake_pred), valid)
            self.log('loss_g', g_loss)
            # # compute l1_loss
            l1_loss = self.l1_criterion(img_pred, img_target)
            self.log('loss_l1', l1_loss)
            # # compute msssim loss
            loss_msssim = self.loss_msssim(img_pred, img_target)
            self.log("loss_msssim", loss_msssim)
            # # compute perceptual loss
            loss_percep = self.loss_percep(img_pred, img_target)
            self.log("loss_percep", loss_percep)
            
            # # combine losses
            loss = self.losses_w['weight_adv'] * g_loss + self.losses_w['weight_l1'] * l1_loss + self.losses_w['weight_msssim'] * loss_msssim + self.losses_w['weight_percep'] * loss_percep
            
            return loss
        
        # train discriminator    
        elif optimizer_idx == 1:
            # discriminator_real loss
            real_pred = self.discriminator(img_seq)
            valid = torch.ones_like(real_pred)
            real_loss = self.gan_criterion(self.actvn(real_pred), valid)   
            
            # discriminator fake loss
            fake_pred = self.discriminator(img_seq_pred)
            fake = torch.zeros_like(fake_pred)
            fake_loss = self.gan_criterion(self.actvn(fake_pred), fake)
            loss = (real_loss + fake_loss) / 2
            
            self.log("loss_d_real", real_loss)
            self.log("loss_d_fake", fake_loss)
            self.log("loss_d", loss)
            
        return loss
    
    
    def validation_step(self, batch, batch_nb):
        # create idx tuple of "in" and "target" frames
        if self.target_frame > self.s or self.target_frame <=0 or self.target_frame is None:
             idx_target = random.randint(0,self.s_in)
        else:
            idx_target = self.target_frame-1
        idx_in = tuple(list(range(self.s))[:idx_target]+list(range(self.s))[idx_target+1:])
        
        # # original img_seq
        img_seq = batch['seq_img']
        # # img_in & timedelta_in                
        img_in = batch['seq_img'][:,idx_in,:]
        timedelta_in = batch['seq_timedelta'][:,idx_in]
        # # img_target % timedelta_target
        img_target = batch['seq_img'][:,idx_target,:]
        timedelta_target = batch['seq_timedelta'][:,idx_target]

        # # run forward pass with img_in, timedelta_in, timedelta_target 
        logits_pred = self._forward(img_in, timedelta_in, timedelta_target)
        # # activate output
        img_pred = self.actvn(logits_pred)
        
        # # get img_seq_pred were the target position is replaced with fake img
        img_seq_pred = copy.deepcopy(img_seq)
        img_seq_pred[:,idx_target,:] = img_pred
        # # change view of img_seq and img_seq_pred (concatenate imgs of sequence along the channel dimension)
        img_seq = img_seq.view(img_seq.shape[0],-1, img_seq.shape[3], img_seq.shape[4])
        img_seq_pred = img_seq_pred.view(img_seq_pred.shape[0],-1, img_seq_pred.shape[3], img_seq_pred.shape[4])
        
        # run discriminator with fake images
        fake_pred = self.discriminator(img_seq_pred)
        # compute gan and l1 and msssim loss
        valid = torch.ones_like(fake_pred)
        
        g_loss = self.gan_criterion(self.actvn(fake_pred), valid)
        self.log('val_loss_g', g_loss)
        # # compute l1_loss
        l1_loss = self.l1_criterion(img_pred, img_target)
        self.log('val_loss_l1', l1_loss)
        # # compute msssim loss
        loss_msssim = self.loss_msssim(img_pred, img_target)
        self.log("val_loss_msssim", loss_msssim)
        # # compute perceptual loss
        loss_percep = self.loss_percep(img_pred, img_target)
        self.log("val_loss_percep", loss_percep)
        
        # # save here the metric that should define the best model. Important for ModelChecker: log name: 'val_loss'
        self.log('val_loss', self.losses_w['weight_l1'] * l1_loss + self.losses_w['weight_msssim'] * loss_msssim + self.losses_w['weight_percep'] * loss_percep, on_step=False, on_epoch=True)
        
        return img_pred
    
    
    def validation_epoch_end(self,val_step_outputs):
        for count, out in enumerate(val_step_outputs):
            # # log sampled images
            if count>0:
                break
            if self.final_actvn=='tanh':
                value_range=(-1,1)
            else:
                value_range=(0,1)
            grid = torchvision.utils.make_grid(out[:4,:],normalize=True,value_range=value_range)
            self.logger.experiment.add_image('val_gen_images', grid,0)
            
    
    def forward(self, x):
        img_in = x['img_in'].to(self.device)
        timedelta_in = x['timedelta_in'].to(self.device)
        timedelta_target = x['timedelta_target'].to(self.device)
        if 'z' in x:
            z = x['z']
        else:
            z = None
        # run forward pass
        logits_pred = self._forward(img_in, timedelta_in, timedelta_target, z=z)
        # in pl forward() is intended for inference, hence the activation is done here
        img_pred = self.actvn(logits_pred)
        
        return img_pred 